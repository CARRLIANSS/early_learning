import torch
from utils.utils import tocuda
from utils.test_utils import estimate_pose_norm_kpts, estimate_pose_from_E, compute_pose_error, pose_auc
import numpy as np
import os
from tqdm import tqdm
from models.loss import MatchLoss
from utils.distributed_utils import is_main_process, reduce_value
import sys


def train_one_epoch(model, optimizer, data_loader, conf, device, epoch, logger_train, cur_global_step):
    model.train()

    loss_function = MatchLoss(conf)
    mean_loss = torch.zeros(1).to(device)
    optimizer.zero_grad()

    # 在进程0中打印训练进度
    if is_main_process():
        data_loader = tqdm(data_loader, file=sys.stdout)

    for step, data in enumerate(data_loader):

        train_data = tocuda(data)

        # run training
        cur_lr = optimizer.param_groups[0]['lr']

        xs = train_data['xs']
        ys = train_data['ys']

        # 求loss用到logits全部向量，和ys_ds的全部距离来求label
        logits, ys_ds, e_hat, y_hat = model(xs, ys)

        loss, ess_loss, classif_loss = loss_function.run(cur_global_step, data, logits, ys_ds, e_hat, y_hat)

        loss.backward()

        loss = reduce_value(loss, average=True)
        mean_loss = (mean_loss * step + loss.detach()) / (step + 1)  # update mean losses

        with torch.no_grad():
            # 多级剪枝后，最后剩下的pairwise的内点率
            is_pos = (ys_ds[-1] < conf.obj_geod_th).type(ys_ds[-1].type())
            is_neg = (ys_ds[-1] >= conf.obj_geod_th).type(ys_ds[-1].type())
            inlier_ratio = torch.sum(is_pos, dim=-1) / (torch.sum(is_pos, dim=-1) + torch.sum(is_neg, dim=-1))
            inlier_ratio = inlier_ratio.mean().item()

        loss_val = [ess_loss, classif_loss, inlier_ratio]
        new_loss_val = []
        for ls in loss_val:
            ls = reduce_value(torch.tensor(ls).to(device), average=True)
            new_loss_val.append(ls.item())

        # 在进程0中打印平均loss
        if is_main_process():
            data_loader.desc = "[epoch {}] mean loss {}".format(epoch, round(mean_loss.item(), 3))
            logger_train.append([cur_lr] + new_loss_val)

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        optimizer.step()
        optimizer.zero_grad()

    # 等待所有进程计算完毕
    if device != torch.device("cpu"):
        torch.cuda.synchronize(device)

    return mean_loss.item()


@torch.no_grad()
def evaluate(model, data_loader, config):
    model.eval()

    err_ts, err_Rs = [], []

    for test_data in tqdm(data_loader, ncols=80):

        xs = test_data['xs'].cuda()
        ys = test_data['ys'].cuda()

        _, _, e_hat, y_hat = model(xs, ys) # y_hat是预测的，ys是gt

        mkpts0 = xs.squeeze()[:, :2].cpu().detach().numpy() # Image1的keypoints
        mkpts1 = xs.squeeze()[:, 2:].cpu().detach().numpy() # Image2的keypoints

        # 提取所有预测对极线距离小于threshold的keypoints
        mask = y_hat.squeeze().cpu().detach().numpy() < config.thr
        mask_kp0 = mkpts0[mask] # 就是算法预测的最可靠的内点
        mask_kp1 = mkpts1[mask]

        if config.use_ransac == True:
            """
            用预测的内点计算R、t，其中使用RANSAC方式生成e_hat（不使用算法预测出来的）
            """
            file_name = '/aucs.txt'
            ret = estimate_pose_norm_kpts(mask_kp0, mask_kp1) # 得到预测的R、t、mask(预测内点中哪些是RANSAC选定的内点)
        else:
            """
            用预测的内点计算R、t，其中使用算法预测e_hat
            """
            file_name = '/aucs_DLT.txt'
            e_hat = e_hat[-1].view(3, 3).cpu().detach().numpy()
            ret = estimate_pose_from_E(mkpts0, mkpts1, mask, e_hat)

        if ret is None:
            err_t, err_R = np.inf, np.inf
        else:
            R, t, inliers = ret # inliers: 通过mask之后剩下的（OANet使用probability向量与threshold作mask）
            R_gt, t_gt = test_data['Rs'], test_data['ts'] # t_gt: [1, 3, 1]
            T_0to1 = torch.cat([R_gt.squeeze(), t_gt.squeeze().unsqueeze(-1)], dim=-1).numpy()
            err_t, err_R = compute_pose_error(T_0to1, R, t)

        err_ts.append(err_t)
        err_Rs.append(err_R)

    # Write the evaluation results to disk.
    out_eval = {'error_t': err_ts, 'error_R': err_Rs}

    # OANet中qt_err
    # 在测试数据中，所有的pose_errors
    pose_errors = []
    for idx in range(len(out_eval['error_t'])):
        pose_error = np.maximum(out_eval['error_t'][idx], out_eval['error_R'][idx])
        pose_errors.append(pose_error)

    thresholds = [5, 10, 20]
    aucs = pose_auc(pose_errors, thresholds) # 计算这几个层次角度误差占比情况
    aucs = [100. * yy for yy in aucs]

    return aucs[0], aucs[1], aucs[2]