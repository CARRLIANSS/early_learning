import torch
from utils.utils import tocuda
from utils.test_utils import denorm, get_pool_result, test_sample, dump_res
import numpy as np
import os
from tqdm import tqdm
from models.loss import MatchLoss
from utils.distributed_utils import is_main_process, reduce_value
import sys

def train_one_epoch(model, optimizer, data_loader, conf, device, epoch, logger_train):
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

        res_logits, res_e_hat = model(train_data)
        loss = 0
        loss_val = []

        for i in range(len(res_logits)):
            loss_i, geo_loss, cla_loss, l2_loss, _, _ = loss_function.run(step, train_data, res_logits[i], res_e_hat[i])
            loss += loss_i
            loss_val += [geo_loss, cla_loss, l2_loss]

        loss.backward()

        loss = reduce_value(loss, average=True)
        mean_loss = (mean_loss * step + loss.detach()) / (step + 1)  # update mean losses

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
def evaluate(model, data_loader, config, cur_global_step, mode):

    model.eval()

    loss_function = MatchLoss(config)

    # save info given by the network
    network_infor_list = ["geo_losses", "cla_losses", "l2_losses", 'precisions', 'recalls', 'f_scores']
    network_info = {info: [] for info in network_infor_list}

    results, pool_arg = [], []
    eval_step, eval_step_i, num_processor = 100, 0, 2

    for test_data in tqdm(data_loader, ncols=80):
        test_data = tocuda(test_data)
        res_logits, res_e_hat = model(test_data)
        y_hat, e_hat = res_logits[-1], res_e_hat[-1]
        loss, geo_loss, cla_loss, l2_loss, prec, rec = loss_function.run(cur_global_step, test_data, y_hat, e_hat)
        info = [geo_loss, cla_loss, l2_loss, prec, rec, 2 * prec * rec / (prec + rec + 1e-15)]
        for info_idx, value in enumerate(info):
            network_info[network_infor_list[info_idx]].append(value)

        if config.use_fundamental:
            # unnorm F
            e_hat = torch.matmul(torch.matmul(test_data['T2s'].transpose(1, 2), e_hat.reshape(-1, 3, 3)),
                                 test_data['T1s'])
            # get essential matrix from fundamental matrix
            e_hat = torch.matmul(torch.matmul(test_data['K2s'].transpose(1, 2), e_hat.reshape(-1, 3, 3)),
                                 test_data['K1s']).reshape(-1, 9)
            e_hat = e_hat / torch.norm(e_hat, dim=1, keepdim=True)

        for batch_idx in range(e_hat.shape[0]):
            test_xs = test_data['xs'][batch_idx].detach().cpu().numpy()
            if config.use_fundamental:  # back to original
                x1, x2 = test_xs[0, :, :2], test_xs[0, :, 2:4]
                T1, T2 = test_data['T1s'][batch_idx].cpu().numpy(), test_data['T2s'][batch_idx].cpu().numpy()
                x1, x2 = denorm(x1, T1), denorm(x2, T2)  # denormalize coordinate
                K1, K2 = test_data['K1s'][batch_idx].cpu().numpy(), test_data['K2s'][batch_idx].cpu().numpy()
                x1, x2 = denorm(x1, K1), denorm(x2, K2)  # normalize coordiante with intrinsic
                test_xs = np.concatenate([x1, x2], axis=-1).reshape(1, -1, 4)

            pool_arg += [(test_xs, test_data['Rs'][batch_idx].detach().cpu().numpy(),
                          test_data['ts'][batch_idx].detach().cpu().numpy(), e_hat[batch_idx].detach().cpu().numpy(),
                          y_hat[batch_idx].detach().cpu().numpy(),
                          test_data['ys'][batch_idx, :, 0].detach().cpu().numpy(), config)]

            eval_step_i += 1
            if eval_step_i % eval_step == 0:
                results += get_pool_result(num_processor, test_sample, pool_arg)
                pool_arg = []
        if len(pool_arg) > 0:
            results += get_pool_result(num_processor, test_sample, pool_arg)

    measure_list = ["err_q", "err_t", "num", 'R_hat', 't_hat']
    eval_res = {}
    for measure_idx, measure in enumerate(measure_list):
        eval_res[measure] = np.asarray([result[measure_idx] for result in results])

    if config.res_path == '':
        config.res_path = os.path.join(config.log_path[:-5], mode)
    tag = "ours" if not config.use_ransac else "ours_ransac"
    ret_val = dump_res(measure_list, config.res_path, eval_res, tag)
    return [ret_val, np.mean(np.asarray(network_info['geo_losses'])), np.mean(np.asarray(network_info['cla_losses'])), \
        np.mean(np.asarray(network_info['l2_losses'])), np.mean(np.asarray(network_info['precisions'])), \
        np.mean(np.asarray(network_info['recalls'])), np.mean(np.asarray(network_info['f_scores']))]
