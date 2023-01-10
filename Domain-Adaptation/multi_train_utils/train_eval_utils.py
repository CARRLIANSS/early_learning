import sys
from tqdm import tqdm
import torch
from multi_train_utils.distributed_utils import reduce_value, is_main_process


def train_one_epoch(model, optimizer, source_loader, target_train_loader, device, epoch, conf):
    model.train()
    mean_loss = torch.zeros(1).to(device)
    optimizer.zero_grad()

    len_source_loader = len(source_loader)
    len_target_loader = len(target_train_loader)
    iter_loader = source_loader if len(source_loader) < len(target_train_loader) else target_train_loader

    if max(len_target_loader, len_source_loader) != 0:
        iter_source, iter_target = iter(source_loader), iter(target_train_loader)

    # 在进程0中打印训练进度
    if is_main_process():
        iter_loader = tqdm(iter_loader, file=sys.stdout)

    for step, _ in enumerate(iter_loader):
        data_source, label_source = next(iter_source) # .next()
        data_target, _ = next(iter_target)  # .next()
        data_source, label_source = data_source.to(device), label_source.to(device)
        data_target = data_target.to(device)

        clf_loss, deepda_loss = model(data_source, label_source, data_target)
        loss = clf_loss + conf.deepda_loss_weight * deepda_loss

        loss.backward()
        loss = reduce_value(loss, average=True)
        mean_loss = (mean_loss * step + loss.detach()) / (step + 1)  # update mean losses

        # 在进程0中打印平均loss
        if is_main_process():
            iter_loader.desc = "[epoch {}] mean loss {}".format(epoch, round(mean_loss.item(), 3))

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
def evaluate(model, target_test_loader, device, use_single_gpu):
    model.eval()

    # 用于存储预测正确的样本个数
    sum_num = torch.zeros(1).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    mean_loss = torch.zeros(1).to(device)

    # 在进程0中打印验证进度
    if is_main_process():
        target_test_loader = tqdm(target_test_loader, file=sys.stdout)

    for step, data in enumerate(target_test_loader):
        images, labels = data
        s_output = model.predict(images.to(device)) if use_single_gpu else model.module.predict(images.to(device))
        loss = criterion(s_output, labels.to(device))
        pred = torch.max(s_output, dim=1)[1]
        sum_num += torch.eq(pred, labels.to(device)).sum()

        loss = reduce_value(loss, average=True)
        mean_loss = (mean_loss * step + loss.detach()) / (step + 1)  # update mean losses

        # 在进程0中打印平均loss
        if is_main_process():
            target_test_loader.desc = "[test] mean loss {}".format(round(mean_loss.item(), 3))

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

    # 等待所有进程计算完毕
    if device != torch.device("cpu"):
        torch.cuda.synchronize(device)

    sum_num = reduce_value(sum_num, average=False)

    return sum_num.item()