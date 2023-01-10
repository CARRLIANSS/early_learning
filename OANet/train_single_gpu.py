from config import Config
import os
from torch.utils.tensorboard import SummaryWriter
import torch
from dataset.dataset import CorrespondencesDataset, collate_fn
from torch.utils.data import DataLoader
from models.oanet import OANet
import torch.optim as optim
from utils.train_eval_utils import train_one_epoch, evaluate
from utils.logger import Logger
import math
import torch.optim.lr_scheduler as lr_scheduler

os.environ['CUDA_VISIBLE_DEVICES'] = '5'

if __name__ == '__main__':
    if torch.cuda.is_available() is False:
        raise EnvironmentError("not find GPU device for training.")

    conf = Config()

    # 在第一个进程中打印信息，并实例化tensorboard
    print('Start Tensorboard with "tensorboard --logdir=runs", view at http://localhost:6006/')
    tb_writer = SummaryWriter()

    # Create directory
    result_path = conf.logs + '/' + conf.dataset
    if not os.path.isdir(result_path + '/train'):
        os.makedirs(result_path + '/train')
        f = open(result_path + '/train' + '/log_train.txt', 'w')
        f.close()
        f = open(result_path + '/train' + '/log_valid.txt', 'w')
        f.close()
    if not os.path.isdir(result_path + '/valid'):
        os.makedirs(result_path + '/valid')
    if os.path.exists(result_path + '/config.th'):
        print('warning: will overwrite config file')
    torch.save(conf.__dict__, result_path + '/config.th')
    # path for saving traning logs
    conf.log_path = result_path + '/train'
    if not os.path.isdir(conf.checkpoint_path):
        os.makedirs(conf.checkpoint_path)
    if not os.path.isdir(conf.best_model_path):
        os.makedirs(conf.best_model_path)

    # Load data
    train_dataset = CorrespondencesDataset(conf.data_tr, conf)
    valid_dataset = CorrespondencesDataset(conf.data_va, conf)

    print('Using {} dataloader workers every process'.format(conf.num_workers))

    train_loader = DataLoader(train_dataset,
                              batch_size=conf.train_batch_size,
                              shuffle=True,
                              pin_memory=True,
                              num_workers=conf.num_workers,
                              collate_fn=collate_fn)
    valid_loader = DataLoader(valid_dataset,
                              batch_size=conf.train_batch_size,
                              shuffle=True,
                              pin_memory=True,
                              num_workers=conf.num_workers,
                              collate_fn=collate_fn)

    # Create model
    model = OANet(conf).cuda()

    best_acc = -1
    logger_train = Logger(os.path.join(conf.log_path, 'log_train.txt'), title='oan')
    logger_train.set_names(['Learning Rate'] + ['Geo Loss', 'Classfi Loss', 'L2 Loss'] * (conf.iter_num + 1))
    logger_valid = Logger(os.path.join(conf.log_path, 'log_valid.txt'), title='oan')
    logger_valid.set_names(['Valid Acc'] + ['Geo Loss', 'Clasfi Loss', 'L2 Loss'])
    # 如果存在预训练权重则载入
    if os.path.exists(conf.pretrained):
        weights_dict = torch.load(conf.pretrained, map_location="cuda")
        best_acc = weights_dict['best_acc']
        load_weights_dict = {k: v for k, v in weights_dict['state_dict'].items()
                             if model.state_dict()[k].numel() == v.numel()}
        model.load_state_dict(load_weights_dict, strict=False)

    # Set optimizer
    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(pg, lr=conf.train_lr, weight_decay=conf.weight_decay)
    # 学习率训练策略（可去），Scheduler https://arxiv.org/pdf/1812.01187.pdf
    lf = lambda x: ((1 + math.cos(x * math.pi / conf.epochs)) / 2) * (1 - conf.lrf) + conf.lrf  # cosine
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)


    for epoch in range(conf.epochs):

        mean_loss = train_one_epoch(model=model,
                                    optimizer=optimizer,
                                    data_loader=train_loader,
                                    device="cuda",
                                    epoch=epoch,
                                    conf=conf,
                                    logger_train=logger_train)

        scheduler.step()

        cur_global_step = epoch * len(train_loader)
        va_res, geo_loss, cla_loss, l2_loss, _, _, _ = evaluate(model, valid_loader, conf, cur_global_step, 'valid')
        logger_valid.append([va_res, geo_loss, cla_loss, l2_loss])
        val_loss = geo_loss + cla_loss

        print("[epoch {}] train_loss: {} val_loss: {} qt_acc: {}".format(epoch, round(mean_loss, 3), round(val_loss, 3), round(va_res, 3)))
        tags = ["train_loss", "val_loss", "qt_acc", "learning_rate"]
        tb_writer.add_scalar(tags[0], mean_loss, epoch)
        tb_writer.add_scalar(tags[1], val_loss, epoch)
        tb_writer.add_scalar(tags[2], va_res, epoch)
        tb_writer.add_scalar(tags[3], optimizer.param_groups[0]["lr"], epoch)

        if va_res > best_acc:
            print("Saving best model with va_res = {}".format(va_res))
            best_acc = va_res
            torch.save({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_acc': best_acc,
                'optimizer': optimizer.state_dict(),
            }, os.path.join(conf.best_model_path, 'model_best.pth'))

        torch.save({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_acc': best_acc,
            'optimizer': optimizer.state_dict(),
        }, os.path.join(conf.checkpoint_path, 'checkpoint.pth'))