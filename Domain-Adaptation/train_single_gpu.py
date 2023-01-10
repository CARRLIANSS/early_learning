import torch
from config import Config
from torch.utils.tensorboard import SummaryWriter
import os
from torchvision import datasets
from models.model import DeepDANet
from multi_train_utils.train_eval_utils import train_one_epoch, evaluate

if __name__ == '__main__':

    conf = Config()
    use_single_gpu = True

    device = torch.device("cuda:5" if torch.cuda.is_available() else "cpu") # change to your gpu_id
    print("\n")
    print(conf.__dict__)
    print("\n")

    print('Start Tensorboard with "tensorboard --logdir=runs", view at http://localhost:6006/')
    tb_writer = SummaryWriter()
    if os.path.exists("./weights") is False:
        os.makedirs("./weights")

    # 实例化验证数据集
    source_dataset = datasets.ImageFolder(root=conf.folder_src, transform=conf.transform['train'])
    target_train_dataset = datasets.ImageFolder(root=conf.folder_tgt, transform=conf.transform['train'])
    target_test_dataset = datasets.ImageFolder(root=conf.folder_tgt, transform=conf.transform['test'])

    batch_size = conf.batch_size
    print('Using {} dataloader workers every process'.format(conf.num_workers))

    source_loader = torch.utils.data.DataLoader(source_dataset,
                                                batch_size=batch_size,
                                                shuffle=True,
                                                pin_memory=True,
                                                num_workers=conf.num_workers)
    target_train_loader = torch.utils.data.DataLoader(target_train_dataset,
                                                      batch_size=batch_size,
                                                      shuffle=True,
                                                      pin_memory=True,
                                                      num_workers=conf.num_workers)
    target_test_loader = torch.utils.data.DataLoader(target_test_dataset,
                                                     batch_size=batch_size,
                                                     shuffle=False,
                                                     pin_memory=True,
                                                     num_workers=conf.num_workers)

    model = DeepDANet(num_class=conf.n_class, base_net=conf.backbone,
                      loss_type=conf.loss_type, use_bottleneck=conf.use_bottleneck).to(device)

    # optimizer
    initial_lr = 1.0
    params = model.get_parameters(initial_lr=initial_lr)
    optimizer = torch.optim.SGD(params, lr=conf.lr, momentum=conf.momentum, weight_decay=conf.weight_decay, nesterov=False)

    # scheduler
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda x: conf.lr * (1. + conf.lr_gamma * float(x)) ** (-conf.lr_decay))

    best_acc = 0
    for epoch in range(conf.epochs):
        mean_loss = train_one_epoch(model=model,
                                    optimizer=optimizer,
                                    source_loader=source_loader,
                                    target_train_loader=target_train_loader,
                                    device=device,
                                    epoch=epoch,
                                    conf=conf)

        scheduler.step()

        sum_num = evaluate(model=model,
                           target_test_loader=target_test_loader,
                           device=device,
                           use_single_gpu=use_single_gpu)

        acc = sum_num / len(target_test_dataset)

        print("[epoch {}] accuracy: {}".format(epoch, round(acc, 3)))
        tags = ["loss", "accuracy", "learning_rate"]
        tb_writer.add_scalar(tags[0], mean_loss, epoch)
        tb_writer.add_scalar(tags[1], acc, epoch)
        tb_writer.add_scalar(tags[2], optimizer.param_groups[0]["lr"], epoch)

        if best_acc < acc:
            torch.save(model.state_dict(), "./weights/model-{}.pth".format(epoch))
            best_acc = acc