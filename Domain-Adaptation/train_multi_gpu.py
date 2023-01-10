import torch
from config import Config
from multi_train_utils.distributed_utils import init_distributed_mode, dist, cleanup
from torch.utils.tensorboard import SummaryWriter
import os
from torchvision import datasets
from models.model import DeepDANet
import tempfile
from multi_train_utils.train_eval_utils import train_one_epoch, evaluate

# CUDA_VISIBLE_DEVICES=0,3 python -m torch.distributed.launch --nproc_per_node=2 --use_env train_multi_gpu_using_launch.py

if __name__ == "__main__":

    if torch.cuda.is_available() is False:
        raise EnvironmentError("not find GPU device for training.")

    conf = Config()
    use_single_gpu = False

    # 初始化各进程环境
    init_distributed_mode(args=conf)

    rank = conf.rank
    device = torch.device(conf.device)
    batch_size = conf.batch_size
    conf.lr *= conf.world_size  # 学习率要根据并行GPU的数量进行倍增
    checkpoint_path = ""

    if rank == 0:  # 在第一个进程中打印信息，并实例化tensorboard
        print('\n')
        print(conf.__dict__)
        print('\n')
        print('Start Tensorboard with "tensorboard --logdir=runs", view at http://localhost:6006/')
        tb_writer = SummaryWriter()
        if os.path.exists("./weights") is False:
            os.makedirs("./weights")

    # 实例化验证数据集
    source_dataset = datasets.ImageFolder(root=conf.folder_src, transform=conf.transform['train'])
    target_train_dataset = datasets.ImageFolder(root=conf.folder_tgt, transform=conf.transform['train'])
    target_test_dataset = datasets.ImageFolder(root=conf.folder_tgt, transform=conf.transform['test'])

    # 给每个rank对应的进程分配训练的样本索引
    source_sampler = torch.utils.data.distributed.DistributedSampler(source_dataset)
    target_train_sampler = torch.utils.data.distributed.DistributedSampler(target_train_dataset)
    target_test_sampler = torch.utils.data.distributed.DistributedSampler(target_test_dataset)

    # 将样本索引每batch_size个元素组成一个list
    source_batch_sampler = torch.utils.data.BatchSampler(source_sampler, batch_size, drop_last=True)
    target_train_batch_sampler = torch.utils.data.BatchSampler(target_train_sampler, batch_size, drop_last=True)

    if rank == 0:
        print('Using {} dataloader workers every process'.format(conf.num_workers))

    source_loader = torch.utils.data.DataLoader(source_dataset,
                                                batch_sampler=source_batch_sampler,
                                                pin_memory=True,
                                                num_workers=conf.num_workers)
    target_train_loader = torch.utils.data.DataLoader(target_train_dataset,
                                                      batch_sampler=target_train_batch_sampler,
                                                      pin_memory=True,
                                                      num_workers=conf.num_workers)
    target_test_loader = torch.utils.data.DataLoader(target_test_dataset,
                                                     batch_size=batch_size,
                                                     sampler=target_test_sampler,
                                                     pin_memory=True,
                                                     num_workers=conf.num_workers)

    model = DeepDANet(num_class=conf.n_class, base_net=conf.backbone,
                      loss_type=conf.loss_type, use_bottleneck=conf.use_bottleneck).to(device)

    # 不存在预训练权重
    checkpoint_path = os.path.join(tempfile.gettempdir(), "initial_weights.pt")
    # 如果不存在预训练权重，需要将第一个进程中的权重保存，然后其他进程载入，保持初始化权重一致
    if rank == 0:
        torch.save(model.state_dict(), checkpoint_path)

    dist.barrier()
    # 这里注意，一定要指定map_location参数，否则会导致第一块GPU占用更多资源
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    # 转为DDP模型
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[conf.gpu])

    # optimizer
    initial_lr = 1.0
    params = model.module.get_parameters(initial_lr=initial_lr)
    optimizer = torch.optim.SGD(params, lr=conf.lr, momentum=conf.momentum, weight_decay=conf.weight_decay, nesterov=False)

    # scheduler
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda x: conf.lr * (1. + conf.lr_gamma * float(x)) ** (-conf.lr_decay))

    best_acc = 0
    for epoch in range(conf.epochs):
        source_sampler.set_epoch(epoch)
        target_train_sampler.set_epoch(epoch)

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

        acc = sum_num / target_test_sampler.total_size

        if rank == 0:
            print("[epoch {}] accuracy: {}".format(epoch, round(acc, 3)))
            tags = ["loss", "accuracy", "learning_rate"]
            tb_writer.add_scalar(tags[0], mean_loss, epoch)
            tb_writer.add_scalar(tags[1], acc, epoch)
            tb_writer.add_scalar(tags[2], optimizer.param_groups[0]["lr"], epoch)

            if best_acc < acc:
                torch.save(model.module.state_dict(), "./weights/model-{}.pth".format(epoch))
                best_acc = acc

    # 删除临时缓存文件
    if rank == 0:
        if os.path.exists(checkpoint_path) is True:
            os.remove(checkpoint_path)

    cleanup()