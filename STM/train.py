import torch
from libs.utils import parse_args, setup, getLogger
import os
from libs.dataset.transform import TrainTransform, TestTransform
from libs.dataset.data import build_dataset
import copy
from torch.utils import data
from libs.models import STM
from libs.models import STAN
from collections import OrderedDict
from libs.utils import train_one_epoch, test
from libs.utils import cross_entropy_loss, mask_iou_loss
from libs.utils import save_checkpoint
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
from libs.davis17_evaluate.evalutation_method import evalutate

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

# 解析配置文件，返回opt实例
opt, _ = parse_args()

device = "cpu"
if torch.cuda.is_available():
    device = 'cuda:{}'.format(opt.gpu_id)

def main():

    if not os.path.isdir(opt.checkpoint):
        os.makedirs(opt.checkpoint)

    if not os.path.isdir(opt.log_dir):
        os.makedirs(opt.log_dir)

    logfile = os.path.join(opt.log_dir, opt.mode + '_log.txt')
    setup(filename=logfile, resume=opt.resume != '')
    log = getLogger(__name__)

    # Data
    log.info('Preparing dataset')

    input_size = tuple(opt.input_size)

    train_transform = TrainTransform(size=input_size)
    test_transform = TestTransform(size=input_size)

    datalist = []
    for dataset, freq, max_skip in zip(opt.trainset, opt.datafreq, opt.max_skip):

        ds = build_dataset(
            name=dataset,
            train=True,
            frames_sampled=opt.frames_sampled,
            transform=train_transform,
            max_skip=max_skip,
            video_sampled=opt.video_sampled
        )

        datalist += [copy.deepcopy(ds) for _ in range(freq)]

    trainset = data.ConcatDataset(datalist)

    valset = build_dataset(
        name=opt.valset,
        train=False,
        transform=test_transform,
        video_sampled=opt.video_sampled
    )

    if opt.data_backend == 'PIL':
        train_loader = data.DataLoader(trainset, batch_size=opt.train_batch, shuffle=True, num_workers=opt.workers,
                                      collate_fn=valset.collate_fn)
    else:
        raise TypeError('unkown data backend {}'.format(opt.data_backend))

    val_loader = data.DataLoader(valset, batch_size=1, shuffle=False, num_workers=opt.workers,
                                 collate_fn=valset.collate_fn)

    # Model
    log.info("creating model")

    # model = STM(opt)
    model = STAN(opt)
    log.info('Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))

    # 冻结某层
    # # 冻结fc1层的参数
    # for name, param in model.named_parameters():
    #     if "fc1" in name:
    #         param.requires_grad = False

    # # 查看参数冻结情况
    # for k, v in model.named_parameters():
    #     print("{}:参数需要计算梯度并更新吗?{}".format(k, v.requires_grad))  # False表示冻结了，True表示没有冻结

    # # 定义一个fliter，只传入requires_grad=True的模型参数
    # optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-2)

    # 冻结所有BN层
    model.eval()
    model = model.to(device)
    # set training parameters
    for p in model.parameters():
        p.requires_grad = True

    # 预训练模型加载
    if opt.resume:
        # Load checkpoint.
        log.info('Resuming from checkpoint {}'.format(opt.resume))
        assert os.path.isfile(opt.resume), 'Error: no checkpoint directory found!'

        checkpoint = torch.load(opt.resume, map_location=device)
        start_epoch = checkpoint['epoch']
        skips_list = checkpoint['max_skip']
        opt.learning_rate = checkpoint['learning_rate']
        model.load_param(checkpoint['state_dict'])

        try:
            if isinstance(skips_list, list):
                for idx, skip in enumerate(skips_list):
                    trainset.datasets[idx].set_max_skip(skip)
            else:
                trainset.set_max_skip(skips_list[0])
        except:
            log.warn('Initializing max skip fail')

    else:
        if opt.pretrain:
            log.info('Initialize pretrained model with weight file {}'.format(opt.pretrain))
            weight = torch.load(opt.pretrain, map_location='cpu')
            if isinstance(weight, OrderedDict):
                model.load_param(weight)
            else:
                model.load_param(weight['state_dict'])

        start_epoch = 0

    criterion = None
    if opt.loss == 'ce':
        criterion = cross_entropy_loss
    elif opt.loss == 'iou':
        criterion = mask_iou_loss
    elif opt.loss == 'both':
        criterion = lambda pred, target, nobjs, ref: cross_entropy_loss(pred, target, nobjs, ref=ref) + mask_iou_loss(
            pred, target, nobjs, ref=ref)
    else:
        raise TypeError('Unknown training loss %s' % opt.loss)

    writer = SummaryWriter(log_dir=opt.log_dir)

    best_JF_M = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())

    optimizer = None
    if opt.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=opt.learning_rate,
                              momentum=opt.momentum[0], weight_decay=opt.weight_decay)
    elif opt.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=opt.learning_rate,
                               betas=opt.momentum, weight_decay=opt.weight_decay)
    else:
        raise TypeError('Unkown optimizer type %s' % opt.optimizer)

    # 学习率调整策略
    CosineLR = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.epoches, eta_min=0)

    for epoch in range(start_epoch, opt.epoches):

        print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, opt.epoches, opt.learning_rate))

        log.info('Skip Info:')
        skip_info = dict()
        if isinstance(trainset, data.ConcatDataset):
            for dataset in trainset.datasets:
                skip_info.update( {type(dataset).__name__: dataset.max_skip} ) # update相同的不加入
        else:
            skip_info.update( {type(trainset).__name__: dataset.max_skip} )
        skip_print = ''
        for k, v in skip_info.items():
            skip_print += '{}: {} '.format(k, v)
        log.info(skip_print) # Davis17: 50 YoutubeVOS: 10

        train_time, train_loss = train_one_epoch(model, train_loader, device, opt, criterion, optimizer, epoch, writer)

        # 训练loss、time日志输出
        log_format = 'Epoch: {} LR: {} Train-Loss: {} Train-Time: {}min'
        log.info(log_format.format(epoch+1, opt.learning_rate, train_loss, train_time // 60))

        if (epoch + 1) % opt.epoch_per_test == 0:
            val_time, val_loss = test(model, val_loader, device, opt, criterion, epoch, writer)
            log.info('results are saved at {}'.format(os.path.join(opt.output_dir, opt.valset)))

            # 验证loss、time日志输出
            log_format = 'Epoch: {} LR: {} Val-Loss: {} Val-Time: {}min'
            log.info(log_format.format(epoch + 1, opt.learning_rate, val_loss, val_time // 60))

            # 验证J&F日志输出
            J_F_M = evalutate(opt, epoch)

            # save best weight
            if J_F_M > best_JF_M:
                best_JF_M = J_F_M
                best_model_wts = copy.deepcopy(model.state_dict())
        else:
            val_time = 0.0

        CosineLR.step()  # 学习率更新
        for param_group in optimizer.param_groups:
            opt.learning_rate = param_group['lr']

        # adjust max_skip
        if (epoch + 1) % opt.increment_per_epoches == 0:
            if isinstance(trainset, data.ConcatDataset):
                for dataset in trainset.datasets:
                    dataset.increase_max_skip()
            else:
                trainset.increase_max_skip()

        # save resume model
        skips = [ds.max_skip for ds in trainset.datasets]
        log.info('Saving resume model...')
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'max_skip': skips,
            'learning_rate': opt.learning_rate,
        }, checkpoint=opt.checkpoint, filename=opt.mode)
        log.info('Successful')

    writer.close()

    # save best model
    log.info('Saving best model...')
    model_name = "best_model.pth"
    torch.save(best_model_wts, model_name)
    log.info('Successful')


if __name__ == '__main__':
    main()







