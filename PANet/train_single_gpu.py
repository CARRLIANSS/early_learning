import torch
from config import Config
import os
from torch.utils.tensorboard import SummaryWriter
from util.utils import set_seed, CLASS_LABELS
import torch.backends.cudnn as cudnn
from models.fewshot import FewShotSeg
import torch.nn as nn
from datasets.customized import voc_fewshot, coco_fewshot
from torchvision.transforms import Compose
from datasets.transforms import RandomMirror, Resize, ToTensorNormalize
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR

if __name__ == '__main__':

    conf = Config()
    os.environ['CUDA_VISIBLE_DEVICES'] = conf.gpu_id

    print("\n")
    print(conf.__dict__)
    print("\n")

    print('Start Tensorboard with "tensorboard --logdir=runs", view at http://localhost:6006/')
    tb_writer = SummaryWriter()
    if os.path.exists("./weights") is False:
        os.makedirs("./weights")

    set_seed(conf.seed)
    cudnn.enabled = True
    cudnn.benchmark = True
    torch.set_num_threads(1)

    print('###### Load data ######')
    data_name = conf.dataset
    if data_name == 'VOC':
        make_data = voc_fewshot
    elif data_name == 'COCO':
        make_data = coco_fewshot
    else:
        raise ValueError('Wrong config for dataset!')
    labels = CLASS_LABELS[data_name][conf.label_sets]
    transforms = Compose([Resize(size=conf.input_size), RandomMirror()])

    dataset = make_data(
        base_dir=conf.path[data_name]['data_dir'],
        split=conf.path[data_name]['data_split'],
        transforms=transforms,
        to_tensor=ToTensorNormalize(),
        labels=labels,
        max_iters=conf.n_steps * conf.batch_size, # batch_size
        n_ways=conf.task['n_ways'],
        n_shots=conf.task['n_shots'],
        n_queries=conf.task['n_queries']
    )

    trainloader = DataLoader(
        dataset,
        batch_size=conf.batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        drop_last=True
    )

    print('###### Create model ######')
    model = FewShotSeg(pretrained_path=conf.path['init_path'], cfg=conf.model).cuda()
    model.train()

    print('###### Set optimizer ######')
    optimizer = torch.optim.SGD(model.parameters(), **conf.optim)
    scheduler = MultiStepLR(optimizer, milestones=conf.lr_milestones, gamma=0.1)
    criterion = nn.CrossEntropyLoss(ignore_index=conf.ignore_label)

    log_loss = {'loss': 0, 'align_loss': 0}
    print('###### Training ######')
    for i_iter, sample_batched in enumerate(trainloader):
        # Prepare input
        support_images = [[shot.cuda() for shot in way]
                          for way in sample_batched['support_images']]
        support_fg_mask = [[shot[f'fg_mask'].float().cuda() for shot in way]
                           for way in sample_batched['support_mask']]
        support_bg_mask = [[shot[f'bg_mask'].float().cuda() for shot in way]
                           for way in sample_batched['support_mask']]

        query_images = [query_image.cuda()
                        for query_image in sample_batched['query_images']]
        query_labels = torch.cat(
            [query_label.long().cuda() for query_label in sample_batched['query_labels']], dim=0)

        # Forward and Backward
        optimizer.zero_grad()
        query_pred, align_loss = model(support_images, support_fg_mask, support_bg_mask, query_images)
        query_loss = criterion(query_pred, query_labels)
        loss = query_loss + align_loss * conf.align_loss_scaler
        loss.backward()
        optimizer.step()
        scheduler.step()

        # Log loss
        query_loss = query_loss.detach().data.cpu().numpy()
        align_loss = align_loss.detach().data.cpu().numpy() if align_loss != 0 else 0

        log_loss['loss'] += query_loss
        log_loss['align_loss'] += align_loss

        # print loss and take snapshots
        if (i_iter + 1) % conf.print_interval == 0:
            loss = log_loss['loss'] / (i_iter + 1)
            align_loss = log_loss['align_loss'] / (i_iter + 1)
            print(f'step {i_iter + 1}: loss: {loss}, align_loss: {align_loss}')

            tags = ["loss", "align_loss"]
            tb_writer.add_scalar(tags[0], loss, i_iter)
            tb_writer.add_scalar(tags[1], align_loss, i_iter)

        if (i_iter + 1) % conf.save_pred_every == 0:
            print('###### Taking snapshot model save ######')
            torch.save(model.state_dict(), "./weights/model-{}.pth".format(i_iter + 1))

    print('###### Saving final model ######')
    torch.save(model.state_dict(), "./weights/model-{}.pth".format(i_iter + 1))