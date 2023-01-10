import torch
from config import Config
import os
from torch.utils.tensorboard import SummaryWriter
from util.utils import set_seed, CLASS_LABELS, get_bbox
import torch.backends.cudnn as cudnn
from datasets.customized import voc_fewshot, coco_fewshot
from datasets.transforms import Resize, ToTensorNormalize, DilateScribble
from torchvision.transforms import Compose
from models.fewshot import FewShotSeg
import torch.nn as nn
from util.metric import Metric
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

if __name__ == '__main__':
    conf = Config()
    os.environ['CUDA_VISIBLE_DEVICES'] = conf.gpu_id

    set_seed(conf.seed)
    cudnn.enabled = True
    cudnn.benchmark = True
    torch.set_num_threads(1)

    print('###### Prepare data ######')
    data_name = conf.dataset
    if data_name == 'VOC':
        make_data = voc_fewshot
        max_label = 20
    elif data_name == 'COCO':
        make_data = coco_fewshot
        max_label = 80
    else:
        raise ValueError('Wrong config for dataset!')

    labels = CLASS_LABELS[data_name]['all'] - CLASS_LABELS[data_name][conf.label_sets]
    transforms = [Resize(size=conf.input_size)]
    if conf.scribble_dilation > 0:
        transforms.append(DilateScribble(size=conf.scribble_dilation))
    transforms = Compose(transforms)

    print('###### Create model ######')
    model = FewShotSeg(pretrained_path=conf.path['init_path'], cfg=conf.model).cuda()
    if not conf.notrain:
        model.load_state_dict(torch.load("./weights/model-30000.pth", map_location='cpu'))
    model.eval()

    print('###### Testing begins ######')
    metric = Metric(max_label=max_label, n_runs=conf.n_runs)
    with torch.no_grad():
        for run in range(conf.n_runs):
            print(f'### Run {run + 1} ###')
            set_seed(conf.seed + run)

            print(f'### Load data ###')
            dataset = make_data(
                base_dir=conf.path[data_name]['data_dir'],
                split=conf.path[data_name]['data_split'],
                transforms=transforms,
                to_tensor=ToTensorNormalize(),
                labels=labels,
                max_iters=conf.n_steps * conf.batch_size,
                n_ways=conf.task['n_ways'],
                n_shots=conf.task['n_shots'],
                n_queries=conf.task['n_queries']
            )
            if conf.dataset == 'COCO':
                coco_cls_ids = dataset.datasets[0].dataset.coco.getCatIds()
            testloader = DataLoader(dataset, batch_size=conf.batch_size, shuffle=False,
                                    num_workers=1, pin_memory=True, drop_last=False)
            print(f"Total # of Data: {len(dataset)}")

            for sample_batched in tqdm(testloader):
                if conf.dataset == 'COCO':
                    label_ids = [coco_cls_ids.index(x) + 1 for x in sample_batched['class_ids']]
                else:
                    label_ids = list(sample_batched['class_ids'])
                support_images = [[shot.cuda() for shot in way]
                                  for way in sample_batched['support_images']]
                suffix = 'scribble' if conf.scribble else 'mask'

                if conf.bbox:
                    support_fg_mask = []
                    support_bg_mask = []
                    for i, way in enumerate(sample_batched['support_mask']):
                        fg_masks = []
                        bg_masks = []
                        for j, shot in enumerate(way):
                            fg_mask, bg_mask = get_bbox(shot['fg_mask'],
                                                        sample_batched['support_inst'][i][j])
                            fg_masks.append(fg_mask.float().cuda())
                            bg_masks.append(bg_mask.float().cuda())
                        support_fg_mask.append(fg_masks)
                        support_bg_mask.append(bg_masks)
                else:
                    support_fg_mask = [[shot[f'fg_{suffix}'].float().cuda() for shot in way]
                                       for way in sample_batched['support_mask']]
                    support_bg_mask = [[shot[f'bg_{suffix}'].float().cuda() for shot in way]
                                       for way in sample_batched['support_mask']]

                query_images = [query_image.cuda()
                                for query_image in sample_batched['query_images']]
                query_labels = torch.cat(
                    [query_label.cuda()for query_label in sample_batched['query_labels']], dim=0)

                query_pred, _ = model(support_images, support_fg_mask, support_bg_mask,
                                      query_images)

                metric.record(np.array(query_pred.argmax(dim=1)[0].cpu()),
                              np.array(query_labels[0].cpu()),
                              labels=label_ids, n_run=run)

            classIoU, meanIoU = metric.get_mIoU(labels=sorted(labels), n_run=run)
            classIoU_binary, meanIoU_binary = metric.get_mIoU_binary(n_run=run)

            print(f'classIoU: {classIoU}')
            print(f'meanIoU: {meanIoU}')
            print(f'classIoU_binary: {classIoU_binary}')
            print(f'meanIoU_binary: {meanIoU_binary}')

    classIoU, classIoU_std, meanIoU, meanIoU_std = metric.get_mIoU(labels=sorted(labels))
    classIoU_binary, classIoU_std_binary, meanIoU_binary, meanIoU_std_binary = metric.get_mIoU_binary()

    print('----- Final Result -----')
    print(f'classIoU mean: {classIoU}')
    print(f'classIoU std: {classIoU_std}')
    print(f'meanIoU mean: {meanIoU}')
    print(f'meanIoU std: {meanIoU_std}')
    print(f'classIoU_binary mean: {classIoU_binary}')
    print(f'classIoU_binary std: {classIoU_std_binary}')
    print(f'meanIoU_binary mean: {meanIoU_binary}')
    print(f'meanIoU_binary std: {meanIoU_std_binary}')
