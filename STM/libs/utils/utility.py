import argparse
from libs.config import getCfg, sanity_check
import os.path as osp
import torch
import os
import cv2
from PIL import Image
import numpy as np
from libs.dataset import DATA_ROOT
from .logger import getLogger
import torch.nn.functional as F

def parse_args():
    parser = argparse.ArgumentParser("STM training") # project name
    parser.add_argument('--cfg', default='', type=str, help='path to config file')
    parser.add_argument('--local_rank', default=-1, type=int,
                        help='process local rank, only used for distributed training')

    args = parser.parse_args() # 实例化
    opt = getCfg()

    # 将config.yaml内容覆盖default配置
    if osp.exists(args.cfg):
        opt.merge_from_file(args.cfg)

    sanity_check(opt)

    return opt, args.local_rank

def mask_iou(pred, target, averaged=True):

    """
    :param pred: [objs, H, W]
    :param target: [objs, H, W]
    :param averaged: mIOU
    :return: IOU of this frame all objects, [1, objs]
    """

    assert len(pred.shape) == 3 and pred.shape == target.shape, "shape error!"

    N = pred.size(0)

    inter = torch.min(pred, target).sum(2).sum(1)
    union = torch.max(pred, target).sum(2).sum(1)

    iou = inter / union

    if averaged:
        iou = torch.mean(iou)

    return iou

def write_mask(mask, info, opt, directory='results'):
    """
    mask: numpy.array of size [T x max_obj x H x W]
    """

    name = info['name']

    if not os.path.exists(directory):
        os.mkdir(directory)

    directory = os.path.join(directory, opt.valset)

    if not os.path.exists(directory):
        os.mkdir(directory)

    video = os.path.join(directory, name)
    if not os.path.exists(video):
        os.mkdir(video)

    h, w = info['size']
    th, tw = mask.shape[2:]
    factor = min(th / h, tw / w)
    sh, sw = int(factor * h), int(factor * w)

    pad_l = (tw - sw) // 2
    pad_t = (th - sh) // 2

    for t in range(mask.shape[0]):
        m = mask[t, :, pad_t:pad_t + sh, pad_l:pad_l + sw]
        m = m.transpose((1, 2, 0))
        rescale_mask = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)

        if 'frame' not in info:
            min_t = 0
            step = 1
            output_name = '{:0>5d}.png'.format(t * step + min_t)
        else:
            output_name = '{}.png'.format(info['frame']['imgs'][t])

        if opt.save_indexed_format == 'index':

            rescale_mask = rescale_mask.argmax(axis=2).astype(np.uint8)
            im = Image.fromarray(rescale_mask).convert('P')
            im.putpalette(info['palette'])
            im.save(os.path.join(video, output_name), format='PNG')

        elif opt.save_indexed_format == 'segmentation':

            rescale_mask = rescale_mask.argmax(axis=2).astype(np.uint8)
            seg = np.zeros((h, w, 3), dtype=np.uint8)
            for k in range(1, rescale_mask.max() + 1):
                seg[rescale_mask == k, :] = info['palette'][(k * 3):(k + 1) * 3][::-1]

            inp_img = cv2.imread(
                os.path.join(DATA_ROOT, opt.valset, 'JPEGImages', '480p', name, output_name.replace('png', 'jpg')))
            im = cv2.addWeighted(inp_img, 0.5, seg, 0.5, 0.0)
            cv2.imwrite(os.path.join(video, output_name), im)

        elif opt.save_indexed_format == 'heatmap':

            rescale_mask[rescale_mask < 0] = 0.0
            rescale_mask = np.max(rescale_mask[:, :, 1:], axis=2)
            rescale_mask = (rescale_mask - rescale_mask.min()) / (rescale_mask.max() - rescale_mask.min()) * 255
            seg = rescale_mask.astype(np.uint8)
            # seg = cv2.GaussianBlur(seg, ksize=(5, 5), sigmaX=2.5)

            seg = cv2.applyColorMap(seg, cv2.COLORMAP_JET)
            inp_img = cv2.imread(
                os.path.join(DATA_ROOT, opt.valset, 'JPEGImages', '480p', name, output_name.replace('png', 'jpg')))
            im = cv2.addWeighted(inp_img, 0.5, seg, 0.5, 0.0)
            cv2.imwrite(os.path.join(video, output_name), im)

        elif opt.save_indexed_format == 'mask':

            fg = np.argmax(rescale_mask, axis=2).astype(np.uint8)

            seg = np.zeros((h, w, 3), dtype=np.uint8)
            seg[fg == 1, :] = info['palette'][3:6][::-1]

            inp_img = cv2.imread(
                os.path.join(DATA_ROOT, opt.valset, 'JPEGImages', '480p', name, output_name.replace('png', 'jpg')))
            im = cv2.addWeighted(inp_img, 0.5, seg, 0.5, 0.0)
            cv2.imwrite(os.path.join(video, output_name), im)

        else:
            raise TypeError('unknown save format {}'.format(opt.save_indexed_format))


def save_checkpoint(state, checkpoint='checkpoint', filename='checkpoint'):
    logger = getLogger(__name__)

    filepath = os.path.join(checkpoint, filename + '.pth.tar')
    torch.save(state, filepath)
    logger.info('save model at {}'.format(filepath))

def Padding(in_list, d, in_size):
    out_list = []
    h, w = in_size
    if h % d > 0:
        new_h = h + d - h % d
    else:
        new_h = h
    if w % d > 0:
        new_w = w + d - w % d
    else:
        new_w = w
    lh, uh = int((new_h - h) / 2), int(new_h - h) - int((new_h - h) / 2)
    lw, uw = int((new_w - w) / 2), int(new_w - w) - int((new_w - w) / 2)
    pad_array = (int(lw), int(uw), int(lh), int(uh))
    for inp in in_list:
        out_list.append(F.pad(inp, pad_array))
    return out_list, pad_array

def Padding_Resume(pre, pad):

    if pad[2] + pad[3] > 0:
        pre = pre[:, :, pad[2]:-pad[3], :]
    if pad[0] + pad[1] > 0:
        pre = pre[:, :, :, pad[0]:-pad[1]]

    return pre