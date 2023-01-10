import torch
import time
import numpy as np
from libs.dataset import oh_convert_to_mask
from libs.utils import AverageMeter
from progress.bar import Bar
from libs.utils import write_mask
from libs.utils import Padding, Padding_Resume

def train_one_epoch(model, train_loader, device, opt, criterion, optimizer, epoch, writer):

    data_time = AverageMeter()
    loss = AverageMeter() # loss.avg 对不断加入的loss取平均（加权平均）

    start_time = time.time()

    bar = Bar('training', max=len(train_loader))

    optimizer.zero_grad()

    for batch_idx, data in enumerate(train_loader):

        frames, masks, objs, _ = data
        max_obj = masks.shape[2] - 1

        # measure data loading time
        data_time.update(time.time() - start_time)

        frames = frames.to(device)
        masks = masks.to(device)

        N, T, C, H, W = frames.size()
        total_loss = 0.0
        for idx in range(N):

            keys = []
            vals = []
            for t in range(1, T):
                frame = frames[idx]
                mask = masks[idx]
                num_objects = objs[idx]

                # memorize
                if t - 1 == 0:
                    tmp_mask = mask[t - 1:t]
                else:
                    tmp_mask = out
                # 用第一帧生成memory信息
                # memory网络首先要对mask去冗余通道
                (frame, tmp_mask), pad = Padding([frame, tmp_mask], 16, (frame.size()[2], frame.size()[3]))
                key, val = model(frame=frame[t - 1:t, :, :, :], masks=tmp_mask, num_objects=num_objects)

                keys.append(key)
                vals.append(val)

                # segment
                tmp_key = torch.cat(keys, dim=1)
                tmp_val = torch.cat(vals, dim=1)
                # 预测第二帧
                logits = model(frame=frame[t:t + 1, :, :, :], key_M=tmp_key, value_M=tmp_val,
                                   num_objects=num_objects, max_objs=max_obj)

                logits = Padding_Resume(logits, pad)
                out = torch.softmax(logits, dim=1)
                gt = mask[t:t + 1]

                total_loss += criterion(out, gt, num_objects, ref=mask[0:1, :num_objects + 1])

            # cycle-consistancy
            frame = frames[idx]
            mask = masks[idx]
            num_objects = objs[idx]
            (frame, out), pad = Padding([frame, out], 16, (frame.size()[2], frame.size()[3]))
            key, val = model(frame=frame[T - 1:T, :, :, :], masks=out, num_objects=num_objects)
            keys.append(key)
            vals.append(val)

            cycle_loss = 0.0
            for t in range(T - 1, 0, -1):
                cm = np.transpose(mask[t].detach().cpu().numpy(), [1, 2, 0])
                if oh_convert_to_mask(cm, max_obj).max() == num_objects:
                    tmp_key = keys[t]
                    tmp_val = vals[t]
                    logits = model(frame=frame[0:1, :, :, :], key_M=tmp_key, value_M=tmp_val,
                                       num_objects=num_objects, max_objs=max_obj)
                    logits = Padding_Resume(logits, pad)
                    first_out = torch.softmax(logits, dim=1)
                    cycle_loss += criterion(first_out, mask[0:1], num_objects, ref=mask[t:t + 1, :num_objects + 1])

            total_loss += cycle_loss

        total_loss = total_loss / (N * (T - 1)) # 对每帧的loss

        # record loss
        if isinstance(total_loss, torch.Tensor) and total_loss.item() > 0.0:
            loss.update(total_loss.item(), 1)
            total_loss.backward() # 梯度计算
            optimizer.step() # 参数更新
            optimizer.zero_grad() # 梯度清零

        # tensorboard
        niter = epoch * len(train_loader) + batch_idx
        writer.add_scalars('Train_loss', {'train_loss': total_loss.data.item()}, niter)

        # measure elapsed time
        end = time.time()
        # plot progress
        bar.suffix = '({batch}/{size}) Data: {data:.3f}s |Loss: {loss:.5f}'.format(
            batch=batch_idx + 1,
            size=len(train_loader),
            data=data_time.val,
            loss=loss.val
        )
        bar.next()
        del frames, masks
    bar.finish()

    return data_time.val, loss.avg

@torch.no_grad()
def test(model, val_loader, device, opt, criterion, epoch, writer):

    model.eval()

    data_time = AverageMeter()
    loss = AverageMeter()  # loss.avg 对不断加入的loss取平均（加权平均）

    start_time = time.time()

    bar = Bar('testing', max=len(val_loader))

    for batch_idx, data in enumerate(val_loader):

        data_time.update(time.time() - start_time)

        frames, masks, objs, infos = data
        frames = frames.to(device)
        masks = masks.to(device)

        info = infos[0]
        max_obj = masks.shape[2] - 1

        N, T, _, H, W = frames.shape
        total_loss = 0.0
        # total_miou = 0.0

        frame = frames[0]
        mask = masks[0]
        num_objects = objs[0]
        pred = [mask[0:1]]
        keys = []
        vals = []
        for t in range(1, T):

            if t - 1 == 0:
                tmp_mask = mask[0:1]
            else:
                tmp_mask = out

            # memorize
            (tmp_frame, tmp_mask), pad = Padding([frame, tmp_mask], 16, (frame.size()[2], frame.size()[3]))
            key, val = model(frame=tmp_frame[t - 1:t], masks=tmp_mask, num_objects=num_objects)

            tmp_key = torch.cat(keys + [key], dim=1)
            tmp_val = torch.cat(vals + [val], dim=1)

            logits = model(frame=tmp_frame[t:t + 1], key_M=tmp_key, value_M=tmp_val,
                               num_objects=num_objects, max_objs=max_obj)
            logits = Padding_Resume(logits, pad)

            out = torch.softmax(logits, dim=1)
            gt = mask[t:t+1]

            total_loss = total_loss + criterion(out, gt, num_objects, ref=mask[0:1, :num_objects + 1])

            pred.append(out)

            # 间隔几帧加入新的key、val
            if (t - 1) % opt.save_freq == 0:
                keys.append(key)
                vals.append(val)

        pred = torch.cat(pred, dim=0)

        pred = pred.detach().cpu().numpy()
        write_mask(pred, info, opt, directory=opt.output_dir)

        total_loss = total_loss / (T - 1)

        # record loss
        if isinstance(total_loss, torch.Tensor) and total_loss.item() > 0.0:
            loss.update(total_loss.item(), 1)

        # tensorboard
        niter = epoch * len(val_loader) + batch_idx
        writer.add_scalars('Val_loss', {'val_loss': total_loss.data.item()}, niter)

        # measure elapsed time
        end = time.time()
        # plot progress
        bar.suffix = '({batch}/{size}) Data: {data:.3f}s |Loss: {loss:.5f}'.format(
            batch=batch_idx + 1,
            size=len(val_loader),
            data=data_time.val,
            loss=loss.val
        )
        bar.next()
    bar.finish()

    return data_time.val, loss.avg