import torch
from libs.utils import mask_iou

def cross_entropy_loss(pred, mask, num_objects, bootstrap=0.4, ref=None):
    # pred: [N x K x H x W]
    # mask: [N x K x H x W] one-hot encoded
    N, _, H, W = mask.shape

    pred = -1 * torch.log(pred) # softmax

    # pred与GT（mask）交叉熵
    ce = pred[:, :num_objects+1] * mask[:, :num_objects+1]

    # 去除冗余信息（ref为第一帧mask）
    if ref is not None:
        valid = torch.sum(ref.view(ref.shape[0], ref.shape[1], -1), dim=-1) > 0 # ref目标通道数无冗余
        valid = valid.float().unsqueeze(2).unsqueeze(3)
        ce *= valid

    # 对每一帧求交叉熵（将各目标通道每个像素交叉熵相加，再转为一维向量）
    loss = torch.sum(ce, dim=1).view(N, -1)
    mloss, _ = torch.sort(loss, dim=-1, descending=True) # 降序排列

    # 选取交叉熵最大的前40%取平均
    num = int(H * W * bootstrap)
    loss = torch.mean(mloss[:, :num])

    return loss

def mask_iou_loss(pred, mask, num_objects, ref=None):

    N, K, H, W = mask.shape
    loss = torch.zeros(1).to(pred.device)

    start = 0 if K == num_objects else 1

    # 冗余目标通道标识
    if ref is not None:
        valid = torch.sum(ref.view(ref.shape[0], ref.shape[1], -1), dim=-1) > 0 # [N, objs]

    for i in range(N):
        obj_loss = (1.0 - mask_iou(pred[i, start:num_objects + start], mask[i, start:num_objects + start], averaged=False))
        if ref is not None:
            obj_loss = obj_loss[valid[i, start:]] # 以第一帧mask目标数为准，防止后续帧目标数增多

        loss += torch.mean(obj_loss)

    # mIou
    loss = loss / N

    return loss

