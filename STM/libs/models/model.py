import torch
import torch.nn as nn
from torchvision import models
import math
from libs.utils import parse_args
import torch.nn.functional as F
from libs.utils import Padding, Padding_Resume

CHANNEL_EXPAND = {
    'resnet18': 1,
    'resnet34': 1,
    'resnet50': 4,
    'resnet101': 4
}

def Soft_Aggregation(ps, max_obj):

    num_objects, H, W = ps.shape # 之前已经去除了通道维
    em = torch.zeros(1, max_obj+1, H, W).to(ps.device)
    em[0,0,:,:] = torch.prod(1-ps, dim=0) # 返回tensor元素所有乘积
    em[0,1:num_objects+1,:,:] = ps
    em =torch.clamp(em, 1e-7, 1-1e-7)
    logit = torch.log((em / (1-em)))

    return logit

class ResBlock(nn.Module):
    def __init__(self, indim, outdim, stride):
        super(ResBlock, self).__init__()

        self.conv1 = nn.Conv2d(indim, outdim, 3, 1, 1)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(outdim, outdim, 3, 1, 1)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        sc = x
        x = self.relu1(x)
        x = self.conv1(x)
        x = self.relu2(x)
        x = self.conv2(x)
        return x + sc

class Encoder_M(nn.Module):
    def __init__(self, arch):
        super(Encoder_M, self).__init__()

        resnet = models.__getattribute__(arch)(pretrained=True)
        self.conv1_f = resnet.conv1
        self.conv1_m = nn.Conv2d(1, 64, 7, 2, 3, bias=False)
        self.conv1_bg = nn.Conv2d(1, 64, 7, 2, 3, bias=False)

        self.bn1 = resnet.bn1
        self.relu = resnet.relu # 1/2, 64
        self.maxpool = resnet.maxpool # 1/4, 64

        self.stage1 = resnet.layer1 # 1/4, 256
        self.stage2 = resnet.layer2 # 1/8, 512
        self.stage3 = resnet.layer3 # 1/16, 1024

    def forward(self, in_f, in_m, in_bg):
        f = in_f
        m = torch.unsqueeze(in_m, dim=1).float() # add channel dim
        bg = torch.unsqueeze(in_bg, dim=1).float()

        a = self.conv1_f(f)
        b = self.conv1_m(m)
        c = self.conv1_bg(bg)

        x = self.conv1_f(f) + self.conv1_m(m) + self.conv1_bg(bg)
        x = self.bn1(x)
        x = self.relu(x) # 1/2, 64
        x = self.maxpool(x) # 1/4, 64
        x = self.stage1(x) # 1/4, 256
        x = self.stage2(x) # 1/8, 512
        x = self.stage3(x) # 1/16, 1024
        return x

class Encoder_Q(nn.Module):
    def __init__(self, arch):
        super(Encoder_Q, self).__init__()

        resnet = models.__getattribute__(arch)(pretrained=True)
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.stage1 = resnet.layer1
        self.stage2 = resnet.layer2
        self.stage3 = resnet.layer3

    def forward(self, in_f):
        x = self.conv1(in_f)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        r2 = self.stage1(x)
        r3 = self.stage2(r2)
        r4 = self.stage3(r3)
        return r2, r3, r4

class KeyValue(nn.Module):
    def __init__(self, in_dim, keydim, valdim):
        super(KeyValue, self).__init__()

        self.conv_key = nn.Conv2d(in_dim, keydim, 3, 1, 1)
        self.conv_value = nn.Conv2d(in_dim, valdim, 3, 1, 1)

    def forward(self, x):
        key = self.conv_key(x)
        value = self.conv_value(x)
        return key, value

class STM_Read(nn.Module):
    def __init__(self, keydim, valdim):
        super(STM_Read, self).__init__()
        self.keydim = keydim
        self.valdim = valdim

    def forward(self, key_M, value_M, key_Q, value_Q, num_objects):
        _, _, H, W = key_M.shape

        km = key_M.permute(0, 2, 3, 1).contiguous().view(num_objects, -1, self.keydim)
        vm = value_M.permute(0, 2, 3, 1).contiguous().view(num_objects, -1, self.valdim)

        kq = key_Q.expand(num_objects, -1, -1, -1).contiguous().view(num_objects, self.keydim, -1)
        vq = value_Q.expand(num_objects, -1, -1, -1).contiguous()

        p = torch.bmm(km, kq)
        p = p / math.sqrt(self.keydim)
        p = torch.softmax(p, dim=1) # 在dim=1维sum=1
        p = torch.transpose(p, 1, 2)

        mem = torch.bmm(p, vm)
        mem = mem.permute(0, 2, 1).contiguous().view(num_objects, self.valdim, H, W)

        mem_out = torch.cat([vq, mem], dim=1)

        return mem_out

class Refine(nn.Module):
    def __init__(self, indim_ski, indim):
        super(Refine, self).__init__()

        self.conv = nn.Conv2d(indim_ski, indim, 3, 1, 1)
        self.resblock1 = ResBlock(indim, indim, 1)
        self.resblock2 = ResBlock(indim, indim, 1)
        self.tr_conv = nn.ConvTranspose2d(indim, indim, 2, 2)

    def forward(self, x, ski):
        y = self.conv(ski)
        y = self.resblock1(y)
        x = self.tr_conv(x)

        x = self.resblock2(x + y)
        return x

class Decoder(nn.Module):
    def __init__(self, indim, outdim, expand):
        super(Decoder, self).__init__()

        self.conv1 = nn.Conv2d(indim, outdim, 3, 1, 1)
        self.resblock = ResBlock(outdim, outdim, 1)
        self.RF1 = Refine(128 * expand, outdim)
        self.RF2 = Refine(64 * expand, outdim)
        self.conv2 = nn.Conv2d(outdim, 2, 3, 1, 1)

    def forward(self, x, st2, st1, f, num_objects):
        s2 = st2.expand(num_objects, -1, -1, -1).contiguous()
        s1 = st1.expand(num_objects, -1, -1, -1).contiguous()

        x = self.conv1(x)
        x = self.RF1(x, s2) # out: 1/8, 256
        x = self.RF2(x, s1) # out: 1/4, 256
        x = self.conv2(F.relu(x))

        x = F.interpolate(x, size=f.shape[2:], mode="bilinear", align_corners=False) # out: 1, 2
        return x

class STM(nn.Module):
    def __init__(self, opt):
        super(STM, self).__init__()

        keydim = opt.keydim
        valdim = opt.valdim
        arch = opt.arch

        expand = CHANNEL_EXPAND[arch]

        self.Encoder_M = Encoder_M(arch)
        self.Encoder_Q = Encoder_Q(arch)

        self.KV_M = KeyValue(256 * expand, keydim, valdim)
        self.KV_Q = KeyValue(256 * expand, keydim, valdim)

        self.STM_Read = STM_Read(keydim, valdim)
        self.Decoder = Decoder(2 * valdim, 256, expand)

    def load_param(self, weight):

        s = self.state_dict()

        # f = open("./model.txt", "w", encoding="utf-8")
        # g = open("./pretrain.txt", "w", encoding="utf-8")
        # for k1, v1 in s.items():
        #     f.write(k1 + '\n')
        # for k2, v2 in weight.items():
        #     g.write(k2 + '\n')

        for key, val in weight.items():
            if s[key].shape == val.shape:
                s[key][...] = val # 就地赋值
            # elif key not in s:
            #     print('ignore weight from not found key {}'.format(key))
            else:
                # print('ignore weight of mistached shape in key {}'.format(key))
                print('ignore weight from not found key {}'.format(key))

    def memorize(self, frame, masks, num_objects):

        frame_batch = []
        mask_batch = []
        bg_batch = []

        # 剔除冗余的one hot mask（0维度是冗余的）
        for o in range(1, num_objects+1):
            frame_batch.append(frame)
            mask_batch.append(masks[:,o]) # 间接去掉fram维度

        for o in range(1, num_objects+1):
            bg_batch.append(torch.clamp(1.0 - masks[:, o], min=0.0, max=1.0)) # 背景区域像素值设为1，目标区域为0

        # make batch
        frame_batch = torch.cat(frame_batch, dim=0)
        mask_batch = torch.cat(mask_batch, dim=0)
        bg_batch = torch.cat(bg_batch, dim=0)

        r = self.Encoder_M(frame_batch, mask_batch, bg_batch)
        key, value = self.KV_M(r)

        return key, value

    def segment(self, frame, key_M, value_M, num_objects, max_objs):

        r2, r3, r4 = self.Encoder_Q(frame)
        key, value = self.KV_Q(r4)
        m = self.STM_Read(key_M, value_M, key, value, num_objects)
        logit = self.Decoder(m, r3, r2, frame, num_objects)
        ps = F.softmax(logit, dim=1)[:,1] # ？
        logit = Soft_Aggregation(ps, max_objs)

        return logit # [T, C, H, W]

    def forward(self, frame, masks=None, key_M=None, value_M=None, num_objects=None, max_objs=None):

        if masks is not None:
            return self.memorize(frame, masks, num_objects)
        else:
            return self.segment(frame, key_M, value_M, num_objects, max_objs)


def test():
    frame = torch.randn(size=(1,3,480,854))
    masks = torch.randn(size=(1,13,480,854))
    (frame, masks), pad = Padding([frame, masks], 16, (frame.size()[2], frame.size()[3]))
    num_objects = 2
    max_objs = 12

    opt, _ = parse_args()

    stm = STM(opt)
    key_M, value_M = stm(frame=frame, masks=masks, num_objects=num_objects)
    pre = stm(frame=frame, key_M=key_M, value_M=value_M, num_objects=num_objects, max_objs=max_objs)

    pre = Padding_Resume(pre, pad)

    print(key_M.size(), value_M.size(), pre.size())

if __name__ == "__main__":
    test()