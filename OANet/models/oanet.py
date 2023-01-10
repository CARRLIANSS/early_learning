import torch
import torch.nn as nn
from config import Config
from models.loss import batch_episym
import time

class PointCN(nn.Module):

    def __init__(self, channels, out_channels=None):
        super(PointCN, self).__init__()
        if not out_channels:
           out_channels = channels
        self.shot_cut = None
        if out_channels != channels:
            self.shot_cut = nn.Conv2d(channels, out_channels, kernel_size=1)

        self.conv = nn.Sequential(
            nn.InstanceNorm2d(channels, eps=1e-3), # coordinate norm
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, out_channels, kernel_size=1),
            nn.InstanceNorm2d(out_channels, eps=1e-3),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=1)
        )

    def forward(self, x):
        out = self.conv(x)
        if self.shot_cut:
            out = out + self.shot_cut(x)
        else:
            out = out + x
        return out

class diff_pool(nn.Module):

    def __init__(self, in_channel, output_points):
        super(diff_pool, self).__init__()
        self.output_points = output_points # clusters
        self.conv = nn.Sequential(
                    nn.InstanceNorm2d(in_channel, eps=1e-3),
                    nn.BatchNorm2d(in_channel),
                    nn.ReLU(),
                    nn.Conv2d(in_channel, output_points, kernel_size=1))

    def forward(self, x):
        embed = self.conv(x)  # b*k*n*1
        S = torch.softmax(embed, dim=2).squeeze(3) # assignment matrix S_pool, Softmax operation on 2000 points
        out = torch.matmul(x.squeeze(3), S.transpose(1, 2)).unsqueeze(3)
        return out

class trans(nn.Module):

    def __init__(self, dim1, dim2):
        super(trans, self).__init__()
        self.dim1 = dim1
        self.dim2 = dim2

    def forward(self, x):
        return x.transpose(self.dim1, self.dim2)

class OAFilter(nn.Module):

    def __init__(self, channels, points, out_channels=None):
        super(OAFilter, self).__init__()
        if not out_channels:
           out_channels = channels
        self.shot_cut = None
        if out_channels != channels:
            self.shot_cut = nn.Conv2d(channels, out_channels, kernel_size=1)

        self.conv1 = nn.Sequential(nn.InstanceNorm2d(channels, eps=1e-3),
                                   nn.BatchNorm2d(channels),
                                   nn.ReLU(),
                                   nn.Conv2d(channels, out_channels, kernel_size=1),  # b*c*n*1
                                   trans(1, 2))

        # Spatial Correlation Layer
        self.conv2 = nn.Sequential(
            nn.BatchNorm2d(points),
            nn.ReLU(),
            nn.Conv2d(points, points, kernel_size=1)
        )

        self.conv3 = nn.Sequential(
            trans(1, 2),
            nn.InstanceNorm2d(out_channels, eps=1e-3),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=1)
        )

    def forward(self, x):
        out = self.conv1(x)
        out = out + self.conv2(out)
        out = self.conv3(out)
        if self.shot_cut:
            out = out + self.shot_cut(x)
        else:
            out = out + x
        return out

class diff_unpool(nn.Module):
    def __init__(self, in_channel, output_points):
        super(diff_unpool, self).__init__()
        self.output_points = output_points
        self.conv = nn.Sequential(nn.InstanceNorm2d(in_channel, eps=1e-3),
                                  nn.BatchNorm2d(in_channel),
                                  nn.ReLU(),
                                  nn.Conv2d(in_channel, output_points, kernel_size=1))

    def forward(self, x_up, x_down):
        #x_up: b*c*n*1
        #x_down: b*c*k*1
        embed = self.conv(x_up)# b*k*n*1
        S = torch.softmax(embed, dim=1).squeeze(3)# b*k*n
        out = torch.matmul(x_down.squeeze(3), S).unsqueeze(3)
        return out

def batch_symeig(X):
    # it is much faster to run symeig on CPU
    X = X.cpu()
    b, d, _ = X.size()
    bv = X.new(b,d,d)
    for batch_idx in range(X.shape[0]):
        e,v = torch.symeig(X[batch_idx,:,:].squeeze(), True)
        bv[batch_idx,:,:] = v
    bv = bv.cuda()
    return bv

def weighted_8points(x_in, logits):
    """
    weighted eight-point algorithm to directly regress essential matrix.
    principle: https://zhuanlan.zhihu.com/p/108606635
    :param x_in: keypoints pair, [1,1,2000,4]
    :param logits: keypoints pair probability, [1,2000]
    :return:
    """

    # x_in: batch * 1 * N * 4
    x_shp = x_in.shape
    # Turn into weights for each sample
    weights = torch.relu(torch.tanh(logits)) # tanh and relu
    x_in = x_in.squeeze(1) # [1,2000,4]

    # Make input data (num_img_pair x num_corr x 4)
    xx = torch.reshape(x_in, (x_shp[0], x_shp[2], 4)).permute(0, 2, 1) # [1,4,2000]

    # Create the matrix to be used for the eight-point algorithm
    X = torch.stack([
        xx[:, 2] * xx[:, 0], xx[:, 2] * xx[:, 1], xx[:, 2],
        xx[:, 3] * xx[:, 0], xx[:, 3] * xx[:, 1], xx[:, 3],
        xx[:, 0], xx[:, 1], torch.ones_like(xx[:, 0])
    ], dim=1).permute(0, 2, 1) # [1,2000,9]
    wX = torch.reshape(weights, (x_shp[0], x_shp[2], 1)) * X # [1,2000,9]
    XwX = torch.matmul(X.permute(0, 2, 1), wX) # [1,9,9]

    # Recover essential matrix from self-adjoing eigen
    v = batch_symeig(XwX) # [1,9,9]
    e_hat = torch.reshape(v[:, :, 0], (x_shp[0], 9)) # [1,9]

    # Make unit norm just in case
    e_hat = e_hat / torch.norm(e_hat, dim=1, keepdim=True)
    return e_hat

class OANBlock(nn.Module):

    def __init__(self, net_channels, input_channel, depth, clusters):
        super(OANBlock, self).__init__()
        channels = net_channels # PointCN embedding size
        self.layer_num = depth # PointCN depth
        print('channels:' + str(channels) + ', layer_num:' + str(self.layer_num))
        self.conv1 = nn.Conv2d(input_channel, channels, kernel_size=1)

        l2_nums = clusters

        self.l1_1 = []
        for _ in range(self.layer_num//2):
            self.l1_1.append(PointCN(channels))

        self.down1 = diff_pool(channels, l2_nums)

        self.l2 = []
        for _ in range(self.layer_num//2):
            self.l2.append(OAFilter(channels, l2_nums))

        self.up1 = diff_unpool(channels, l2_nums)

        self.l1_2 = []
        self.l1_2.append(PointCN(2*channels, channels))
        for _ in range(self.layer_num//2-1):
            self.l1_2.append(PointCN(channels))

        self.l1_1 = nn.Sequential(*self.l1_1)
        self.l1_2 = nn.Sequential(*self.l1_2)
        self.l2 = nn.Sequential(*self.l2)

        self.output = nn.Conv2d(channels, 1, kernel_size=1)

    def forward(self, data, xs):
        #data: b*c*n*1
        batch_size, num_pts = data.shape[0], data.shape[2]
        x1_1 = self.conv1(data) # first conv, [b,4,2000,1] -> [b,128,2000,1]
        x1_1 = self.l1_1(x1_1) # pointCN, [b,128,2000,1] -> [b,128,2000,1]
        x_down = self.down1(x1_1) # diff_pool, [b,128,2000,1] -> [b,128,500,1]
        x2 = self.l2(x_down) # oanet_filter, [b,128,500,1] -> [b,128,500,1]
        x_up = self.up1(x1_1, x2) # diff_unpool, [b,128,500,1] -> [b,128,2000,1]
        out = self.l1_2( torch.cat([x1_1,x_up], dim=1)) # pointCN, [b,128,2000,1] -> [b,128,2000,1]

        logits = torch.squeeze(torch.squeeze(self.output(out),3),1) # to probability, [b,128,2000,1] -> [b,2000]
        e_hat = weighted_8points(xs, logits) # [1,9]

        x1, x2 = xs[:,0,:,:2], xs[:,0,:,2:4] # [1,2000,2]
        e_hat_norm = e_hat
        residual = batch_episym(x1, x2, e_hat_norm).reshape(batch_size, 1, num_pts, 1) # distance of x1 and x2? [1,1,2000,1]

        return logits, e_hat, residual


class OANet(nn.Module):

    def __init__(self, conf):
        super(OANet, self).__init__()
        self.iter_num = conf.iter_num # The number of network iterations
        depth_each_stage = conf.net_depth // (conf.iter_num + 1) # There is pointCN at the beginning and end
        self.side_channel = (conf.use_ratio == 2) + (conf.use_mutual == 2)
        self.weights_init = OANBlock(conf.net_channels, 4 + self.side_channel, depth_each_stage, conf.clusters) # 4 elements form key point pair coordinate information, which are 4 input channels
        self.weights_iter = [OANBlock(conf.net_channels, 6 + self.side_channel, depth_each_stage, conf.clusters) for _ in range(conf.iter_num)]
        self.weights_iter = nn.Sequential(*self.weights_iter)

    def forward(self, data):
        assert data['xs'].dim() == 4 and data['xs'].shape[1] == 1
        # data: b*1*n*c
        input = data['xs'].transpose(1, 3)
        if self.side_channel > 0:
            sides = data['sides'].transpose(1,2).unsqueeze(3)
            input = torch.cat([input, sides], dim=1)

        res_logits, res_e_hat = [], []
        logits, e_hat, residual = self.weights_init(input, data['xs'])
        res_logits.append(logits), res_e_hat.append(e_hat)


        for i in range(self.iter_num):
            logits, e_hat, residual = self.weights_iter[i](torch.cat([input, residual.detach(), torch.relu(torch.tanh(logits)).reshape(residual.shape).detach()],dim=1),
                                                           data['xs']) # residual and logits info add to input
            res_logits.append(logits), res_e_hat.append(e_hat)

        return res_logits, res_e_hat

def test():

    k1s = torch.randn(size=(32,1)).cuda().float() # essential matrix estimation
    k2s = torch.randn(size=(32, 1)).cuda().float()
    rs = torch.randn(size=(32, 3, 3)).cuda().float()
    ts = torch.randn(size=(32, 3, 1)).cuda().float()
    xs = torch.randn(size=(32, 1, 2000, 4)).cuda().float()
    ys = torch.randn(size=(32, 2000, 1)).cuda().float()
    t1s = torch.randn(size=(32, 1)).cuda().float() # essential matrix estimation
    t2s = torch.randn(size=(32, 1)).cuda().float()
    virtpts = torch.randn(size=(32, 400, 4)).cuda().float()
    sides = []

    data = {}
    dict_key = ['K1s', 'K2s', 'Rs', 'ts', 'xs', 'ys', 'T1s', 'T2s', 'virtPts', 'sides']
    dict_value = [k1s, k2s, rs, ts, xs, ys, t1s, t2s, virtpts, sides]
    for index, k in enumerate(dict_key):
        data[k] = dict_value[index]

    conf = Config()
    model = OANet(conf).cuda()
    logits, e_hat = model(data)

    print(logits[0].size(), e_hat[0].size(), logits[1].size(), e_hat[1].size())

if __name__ == '__main__':
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '5'
    test()