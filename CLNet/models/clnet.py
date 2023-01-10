import torch
import torch.nn as nn

def batch_episym(x1, x2, F):
    batch_size, num_pts = x1.shape[0], x1.shape[1]
    x1 = torch.cat([x1, x1.new_ones(batch_size, num_pts,1)], dim=-1).reshape(batch_size, num_pts,3,1)
    x2 = torch.cat([x2, x2.new_ones(batch_size, num_pts,1)], dim=-1).reshape(batch_size, num_pts,3,1)
    F = F.reshape(-1,1,3,3).repeat(1,num_pts,1,1)
    x2Fx1 = torch.matmul(x2.transpose(2,3), torch.matmul(F, x1)).reshape(batch_size,num_pts)
    Fx1 = torch.matmul(F,x1).reshape(batch_size,num_pts,3)
    Ftx2 = torch.matmul(F.transpose(2,3),x2).reshape(batch_size,num_pts,3)

    ys = x2Fx1**2 * (
            1.0 / (Fx1[:, :, 0]**2 + Fx1[:, :, 1]**2 + 1e-15) +
            1.0 / (Ftx2[:, :, 0]**2 + Ftx2[:, :, 1]**2 + 1e-15))

    return ys

def knn(x, k):
    """
    对于每个pairwise，找k个与之距离最近的pairwise，并记录他们的index
    """
    inner = -2*torch.matmul(x.transpose(2, 1), x) # [32, 2000, 168] * [32, 168, 2000] -> [32, 2000, 2000]
    xx = torch.sum(x**2, dim=1, keepdim=True) # [32, 168, 2000] -> [32, 1, 2000]
    pairwise_distance = -xx - inner - xx.transpose(2, 1) # [32, 2000, 2000]

    idx = pairwise_distance.topk(k=k, dim=-1)[1]   # [32, 2000, k], 返回的是index

    return idx[:, :, :]

def get_graph_feature(x, k=20, idx=None):
    """
    对于每个pairwise，在embedding长度维度上进行融合，前半部分表示embedding本身，后半部分表示与之相邻的pairwise的embedding
    """
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points) # [32, 128, 2000, 1] -> [32, 128, 2000]
    if idx is None:
        idx_out = knn(x, k=k) # [32, 128, 2000] -> [32, 2000, 9]
    else:
        idx_out = idx
    device = x.device

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points # [32, 1, 1]

    idx = idx_out + idx_base # 以一个batch为单位进行index排序，[32, 2000, 9]

    idx = idx.view(-1) # 该batch每个pairwise邻近的k个pairwise的下标序列，32 * 2000 * 9

    _, num_dims, _ = x.size() # 128

    x = x.transpose(2, 1).contiguous() # [32, 2000, 128]
    feature = x.view(batch_size*num_points, -1)[idx, :] # [32 * 2000 * 9, 128]，以batch为单位的pairwise相邻k个pairwise的embedding
    feature = feature.view(batch_size, num_points, k, num_dims) # [32 * 2000 * 9, 128] -> [32, 2000, 9, 128]
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1) # [32, 2000, 1, 128] -> [32, 2000, 9, 128]
    # 在embedding长度维度上进行融合，前半部分表示embedding本身，后半部分表示与之相邻的pairwise的与本身残差
    feature = torch.cat((x, x - feature), dim=3).permute(0, 3, 1, 2).contiguous() # [32, 256, 2000, 9]
    return feature

class ResNet_Block(nn.Module):
    """
    结构与resnet block一致，只是加了IN层、用的是1*1卷积表示MLP
    """
    def __init__(self, inchannel, outchannel, downsample=None):
        super(ResNet_Block, self).__init__()

        self.downsample = nn.Conv2d(inchannel, outchannel, (1, 1)) if downsample is not None else None

        self.conv1 = nn.Conv2d(inchannel, outchannel, (1, 1))
        self.in1 = nn.InstanceNorm2d(outchannel)
        self.bn1 = nn.BatchNorm2d(outchannel)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(outchannel, outchannel, (1, 1))
        self.in2 = nn.InstanceNorm2d(outchannel)
        self.bn2 = nn.BatchNorm2d(outchannel)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.in1(out)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.in2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual
        out = self.relu(out)

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
    # x_in: [32, 1, 500, 4], logits: [32, 2, 500, 1]
    mask = logits[:, 0, :, 0] # [32, 500]
    weights = logits[:, 1, :, 0] # [32, 500]

    mask = torch.sigmoid(mask)
    # softmax的基础上乘于一个系数
    weights = torch.exp(weights) * mask
    weights = weights / (torch.sum(weights, dim=-1, keepdim=True) + 1e-5)

    x_shp = x_in.shape
    x_in = x_in.squeeze(1)

    xx = torch.reshape(x_in, (x_shp[0], x_shp[2], 4)).permute(0, 2, 1).contiguous()

    X = torch.stack([
        xx[:, 2] * xx[:, 0], xx[:, 2] * xx[:, 1], xx[:, 2],
        xx[:, 3] * xx[:, 0], xx[:, 3] * xx[:, 1], xx[:, 3],
        xx[:, 0], xx[:, 1], torch.ones_like(xx[:, 0])
    ], dim=1).permute(0, 2, 1).contiguous()
    wX = torch.reshape(weights, (x_shp[0], x_shp[2], 1)) * X
    XwX = torch.matmul(X.permute(0, 2, 1).contiguous(), wX)

    # Recover essential matrix from self-adjoing eigen, eigen是开源线性代数库，可以很快的进行矩阵计算
    v = batch_symeig(XwX)  # [1,9,9]，求出XwX的特征向量
    e_hat = torch.reshape(v[:, :, 0], (x_shp[0], 9))  # [1,9]

    # Make unit norm just in case
    e_hat = e_hat / torch.norm(e_hat, dim=1, keepdim=True)
    return e_hat

class DGCNN_Block(nn.Module):
    def __init__(self, knn_num=9, in_channel=128):
        super(DGCNN_Block, self).__init__()
        self.knn_num = knn_num
        self.in_channel = in_channel

        assert self.knn_num == 9 or self.knn_num == 6
        if self.knn_num == 9:
            self.conv = nn.Sequential(
                nn.Conv2d(self.in_channel*2, self.in_channel, (1, 3), stride=(1, 3)), # 降低最后一个维度至3
                nn.BatchNorm2d(self.in_channel),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.in_channel, self.in_channel, (1, 3)), # 降低最后一个维度至1
                nn.BatchNorm2d(self.in_channel),
                nn.ReLU(inplace=True),
            )
        if self.knn_num == 6:
            self.conv = nn.Sequential(
                nn.Conv2d(self.in_channel*2, self.in_channel, (1, 3), stride=(1, 3)), # 降低最后一个维度至2
                nn.BatchNorm2d(self.in_channel),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.in_channel, self.in_channel, (1, 2)), # 降低最后一个维度至1
                nn.BatchNorm2d(self.in_channel),
                nn.ReLU(inplace=True),
            )

    def forward(self, features):
        B, _, N, _ = features.shape # [32, 128, 2000, 1]
        out = get_graph_feature(features, k=self.knn_num) # [32, 128, 2000, 1] -> [32, 256, 2000, 9]
        out = self.conv(out) # [32, 256, 2000, 9] -> [32, 128, 2000, 1]
        return out

class GCN_Block(nn.Module):
    def __init__(self, in_channel):
        super(GCN_Block, self).__init__()
        self.in_channel = in_channel
        self.conv = nn.Sequential(
            nn.Conv2d(self.in_channel, self.in_channel, (1, 1)),
            nn.BatchNorm2d(self.in_channel),
            nn.ReLU(inplace=True),
        )

    def attention(self, w):
        w = torch.relu(torch.tanh(w)).unsqueeze(-1) # [32, 2000] -> [32, 2000, 1]
        A = torch.bmm(w, w.transpose(1, 2)) # W*W^T, [32, 2000, 1] -> [32, 2000, 2000]
        return A

    def graph_aggregation(self, x, w):
        B, _, N, _ = x.size()
        with torch.no_grad(): # 可以写到forward中
            A = self.attention(w) # [32, 2000, 2000], 每个pairwise（包含空间信息）的相互关系矩阵
            I = torch.eye(N).unsqueeze(0).to(x.device).detach() # 单位矩阵, [1, 2000, 2000]
            A = A + I # [32, 2000, 2000], 对角线大于1
            D_out = torch.sum(A, dim=-1) # [32, 2000, 2000] -> [32, 2000]
            D = (1 / D_out) ** 0.5
            D = torch.diag_embed(D) # 将D放置对角线上，[32, 2000, 2000]
            L = torch.bmm(D, A) # 将相互关系矩阵每行乘于一个系数，[32, 2000, 2000]
            L = torch.bmm(L, D) # 将相互关系矩阵每列乘于一个系数，[32, 2000, 2000]
        out = x.squeeze(-1).transpose(1, 2).contiguous() # [32, 128, 2000, 1] -> [32, 2000, 128]
        out = torch.bmm(L, out).unsqueeze(-1) # 相互关系作为激励乘于out
        out = out.transpose(1, 2).contiguous() # [32, 128, 2000, 1]

        return out

    def forward(self, x, w):
        out = self.graph_aggregation(x, w) # [32, 128, 2000, 1]
        out = self.conv(out) # MLP, [32, 128, 2000, 1]
        return out

class DS_Block(nn.Module):
    def __init__(self, initial=False, predict=False, out_channel=128, k_num=8, sampling_rate=0.5):
        super(DS_Block, self).__init__()
        self.initial = initial
        self.in_channel = 4 if self.initial is True else 6
        self.out_channel = out_channel
        self.k_num = k_num
        self.predict = predict
        self.sr = sampling_rate

        self.conv = nn.Sequential(
            nn.Conv2d(self.in_channel, self.out_channel, (1, 1)),
            nn.BatchNorm2d(self.out_channel),
            nn.ReLU(inplace=True)
        )

        self.gcn = GCN_Block(self.out_channel)

        self.embed_0 = nn.Sequential(
            ResNet_Block(self.out_channel, self.out_channel, downsample=None),
            ResNet_Block(self.out_channel, self.out_channel, downsample=None),
            ResNet_Block(self.out_channel, self.out_channel, downsample=None),
            ResNet_Block(self.out_channel, self.out_channel, downsample=None),
            DGCNN_Block(self.k_num, self.out_channel),
            ResNet_Block(self.out_channel, self.out_channel, downsample=None),
            ResNet_Block(self.out_channel, self.out_channel, downsample=None),
            ResNet_Block(self.out_channel, self.out_channel, downsample=None),
            ResNet_Block(self.out_channel, self.out_channel, downsample=None),
        )
        self.embed_1 = nn.Sequential(
            ResNet_Block(self.out_channel, self.out_channel, downsample=None),
        )
        self.linear_0 = nn.Conv2d(self.out_channel, 1, (1, 1))
        self.linear_1 = nn.Conv2d(self.out_channel, 1, (1, 1))

        if self.predict == True:
            self.embed_2 = ResNet_Block(self.out_channel, self.out_channel, downsample=None)
            self.linear_2 = nn.Conv2d(self.out_channel, 2, (1, 1))

    def down_sampling(self, x, y, weights, indices, features=None, predict=False):
        """
        对x、y、w0的embedding数量下采样
        """
        B, _, N , _ = x.size()
        indices = indices[:, :int(N*self.sr)] # 取前50%
        with torch.no_grad():
            y_out = torch.gather(y, dim=-1, index=indices) # 取系数前50%的distance
            w_out = torch.gather(weights, dim=-1, index=indices) # 取系数前50%的w0
        indices = indices.view(B, 1, -1, 1) # [32, 1000] -> [32, 1, 1000, 1]

        if predict == False:
            with torch.no_grad():
                x_out = torch.gather(x[:, :, :, :4], dim=2, index=indices.repeat(1, 1, 1, 4)) # 取系数前50%的x
            return x_out, y_out, w_out # 对x、y、w0的embedding数量下采样
        else:
            with torch.no_grad():
                x_out = torch.gather(x[:, :, :, :4], dim=2, index=indices.repeat(1, 1, 1, 4))
            feature_out = torch.gather(features, dim=2, index=indices.repeat(1, 128, 1, 1)) # [32, 128, 1000, 1] -> [32, 128, 500, 1]
            return x_out, y_out, w_out, feature_out

    def forward(self, x, y):
        B, _, N , _ = x.size()
        out = x.transpose(1, 3).contiguous() # [32, 1, 2000, 4] -> [32, 4, 2000, 1]
        out = self.conv(out) # [32, 4, 2000, 1] -> [32, 128, 2000, 1]

        out = self.embed_0(out) # [32, 128, 2000, 1] -> [32, 128, 2000, 1], 捕获到局部/空间信息
        w0 = self.linear_0(out).view(B, -1) # 压缩, [32, 128, 2000, 1] -> [32, 2000]

        # linear_0只有一个参数，故将其固定不参与学习，不然每次变化太大
        out_g = self.gcn(out, w0.detach()) # 生成关系矩阵作为激励乘于out，[32, 128, 2000, 1] -> [32, 128, 2000, 1]
        out = out_g + out # [32, 128, 2000, 1]

        out = self.embed_1(out) # [32, 128, 2000, 1] -> [32, 128, 2000, 1]
        w1 = self.linear_1(out).view(B, -1) # [32, 128, 2000, 1] -> [32, 2000]

        if self.predict == False:
            w1_ds, indices = torch.sort(w1, dim=-1, descending=True) # 对2000个压缩内容排序，[32, 2000] -> [32, 2000]
            w1_ds = w1_ds[:, :int(N*self.sr)] # 取前50%
            x_ds, y_ds, w0_ds = self.down_sampling(x, y, w0, indices, None, self.predict) # 对x、y、w0的embedding数量下采样

            return x_ds, y_ds, [w0, w1], [w0_ds, w1_ds]
        else:
            w1_ds, indices = torch.sort(w1, dim=-1, descending=True)
            w1_ds = w1_ds[:, :int(N*self.sr)]
            x_ds, y_ds, w0_ds, out = self.down_sampling(x, y, w0, indices, out, self.predict)
            out = self.embed_2(out) # [32, 128, 500, 1] -> [32, 128, 500, 1]
            w2 = self.linear_2(out) # [32, 128, 500, 1] -> [32, 2, 500, 1]
            e_hat = weighted_8points(x_ds, w2)

            return x_ds, y_ds, [w0, w1, w2[:, 0, :, 0]], [w0_ds, w1_ds], e_hat

class CLNet(nn.Module):
    def __init__(self, config):
        super(CLNet, self).__init__()

        self.ds_0 = DS_Block(initial=True, predict=False, out_channel=128, k_num=9, sampling_rate=config.sr)#1.0)
        self.ds_1 = DS_Block(initial=False, predict=True, out_channel=128, k_num=6, sampling_rate=config.sr)

    def forward(self, x, y):
        B, _, N, _ = x.shape

        # x1: [32, 1, 1000, 4]
        # y1: [32, 1000]
        # ws0: local_[32, 2000], global_[32, 2000]
        # w_ds0: local_[32, 1000], global_[32, 1000]
        x1, y1, ws0, w_ds0 = self.ds_0(x, y)

        w_ds0[0] = torch.relu(torch.tanh(w_ds0[0])).reshape(B, 1, -1, 1) # [32, 1, 1000, 1]
        w_ds0[1] = torch.relu(torch.tanh(w_ds0[1])).reshape(B, 1, -1, 1) # [32, 1, 1000, 1]
        x_ = torch.cat([x1, w_ds0[0].detach(), w_ds0[1].detach()], dim=-1) # [32, 1, 1000, 6]

        # x2: [32, 1, 500, 4]
        # y2: [32, 500]
        # ws1: local_[32, 1000], global_[32, 1000], final_[32, 500]
        # w_ds1: local_[32, 500], global_[32, 500]
        # e_hat: [32, 9]
        x2, y2, ws1, w_ds1, e_hat = self.ds_1(x_, y1)

        with torch.no_grad():
            # 用e_hat估计的对极限距离
            y_hat = batch_episym(x[:, 0, :, :2], x[:, 0, :, 2:], e_hat) # [32, 2000]

        return ws0 + ws1, [y, y, y1, y1, y2], [e_hat], y_hat