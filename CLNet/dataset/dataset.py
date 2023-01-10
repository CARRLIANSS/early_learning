from torch.utils.data import Dataset
import numpy as np
import cv2
import h5py
from utils.utils import np_skew_symmetric
import torch

class CorrespondencesDataset(Dataset):

    def __init__(self, filename, config):
        self.config = config
        self.filename = filename # dump data
        self.data = None

    def norm_input(self, x):
        x_mean = np.mean(x, axis=0)  # [2,]
        dist = x - x_mean
        meandist = np.sqrt((dist ** 2).sum(axis=1)).mean()  # float
        scale = np.sqrt(2) / meandist
        T = np.zeros([3, 3])
        T[0, 0], T[1, 1], T[2, 2] = scale, scale, 1
        T[0, 2], T[1, 2] = -scale * x_mean[0], -scale * x_mean[1]
        x = x * np.asarray([T[0, 0], T[1, 1]]) + np.array([T[0, 2], T[1, 2]])
        return x, T

    def correctMatches(self, e_gt):
        step = 0.1
        xx, yy = np.meshgrid(np.arange(-1, 1, step), np.arange(-1, 1, step))  # [20,20]
        # Points in first image before projection
        pts1_virt_b = np.float32(np.vstack((xx.flatten(), yy.flatten())).T)
        # Points in second image before projection
        pts2_virt_b = np.float32(pts1_virt_b)
        pts1_virt_b, pts2_virt_b = pts1_virt_b.reshape(1, -1, 2), pts2_virt_b.reshape(1, -1, 2)

        pts1_virt_b, pts2_virt_b = cv2.correctMatches(e_gt.reshape(3, 3), pts1_virt_b, pts2_virt_b)

        return pts1_virt_b.squeeze(), pts2_virt_b.squeeze()

    def __getitem__(self, index):
        if self.data is None:
            self.data = h5py.File(self.filename,'r')

        xs = np.asarray(self.data['xs'][str(index)]) # keypoints pair
        ys = np.asarray(self.data['ys'][str(index)]).squeeze(-1) # keypoints distance, Used for internal and external point judgment
        R = np.asarray(self.data['Rs'][str(index)]) # extrinsic
        t = np.asarray(self.data['ts'][str(index)]) # extrinsic

        side = [] # Additional information for input
        if self.config.use_ratio == 0 and self.config.use_mutual == 0:
            pass
        elif self.config.use_ratio == 1 and self.config.use_mutual == 0: # Filter matching keypoints by probability before network
            mask = np.asarray(self.data['ratios'][str(index)]).reshape(-1) < self.config.ratio_test_th # ratio_test_th=0.8
            xs = xs[:, mask, :]
            ys = ys[mask, :]
        elif self.config.use_ratio == 0 and self.config.use_mutual == 1: # Filter matching keypoints by MNN before network
            mask = np.asarray(self.data['mutuals'][str(index)]).reshape(-1).astype(bool)
            xs = xs[:, mask, :]
            ys = ys[mask, :]
        elif self.config.use_ratio == 0 and self.config.use_mutual == 2: # Add use_mutual info to network input
            side = np.asarray(self.data['mutuals'][str(index)]).reshape(-1, 1)
        elif self.config.use_ratio == 2 and self.config.use_mutual == 0: # Add use_ratio info to network input
            side = np.asarray(self.data['ratios'][str(index)]).reshape(-1, 1)
        elif self.config.use_ratio == 2 and self.config.use_mutual == 2: # Add use_mutual and use_ratio info to network input
            side.append(np.asarray(self.data['ratios'][str(index)]).reshape(-1, 1))
            side.append(np.asarray(self.data['mutuals'][str(index)]).reshape(-1, 1))
            side = np.concatenate(side, axis=-1)
        else:
            raise NotImplementedError

        e_gt_unnorm = np.reshape(np.matmul(
            np.reshape(np_skew_symmetric(t.astype('float64').reshape(1, 3)), (3, 3)),
            np.reshape(R.astype('float64'), (3, 3))), (3, 3)) # groundtruth of essential matrix
        e_gt = e_gt_unnorm / np.linalg.norm(e_gt_unnorm) # divide by norm/范数

        if self.config.use_fundamental: # train fundamental matrix estimation
            # intrinsics
            cx1 = np.asarray(self.data['cx1s'][str(index)])
            cy1 = np.asarray(self.data['cy1s'][str(index)])
            cx2 = np.asarray(self.data['cx2s'][str(index)])
            cy2 = np.asarray(self.data['cy2s'][str(index)])
            f1 = np.asarray(self.data['f1s'][str(index)])
            f2 = np.asarray(self.data['f2s'][str(index)])
            f1 = f1[0] if f1.ndim == 2 else f1
            f2 = f2[0] if f2.ndim == 2 else f2
            # pack
            K1 = np.asarray([
                [f1[0], 0, cx1[0]],
                [0, f1[1], cy1[0]],
                [0, 0, 1]
            ])
            K2 = np.asarray([
                [f2[0], 0, cx2[0]],
                [0, f2[1], cy2[0]],
                [0, 0, 1]
            ])

            # keypoint coordinate
            x1, x2 = xs[0, :, :2], xs[0, :, 2:4] # unpack
            x1 = x1 * np.asarray([K1[0, 0], K1[1, 1]]) + np.array([K1[0, 2], K1[1, 2]]) # Some operations of coordinate and K
            x2 = x2 * np.asarray([K2[0, 0], K2[1, 1]]) + np.array([K2[0, 2], K2[1, 2]])

            # norm input
            x1, T1 = self.norm_input(x1)  # T contains some parameters of the norm operation
            x2, T2 = self.norm_input(x2)

            xs = np.concatenate([x1, x2], axis=-1).reshape(1, -1, 4) # pack

            # get F
            e_gt = np.matmul(np.matmul(np.linalg.inv(K2).T, e_gt), np.linalg.inv(K1)) # [K2^(-1) * e_gt] * K1, gt of fundamental matrix
            # get F after norm
            e_gt_unnorm = np.matmul(np.matmul(np.linalg.inv(T2).T, e_gt), np.linalg.inv(T1))
            e_gt = e_gt_unnorm / np.linalg.norm(e_gt_unnorm)
        else:
            # If it is to find the fundamental matrix, it needs K matrix to operate,
            # if it is to find the essential matrix, it does not need.
            K1, K2 = np.zeros(1), np.zeros(1)
            T1, T2 = np.zeros(1), np.zeros(1)

        pts1_virt, pts2_virt = self.correctMatches(e_gt) # [400,2], Corrected coordinates
        pts_virt = np.concatenate([pts1_virt, pts2_virt], axis=1).astype('float64') # [400,4]

        return {'K1': K1, 'K2': K2, 'R': R, 't': t, 'xs': xs, 'ys': ys, 'T1': T1, 'T2': T2, 'virtPt': pts_virt, 'side': side}

    def reset(self):
        if self.data is not None:
            self.data.close()
        self.data = None

    def __len__(self):
        if self.data is None:
            self.data = h5py.File(self.filename,'r')
            _len = len(self.data['xs'])
            self.data.close()
            self.data = None
        else:
            _len = len(self.data['xs'])
            self.data.close()
            self.data = None
        return _len

    def __del__(self):
        if self.data is not None:
            self.data.close()
            self.data = None


def collate_fn(batch):
    batch_size = len(batch)
    numkps = np.array([sample['xs'].shape[1] for sample in batch]) # [2000]
    cur_num_kp = int(numkps.min()) # This code works if filter keypoints are used before the network

    data = {}
    dict_key = ['K1s', 'K2s', 'Rs', 'ts', 'xs', 'ys', 'T1s', 'T2s', 'virtPts', 'sides']
    for k in dict_key:
        data[k] = []

    for sample in batch:
        # intrinsics and extrinsics
        data['K1s'].append(sample['K1'])
        data['K2s'].append(sample['K2'])
        data['T1s'].append(sample['T1'])
        data['T2s'].append(sample['T2'])
        data['Rs'].append(sample['R'])
        data['ts'].append(sample['t'])
        data['virtPts'].append(sample['virtPt'])

        # keypoints
        if sample['xs'].shape[1] > cur_num_kp:
            sub_idx = np.random.choice(sample['xs'].shape[1], cur_num_kp, replace=False) # cur_num_kp random numbers from 0 to sample['xs'].shape[1]
            data['xs'].append(sample['xs'][:, sub_idx, :]) # Reduce the number of keypoints to cur_num_kp
            data['ys'].append(sample['ys'][sub_idx])
            if len(sample['side']) != 0:
                data['sides'].append(sample['side'][sub_idx,:])
        else:
            data['xs'].append(sample['xs'])
            data['ys'].append(sample['ys'])
            if len(sample['side']) != 0:
                data['sides'].append(sample['side'])

    for key in dict_key[:-1]:
        data[key] = torch.from_numpy(np.stack(data[key])).float() # item of batch to stack
    if data['sides'] != []:
        data['sides'] = torch.from_numpy(np.stack(data['sides'])).float()

    return data