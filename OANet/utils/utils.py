import numpy as np
import random
import torch

def np_skew_symmetric(v):
    """
    生成互为相反数的斜对称矩阵
    :param v: extrinsics t
    :return: skew symmetric
    """
    zero = np.zeros_like(v[:, 0])

    M = np.stack([
        zero, -v[:, 2], v[:, 1],
        v[:, 2], zero, -v[:, 0],
        -v[:, 1], v[:, 0], zero,
    ], axis=1)

    return M

def torch_skew_symmetric(v):

    zero = torch.zeros_like(v[:, 0])

    M = torch.stack([
        zero, -v[:, 2], v[:, 1],
        v[:, 2], zero, -v[:, 0],
        -v[:, 1], v[:, 0], zero,
    ], dim=1)

    return M

def set_seed(seed):
    """
    Set the random seed
    """
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def tocuda(data):
    # convert tensor data in dictionary to cuda when it is a tensor
    for key in data.keys():
        if type(data[key]) == torch.Tensor:
            data[key] = data[key].cuda()

    return data