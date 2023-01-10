import torch
import numpy as np
from ..utils.logger import getLogger
from torch.utils.data import Dataset

DATA_ROOT = '/home/VOS/DATA_ROOT'
MAX_TRAINING_OBJ = 7 # 训练处理最多的目标数
MAX_TRAINING_SKIP = 25 # 采样帧的skip

__DATASET_CONTAINER = {}

def dataset_register(name, dataset):
    if name in __DATASET_CONTAINER:
        raise TypeError('dataset with name {} has already been registered'.format(name))

    __DATASET_CONTAINER[name] = dataset
    dataset.set_alias(name)

def build_dataset(name, *args, **kwargs):

    logger = getLogger(__name__)

    if name not in __DATASET_CONTAINER:
        logger.error('invalid dataset name is encountered. The current acceptable datasets are:')
        support_sets = ' '.join(list(__DATASET_CONTAINER.keys()))
        logger.error(support_sets)
        raise TypeError('name not found for dataset {}'.format(name))

    return __DATASET_CONTAINER[name](*args, **kwargs) # 实例化该dataset对象

# 多通道单目标转换为单通道多目标PAL8
def oh_convert_to_mask(oh, max_obj):

    if isinstance(oh, np.ndarray):
        mask = np.zeros(oh.shape[:2], dtype=np.uint8)
    else:
        mask = torch.zeros(oh.shape[:2])

    for k in range(max_obj+1):
        mask[oh[:, :, k]==1] = k

    return mask

# 单通道多目标PAL8图，转换为多通道单目标图
def mask_convert_to_oh(mask, max_obj):

    oh = []
    for k in range(max_obj+1):
        oh.append(mask==k)

    if isinstance(mask, np.ndarray):
        oh = np.stack(oh, axis=-1)
    else:
        oh = torch.cat(oh, dim=-1).float()

    return oh

class BaseDataset(Dataset):
    """
    davis and youtube_vos base dataset.
    """

    alias = None

    def increase_max_skip(self):
        pass

    def set_max_skip(self, max_skip):
        pass

    @classmethod
    def set_alias(cls, name):
        cls.alias = name

    @classmethod
    def get_alias(cls, name):
        return cls.alias