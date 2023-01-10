from libs.dataset.data import *
import os
import yaml
from PIL import Image
import numpy as np
import random

class Davis17(BaseDataset):
    def __init__(self, train=True, frames_sampled=3,
                 transform=None, max_skip=50, increment_per_epoches=5, video_sampled=1):

        data_dir = os.path.join(DATA_ROOT, "DAVIS17", "train")
        db_file = os.path.join(DATA_ROOT, 'DAVIS17', 'db_info.yaml')

        self.root = data_dir
        self.videos_root = os.path.join(self.root, 'JPEGImages', '480p')
        self.annos_root = os.path.join(self.root, 'Annotations', '480p')
        self.max_obj = 0

        # extract annotation information
        with open(db_file, 'r') as f:
            db = yaml.load(f, Loader=yaml.Loader)['sequences']
            self.info = db

            mode = 'train' if train else 'val'
            self.videos_name = [info['name'] for info in db if info['set'] == mode]

            # max_obj
            for vid in self.videos_name:
                objn = np.array(Image.open(os.path.join(self.annos_root, vid, '00000.png'))).max()
                self.max_obj = max(objn, self.max_obj)

        self.video_sampled = video_sampled # 视频采样
        self.frames_samples = frames_sampled # 帧采样
        self.length = video_sampled * len(self.videos_name)
        self.max_skip = max_skip
        self.increment = increment_per_epoches

        self.transform = transform
        self.train = train

    def increase_max_skip(self):
        self.max_skip = min(self.max_skip + self.increment, MAX_TRAINING_SKIP)

    def set_max_skip(self, max_skip):
        self.max_skip = max_skip

    # 输出一个视频tensor，[T, C, H, W]
    def __getitem__(self, idx):

        vid = self.videos_name[(idx // self.video_sampled)]

        frames_floder = os.path.join(self.videos_root, vid)
        annos_floder = os.path.join(self.annos_root, vid)

        frames_name = [frame[:5] for frame in os.listdir(frames_floder)]
        frames_name.sort()
        nframes = len(frames_name)

        num_obj = 0
        while num_obj == 0:

            try:
                if self.train:
                    last_sample = -1
                    frame_samples = []

                    # 挑选frames_samples，保证先后顺序，但不连续
                    # 当视频总帧数小于frames_sampled时，采样数为视频总帧数
                    # 第一帧是保证采样数的情况下，在一个范围随机选择
                    # 而后的frames_samples，如果不满足max_skip，就在剩下范围随机选择
                    nsamples = min(nframes, self.frames_samples)
                    for i in range(nsamples):
                        if i == 0:
                            last_sample = random.sample(range(0, nframes - nsamples + 1), 1)[0]
                        else:
                            last_sample = random.sample(
                                range(last_sample + 1, min(last_sample + self.max_skip + 1, nframes - nsamples + i + 1)), 1)[0]
                        frame_samples.append(frames_name[last_sample])

                # 测试：视频不采样，返回的视频每帧的列表
                else:
                    frame_samples = frames_name

                frames_list = [np.array(Image.open(os.path.join(frames_floder, name + '.jpg'))) for name in frame_samples]
                masks_list = [np.array(Image.open(os.path.join(annos_floder, name + '.png'))) for name in frame_samples]

                # clear dirty data
                for msk in masks_list:
                    msk[msk == 255] = 0

                num_obj = masks_list[0].max()

            # 捕获采样过程的错误导致file not found
            except FileNotFoundError as fnfe:
                # placeholder
                print('[WARNING] build place holder for video mask')
                masks_list = [np.array(Image.open(os.path.join(annos_floder, '00000.png')))] * nframes

                # clear dirty data
                for msk in masks_list:
                    msk[msk == 255] = 0

                num_obj = masks_list[0].max()

            except OSError as ose:
                num_obj = 0
                continue

        # 控制训练过程的num_obj数量，保证训练效率
        if self.train:
            num_obj = min(num_obj, MAX_TRAINING_OBJ)

        masks_list = [mask_convert_to_oh(msk, self.max_obj) for msk in masks_list]

        info = {'name': vid}
        info['palette'] = Image.open(os.path.join(annos_floder, frames_name[0] + '.png')).getpalette()
        info['size'] = frames_list[0].shape[:2]

        if self.transform is None:
            raise RuntimeError('Lack of proper transformation')

        try:
            frames, masks = self.transform(frames_list, masks_list)
        except Exception as exp:
            print(exp)
            print('Interruption at samples:')
            print(vid, frame_samples)
            exit()

        return frames, masks, num_obj, info

    def __len__(self):
        return self.length

    @staticmethod
    def collate_fn(batch):

        frames = torch.stack([sample[0] for sample in batch])
        masks = torch.stack([sample[1] for sample in batch])

        objs = [sample[2] for sample in batch]

        try:
            info = [sample[3] for sample in batch]
        except IndexError as ie:
            info = None

        return frames, masks, objs, info


def test():
    from libs.dataset import TrainTransform

    ds = Davis17(train=True, frames_sampled=3, transform=TrainTransform)


# 将davis类注册进data的私有变量中
dataset_register("DAVIS17", Davis17)