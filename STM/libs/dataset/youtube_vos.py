from libs.dataset.data import *
import os
from PIL import Image
import numpy as np
import random

# 只做训练集（尝试离线将val数据与train合并）
class YoutubeVOS(BaseDataset):
    def __init__(self, train=True, frames_sampled=3,
                 transform=None, max_skip=50, increment_per_epoches=5, video_sampled=1):
        data_dir = os.path.join(DATA_ROOT, "YoutubeVOS")

        # mode = 'train' if train else 'val' # 离线合并不分train、val
        self.root = data_dir
        self.videos_root = os.path.join(self.root, 'train', 'JPEGImages')
        self.annos_root = os.path.join(self.root, 'train', 'Annotations')
        self.max_obj = 12

        self.videos_name = os.listdir(self.videos_root)

        # # max_obj
        # for vid in self.videos_name:
        #     ann1 = os.listdir(os.path.join(self.annos_root, vid))[0]
        #     objn = np.array(Image.open(os.path.join(self.annos_root, vid, ann1))).max()
        #     self.max_obj = max(objn, self.max_obj)

        self.video_sampled = video_sampled  # 视频采样
        self.frames_samples = frames_sampled  # 帧采样
        self.length = video_sampled * len(self.videos_name)  # 视频个数
        self.max_skip = max_skip
        self.increment = increment_per_epoches

        self.transform = transform

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
        num_obj = min(num_obj, MAX_TRAINING_OBJ)

        masks_list = [mask_convert_to_oh(msk, self.max_obj) for msk in masks_list]

        info = {'name': vid}
        info['palette'] = Image.open(os.path.join(annos_floder, frames_name[0] + '.png')).getpalette()
        info['size'] = frames_list[0].shape[:2]

        if self.transform is None:
            raise RuntimeError('Lack of proper transformation')

        frames, masks = self.transform(frames_list, masks_list)

        return frames, masks, num_obj, info

    def __len__(self):
        return self.length


dataset_register("YoutubeVOS", YoutubeVOS)