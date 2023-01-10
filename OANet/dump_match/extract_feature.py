import cv2
import numpy as np
import h5py
import os
import glob
from tqdm import tqdm
import argparse

class Config(object):

    def __init__(self, input_path):
        self.input_path = input_path # yfcc100m, sun3d_train, sun3d_test
        self.img_glob = '*/*/images/*.jpg' # Glob match
        self.num_kp = 2000 # keypoint number
        self.suffix = 'sift-2000' # suffix of filename


class ExtractSIFT(object):
    """
    Extract key point information and describer by SIFT algorithm for each image.
    """
    def __init__(self, num_kp, contrastThreshold=1e-5):
        self.sift = cv2.SIFT_create(nfeatures=num_kp, contrastThreshold=contrastThreshold)

    def run(self, img_path):
        img = cv2.imread(img_path)
        cv_kp, desc = self.sift.detectAndCompute(img, None) # keypoint and describer

        # keypoint attribution: keypoint's coordinate, keypoint's diameter, keypoint's direction/gradient' direction
        kp = np.array([[_kp.pt[0], _kp.pt[1], _kp.size, _kp.angle] for _kp in cv_kp])
        return kp, desc


def dump_feature(pts, desc, filename):
    """
    Offline storage in hdf5 format.
    """
    with h5py.File(filename, "w") as ifp:
        ifp.create_dataset('keypoints', pts.shape, dtype=np.float32)
        ifp.create_dataset('descriptors', desc.shape, dtype=np.float32)
        ifp["keypoints"][:] = pts
        ifp["descriptors"][:] = desc


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='extract sift.')
    parser.add_argument('--input_path', type=str, default='/home/OANet/author/raw_data/yfcc100m/',
                        help='Image directory or movie file or "camera" (for webcam).')
    opt = parser.parse_args()

    conf = Config(opt.input_path)

    detector = ExtractSIFT(conf.num_kp)

    # get image lists
    glob_path = os.path.join(conf.input_path, conf.img_glob)
    listing = glob.glob(glob_path)

    for img_path in tqdm(listing):
        kp, desc = detector.run(img_path)
        save_path = img_path + '.' + conf.suffix + '.hdf5' # dump current directory
        dump_feature(kp, desc, save_path)