from dump_match.dataset import Dataset
import os

class Config(object):

    def __init__(self):

        self.raw_data_path = '/home/OANet/author/raw_data/'
        self.dump_intermediate_dir = '/home/OANet/author/raw_data/data_dump/'
        self.dump_collect_dir = self.make_dump()
        self.desc_name = 'sift-2000'  # prefix of desc filename
        self.vis_th = 0.35  # visibility threshold
        self.pair_num = 1000  # pair num

    def make_dump(self):
        dirs = '../data_dump'
        if not os.path.exists(dirs):
            os.makedirs(dirs)
        return dirs


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '5'

    conf = Config()

    test_seqs = ['te-brown1/', 'te-brown2/', 'te-brown3/', 'te-brown4/', 'te-brown5/', 'te-hotel1/',
                 'te-harvard1/', 'te-harvard2/', 'te-harvard3/', 'te-harvard4/',
                 'te-mit1/', 'te-mit2/', 'te-mit3/', 'te-mit4/', 'te-mit5/']
    sun3d_te = Dataset(conf.raw_data_path + 'sun3d_test/', conf.dump_intermediate_dir, conf.dump_collect_dir,
                       'sun3d-' + conf.desc_name + '-test.hdf5',
                       test_seqs, 'test', conf.desc_name, conf.vis_th, 1000, conf.raw_data_path + 'pairs/')
    sun3d_te.dump_data()

    with open('sun3d_train.txt','r') as ofp:
        train_seqs = ofp.read().split('\n')
    if len(train_seqs[-1]) == 0:
        del train_seqs[-1]
    train_seqs = [seq.replace('/','-')[:-1] for seq in train_seqs]
    print('train seq len '+str(len(train_seqs)))
    sun3d_tr_va = Dataset(conf.raw_data_path+'sun3d_train/', conf.dump_intermediate_dir, conf.dump_collect_dir,
                          'sun3d-'+conf.desc_name+'-val.hdf5',
                          train_seqs, 'val', conf.desc_name, conf.vis_th, 100, None)
    sun3d_tr_va.dump_data()

    sun3d_tr_tr = Dataset(conf.raw_data_path+'sun3d_train/', conf.dump_intermediate_dir, conf.dump_collect_dir,
                          'sun3d-'+conf.desc_name+'-train.hdf5',
                          train_seqs, 'train', conf.desc_name, conf.vis_th, 10000, None)
    sun3d_tr_tr.dump_data()