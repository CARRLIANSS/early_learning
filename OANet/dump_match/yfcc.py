from dump_match.dataset import Dataset
import os

class Config(object):

    def __init__(self):
        self.raw_data_path = '/home/OANet/author/raw_data/'
        self.dump_intermediate_dir = '/home/OANet/author/raw_data/data_dump/'
        self.dump_collect_dir = self.make_dump()
        self.desc_name = 'sift-2000' # prefix of desc filename
        self.vis_th = 50 # visibility threshold
        self.pair_num = 1000 # pair num

    def make_dump(self):
        dirs = '../data_dump'
        if not os.path.exists(dirs):
            os.makedirs(dirs)
        return dirs


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '5'

    conf = Config()

    # dump yfcc test
    test_seqs = ['buckingham_palace', 'notre_dame_front_facade', 'reichstag', 'sacre_coeur']
    yfcc_te = Dataset(conf.raw_data_path + 'yfcc100m/', conf.dump_intermediate_dir, conf.dump_collect_dir,
                      'yfcc-' + conf.desc_name + '-test.hdf5',
                      test_seqs, 'test', conf.desc_name, conf.vis_th, conf.pair_num, conf.raw_data_path + 'pairs/')
    yfcc_te.dump_data()

    # dump yfcc training seqs
    with open('yfcc_train.txt','r') as ofp:
        train_seqs = ofp.read().split('\n')
    if len(train_seqs[-1]) == 0:
        del train_seqs[-1]
    print('train seq len '+str(len(train_seqs)))
    yfcc_tr_va = Dataset(conf.raw_data_path + 'yfcc100m/', conf.dump_intermediate_dir, conf.dump_collect_dir,
                         'yfcc-' + conf.desc_name + '-val.hdf5',
                         train_seqs, 'val', conf.desc_name, conf.vis_th, 100, None)
    yfcc_tr_va.dump_data()

    yfcc_tr_tr = Dataset(conf.raw_data_path + 'yfcc100m/', conf.dump_intermediate_dir, conf.dump_collect_dir,
                         'yfcc-' + conf.desc_name + '-train.hdf5',
                         train_seqs, 'train', conf.desc_name, conf.vis_th, 10000, None)
    yfcc_tr_tr.dump_data()