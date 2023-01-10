import os
import h5py
import pickle
from dump_match.sequence import Sequence
import numpy as np

class Dataset(object):

    def __init__(self, dataset_path, dump_intermediate_dir, dump_collect_dir, dump_file, seqs, mode, desc_name, vis_th, pair_num, pair_path):
        self.dataset_path = dataset_path

        self.dump_intermediate_dir = dump_intermediate_dir  # offline directory
        self.dump_collect_dir = dump_collect_dir # offline directory
        self.dump_file = os.path.join(dump_collect_dir, dump_file) # offline file

        self.seqs = seqs # test(4+15) or train(68+239) sequences
        self.mode = mode
        self.desc_name = desc_name
        self.vis_th = vis_th # visibility threshold
        self.pair_num = pair_num # test: 1000, train: 10000
        self.pair_path = pair_path # in the raw_data directory

    def collect(self):
        """
        Merge all sequences of the dataset (format hdf5),
        and place the merged file in ./data_dump, which is the actual data used.
        """
        data_type = ['xs', 'ys', 'Rs', 'ts', 'ratios', 'mutuals', 'cx1s', 'cy1s', 'cx2s', 'cy2s', 'f1s', 'f2s']
        pair_idx = 0
        with h5py.File(self.dump_file, 'w') as f:
            data = {}
            for tp in data_type:
                data[tp] = f.create_group(tp)
            for seq in self.seqs:
                print(seq)
                data_seq = {}
                for tp in data_type:
                    data_seq[tp] = pickle.load(open(
                        self.dump_intermediate_dir + '/' + seq + '/' + self.desc_name + '/' + self.mode + '/' + str(tp) + '.pkl',
                        'rb'))
                seq_len = len(data_seq['xs'])

                for i in range(seq_len):
                    for tp in data_type:
                        data_item = data_seq[tp][i]
                        if tp in ['cx1s', 'cy1s', 'cx2s', 'cy2s', 'f1s', 'f2s']:
                            data_item = np.asarray([data_item])
                        data_i = data[tp].create_dataset(str(pair_idx), data_item.shape, dtype=np.float32)
                        data_i[:] = data_item.astype(np.float32)
                    pair_idx = pair_idx + 1
                print('pair idx now ' + str(pair_idx))

    def dump_data(self):
        # make sure you have already saved the features
        for seq in self.seqs:
            # has been in existence, sun3d train dataset need to rematch
            pair_name = None if self.pair_path is None else self.pair_path + '/' + seq.rstrip("/") + '-te-' + str(self.pair_num) + '-pairs.pkl'
            seq_path = self.dataset_path + '/' + seq + '/' + self.mode
            dump_dir = self.dump_intermediate_dir + '/' + seq + '/' + self.desc_name + '/' + self.mode
            print(seq_path)

            dataset = Sequence(seq_path, dump_dir, self.desc_name, self.vis_th, self.pair_num, pair_name) # instantiate the sequence
            print('dump intermediate files.')
            # Transition storage, the format is h5(including idx_sort , ratio_test, mutual_nearest), saved in 'dump' directory for each pairs
            dataset.dump_intermediate()
            print('dump matches.')
            dataset.dump_datasets() # Dump the 12 kinds of data of all pairs of the sequence as pkl
        print('collect pkl.')
        self.collect()