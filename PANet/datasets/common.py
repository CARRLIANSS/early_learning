import random
from torch.utils.data import Dataset

class BaseDataset(Dataset):

    def __init__(self, base_dir):
        self._base_dir = base_dir
        self.aux_attrib = {}
        self.aux_attrib_args = {}
        self.ids = []

    def add_attrib(self, key, func, func_args):
        if key in self.aux_attrib:
            raise KeyError("Attribute '{0}' already exists, please use 'set_attrib'.".format(key))
        else:
            self.set_attrib(key, func, func_args)

    def set_attrib(self, key, func, func_args):
        self.aux_attrib[key] = func
        self.aux_attrib_args[key] = func_args

    def del_attrib(self, key):
        self.aux_attrib.pop(key)
        self.aux_attrib_args.pop(key)

    def subsets(self, sub_ids, sub_args_lst=None):
        indices = [[self.ids.index(id_) for id_ in ids] for ids in sub_ids]
        if sub_args_lst is not None:
            subsets = [Subset(dataset=self, indices=index, sub_attrib_args=args)
                       for index, args in zip(indices, sub_args_lst)]
        else:
            subsets = [Subset(dataset=self, indices=index) for index in indices]
        return subsets

    def __len__(self):
        pass

    def __getitem__(self, idx):
        pass


class Subset(Dataset):

    def __init__(self, dataset, indices, sub_attrib_args=None):
        self.dataset = dataset
        self.indices = indices
        self.sub_attrib_args = sub_attrib_args

    def __getitem__(self, idx):
        if self.sub_attrib_args is not None:
            for key in self.sub_attrib_args:
                self.dataset.aux_attrib_args[key].update(self.sub_attrib_args[key])
        return self.dataset[self.indices[idx]]

    def __len__(self):
        return len(self.indices)


class PairedDataset(Dataset):

    def __init__(self, datasets, n_elements, max_iters, same=True, pair_based_transforms=None):
        super().__init__()
        self.datasets = datasets
        self.n_datasets = len(self.datasets)
        self.n_data = [len(dataset) for dataset in self.datasets]
        self.n_elements = n_elements
        self.max_iters = max_iters
        self.pair_based_transforms = pair_based_transforms
        if same:
            if isinstance(self.n_elements, int):
                datasets_indices = [random.randrange(self.n_datasets)
                                    for _ in range(self.max_iters)]
                self.indices = [[(dataset_idx, data_idx)
                                 for data_idx in random.choices(range(self.n_data[dataset_idx]),
                                                                k=self.n_elements)]
                                for dataset_idx in datasets_indices]
            else:
                raise ValueError("When 'same=true', 'n_element' should be an integer.")
        else:
            if isinstance(self.n_elements, list):
                self.indices = [[(dataset_idx, data_idx)
                                 for i, dataset_idx in enumerate(
                                     random.sample(range(self.n_datasets), k=len(self.n_elements)))
                                 for data_idx in random.sample(range(self.n_data[dataset_idx]),
                                                               k=self.n_elements[i])]
                                for i_iter in range(self.max_iters)]
            elif self.n_elements > self.n_datasets:
                raise ValueError("When 'same=False', 'n_element' should be no more than n_datasets")
            else:
                self.indices = [[(dataset_idx, random.randrange(self.n_data[dataset_idx]))
                                 for dataset_idx in random.sample(range(self.n_datasets),
                                                                  k=n_elements)]
                                for i in range(max_iters)]

    def __len__(self):
        return self.max_iters

    def __getitem__(self, idx):
        sample = [self.datasets[dataset_idx][data_idx]
                  for dataset_idx, data_idx in self.indices[idx]]
        if self.pair_based_transforms is not None:
            for transform, args in self.pair_based_transforms:
                sample = transform(sample, **args)
        return sample