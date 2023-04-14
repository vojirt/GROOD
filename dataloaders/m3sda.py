import os
from torchvision import datasets
import torch



class M3SDA_Filter(datasets.ImageFolder):
    def __Filter__(self, known):
        datas, targets = self.imgs, self.targets
        new_datas, new_targets = [], []
        for i in range(len(datas)):
            if datas[i][1] in known:
                new_item = (datas[i][0], known.index(datas[i][1]))
                new_datas.append(new_item)
                # new_targets.append(targets[i])
                new_targets.append(known.index(targets[i]))
        datas, targets = new_datas, new_targets
        self.samples, self.imgs, self.targets = datas, datas, targets


class M3SDA():
    def __init__(self, cfg, dataset_root, transform_train=None, transform_test=None):
        self.dataset_root = dataset_root
        self.val_fraction = cfg.DATASET.VAL_FRACTION
        self.rng_seed = cfg.SYSTEM.RNG_SEED
        self.transform_train = transform_train
        self.transform_test = transform_test

        self.known = cfg.DATASET.SELECTED_LABELS 
        self.unknown = list(set(list(range(0, 345))) - set(self.known))

    def get_dataset_split(self, dataset, split):
        if dataset in ["clipart", "infograph", "painting", "quickdraw", "real", "sketch"]:
            if split == "train":
                dataset = M3SDA_Filter(os.path.join(self.dataset_root, dataset, "train"), transform=self.transform_train)
                dataset.__Filter__(known=self.known)
                val_data_size = int(float(len(dataset)) * self.val_fraction)
                train_data_size = len(dataset) - val_data_size
                return torch.utils.data.random_split(dataset, [train_data_size, val_data_size], generator=torch.Generator().manual_seed(self.rng_seed))[0]
            elif split == "val":
                dataset = M3SDA_Filter(os.path.join(self.dataset_root, dataset, "train"), transform=self.transform_test)
                dataset.__Filter__(known=self.known)
                val_data_size = int(float(len(dataset)) * self.val_fraction)
                train_data_size = len(dataset) - val_data_size
                return torch.utils.data.random_split(dataset, [train_data_size, val_data_size], generator=torch.Generator().manual_seed(self.rng_seed))[1]
            elif split == "test": 
                return datasets.ImageFolder(os.path.join(self.dataset_root, dataset, "test"), transform=self.transform_test)
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError

