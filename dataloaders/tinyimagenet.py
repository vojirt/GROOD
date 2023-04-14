import os
import numpy as np
from sklearn.model_selection import train_test_split
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor 
from torch.utils.data import Subset
import requests
import zipfile
from io import BytesIO


# from https://github.com/iCGY96/ARPL/blob/master/datasets/osr_dataloader.py 
class Tiny_ImageNet_Filter(ImageFolder):
    """Tiny_ImageNet Dataset.
    """
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


class TinyImageNet():
    def __init__(self, cfg, dataset_root, dataset_name, transform_train=ToTensor(), transform_test=ToTensor(), val_fraction=0.1):
        self.num_classes = cfg.MODEL.NUM_CLASSES

        self.known = cfg.DATASET.SELECTED_LABELS 
        self.unknown = list(set(list(range(0, 200))) - set(self.known))

        self.class_map = dict(zip(self.known, range(len(self.known))))
        self.inverse_class_map = dict(zip(range(len(self.known)), self.known))
        
        if not os.path.isdir(os.path.join(dataset_root, 'tiny-imagenet-200', 'train')):
            os.makedirs(dataset_root)
            durl = "http://cs231n.stanford.edu/tiny-imagenet-200.zip" 
            print(f"Downloading tinyimagenet dataset from {durl}.") 
            req = requests.get(durl)
            print('Downloading Completed')
            print(f"Extracting dataset to {dataset_root}.")
            zf = zipfile.ZipFile(BytesIO(req.content))
            zf.extractall(dataset_root)

        trainset = Tiny_ImageNet_Filter(os.path.join(dataset_root, 'tiny-imagenet-200', 'train'), transform=transform_train)
        trainset.__Filter__(known=self.known)

        valset = Tiny_ImageNet_Filter(os.path.join(dataset_root, 'tiny-imagenet-200', 'train'), transform=transform_test)
        valset.__Filter__(known=self.known)
        
        self.test_split = Tiny_ImageNet_Filter(os.path.join(dataset_root, 'tiny-imagenet-200', 'val'), transform=transform_test)

        cache_filename = os.path.join(cfg.EXPERIMENT.OUT_DIR, cfg.EXPERIMENT.NAME, "data", f"{dataset_name}_train_val_indexes.npz")
        os.makedirs(os.path.dirname(cache_filename), exist_ok=True)
        if os.path.isfile(cache_filename):
            cache = np.load(cache_filename)
            train_indices = cache["train_indices"]
            val_indices = cache["val_indices"]
            print (f"Loaded cached train/val split from: ", cache_filename)
        else:
            # generate indices: instead of the actual data we pass in integers instead
            targets = trainset.targets
            train_indices, val_indices, _, _ = train_test_split(
                range(len(trainset)),
                targets,
                stratify=targets,
                test_size=val_fraction,
            )
            np.savez(cache_filename, train_indices=train_indices, val_indices=val_indices)
            print (f"Cached train/val split to: ", cache_filename)

        # generate subset based on indices
        self.train_split = Subset(trainset, train_indices)
        self.val_split = Subset(valset, val_indices)

    def get_split(self, split):
        if split == "train":
            return self.train_split
        elif split == "val":
            return self.val_split
        elif split == "test": 
            return self.test_split
        else:
            raise NotImplementedError
