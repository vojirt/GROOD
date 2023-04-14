import os
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
import torch
from torchvision import datasets
from torchvision.transforms import ToTensor 
from PIL import Image


class TVDatasetWrapper():
    def __init__(self, cfg, dataset, dataset_name, use_split_keyword=True,
            transform_train=ToTensor(), transform_test=ToTensor(), 
            root_dir="./_data/", other_train_split_name=None):
        if other_train_split_name is not None:
            train_split_name = other_train_split_name
        else:
            train_split_name = "train"
        selected_labels = cfg.DATASET.SELECTED_LABELS
        if not use_split_keyword:
            train = dataset(root=root_dir, train=True, download=True, transform=transform_train)
            val = dataset(root=root_dir, train=True, download=True, transform=transform_test)
            self.test_split = dataset(root=root_dir, train=False, download=True, transform=transform_test)
        else:
            train = dataset(root=root_dir, split=train_split_name, download=True, transform=transform_train)
            val = dataset(root=root_dir, split=train_split_name, download=True, transform=transform_test)
            self.test_split = dataset(root=root_dir, split="test", download=True, transform=transform_test)

        self.class_map = dict(zip(selected_labels, range(len(selected_labels))))
        self.inverse_class_map = dict(zip(range(len(selected_labels)), selected_labels))

        # keep only train data with selected labels
        if hasattr(train, "targets"):
            print(f"Train dataset size: {len(train.targets)}")
            indices = [idx for idx, target in enumerate(train.targets) if target in selected_labels]
            print(f"Number of selected data samples: {len(indices)}")

            train.data = train.data[indices]
            train.targets = np.array(train.targets)[indices]
            train_targets = np.zeros_like(train.targets)
            for c in selected_labels:
                train_targets[train.targets == c] = self.class_map[c]
            train.targets = train_targets.tolist()

            val.data = val.data[indices]
            val.targets = np.array(val.targets)[indices]
            val_targets = np.zeros_like(val.targets)
            for c in selected_labels:
                val_targets[val.targets == c] = self.class_map[c]
            val.targets = val_targets.tolist()
        else:
            indices = [idx for idx, target in enumerate(train.labels) if target in selected_labels]

            train.data = train.data[indices]
            train.labels = torch.tensor(train.labels)[indices]
            train_targets = np.zeros_like(train.labels)
            for c in selected_labels:
                train_targets[train.labels == c] = self.class_map[c]
            train.labels = torch.from_numpy(train_targets)

            val.data = val.data[indices]
            val.labels = torch.tensor(val.labels)[indices]
            val_targets = np.zeros_like(val.labels)
            for c in selected_labels:
                val_targets[val.labels == c] = self.class_map[c]
            val.labels = torch.from_numpy(val_targets)

        cache_filename = os.path.join(cfg.EXPERIMENT.OUT_DIR, cfg.EXPERIMENT.NAME, "data", f"{dataset_name}_train_val_indexes.npz")
        os.makedirs(os.path.dirname(cache_filename), exist_ok=True)
        if os.path.isfile(cache_filename):
            cache = np.load(cache_filename)
            train_indices = cache["train_indices"]
            val_indices = cache["val_indices"]
            print (f"Loaded cached train/val split from: ", cache_filename)
        else:
            # generate indices: instead of the actual data we pass in integers instead
            targets = train.targets if hasattr(train, "targets") else train.labels
            if cfg.DATASET.VAL_FRACTION > 0.0:
                train_indices, val_indices, _, _ = train_test_split(
                    range(len(train)),
                    targets,
                    stratify=targets,
                    test_size=cfg.DATASET.VAL_FRACTION,
                )
            else:
                train_indices = torch.arange(len(targets), dtype=torch.long)
                val_indices = []

            np.savez(cache_filename, train_indices=train_indices, val_indices=val_indices)
            print (f"Cached train/val split to: ", cache_filename)

        print(f"train_indices size: {len(train_indices)}")
        print(f"val_indices size: {len(val_indices)}")
        # generate subset based on indices
        self.train_split = Subset(train, train_indices)
        self.val_split = Subset(val, val_indices)

    def get_split(self, split):
        if split == "train":
            return self.train_split
        elif split == "val":
            return self.val_split
        elif split == "test": 
            return self.test_split
        else:
            raise NotImplementedError

class MNISTRGB(datasets.MNIST):
    """MNIST Dataset.
    """
    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])
        img = Image.fromarray(img.numpy(), mode='L')
        img = img.convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

class KMNISTRGB(datasets.KMNIST):
    """KMNIST Dataset.
    """
    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])
        img = Image.fromarray(img.numpy(), mode='L')
        img = img.convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


def test_dataloader():
    from matplotlib import pyplot as plt
    from einops import rearrange
    from yacs.config import CfgNode as CN
    from dataloaders import make_data_loader


    cfg = CN()
    cfg.DATASET = CN()
    cfg.DATASET.SELECTED_LABELS = [0, 1, 2, 3, 4]
    cfg.DATASET.AUTO_AUGMENT = False
    cfg.DATASET.BASIC_AUGMENT = True
    cfg.INPUT = CN()
    cfg.INPUT.BATCH_SIZE=2
    cfg.INPUT.IMG_SZ=32
    cfg.MODEL = CN()
    cfg.MODEL.NUM_CLASSES = len(cfg.DATASET.SELECTED_LABELS)
    cfg.EXPERIMENT= CN()
    cfg.EXPERIMENT.OUT_DIR = "/home/vojirtom/code/grood/_out/tmp/"
    cfg.EXPERIMENT.NAME = "torchvision_dataset_test"

    dataset = "cifar10"
    cfg.DATASET.TRAIN = dataset
    cfg.DATASET.VAL = dataset
    cfg.DATASET.TEST = dataset

    train_loader, val_loader, test_loader, inverse_class_map = make_data_loader(cfg)
   
    for (x, y) in train_loader:
        print (x.size(), y.size())
        print (x.min(), x.max())
        fig, (ax1, ax2) = plt.subplots(1, 2)
        ax1.imshow(rearrange(x[0, ...], "c h w -> h w c").squeeze())
        ax1.set_title(f"class {y[0]}")
        ax2.imshow(rearrange(x[1, ...], "c h w -> h w c").squeeze())
        ax2.set_title(f"class {y[1]}")
        plt.show()


if __name__ == "__main__":
    test_dataloader()





