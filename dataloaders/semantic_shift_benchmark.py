import os
import numpy as np
from torch.utils.data import Subset
from torchvision.transforms import ToTensor 
from torchvision.datasets import ImageFolder
from sklearn.model_selection import train_test_split


# from https://github.com/iCGY96/ARPL/blob/master/datasets/osr_dataloader.py 
class SemanticShiftBenchmark_Filter(ImageFolder):
    """Semantic Shift Benchmark.
    """
    def __Filter__(self, selected_classes):
        datas, targets = self.imgs, self.targets
        new_datas, new_targets = [], []
        for i in range(len(datas)):
            if datas[i][1] in selected_classes:
                new_item = (datas[i][0], selected_classes.index(datas[i][1]))
                new_datas.append(new_item)
                # new_targets.append(targets[i])
                new_targets.append(selected_classes.index(targets[i]))
        datas, targets = new_datas, new_targets
        self.samples, self.imgs, self.targets = datas, datas, targets


class SemanticShiftBenchmark():
    def __init__(self, cfg, dataset_root, dataset_name, transform_train=ToTensor(), transform_test=ToTensor()):
        val_fraction = cfg.DATASET.VAL_FRACTION

        selected_labels = cfg.DATASET.SELECTED_LABELS
        self.class_map = dict(zip(selected_labels, range(len(selected_labels))))
        self.inverse_class_map = dict(zip(range(len(selected_labels)), selected_labels))

        trainset = SemanticShiftBenchmark_Filter(os.path.join(dataset_root, "train"), transform=transform_train)
        trainset.__Filter__(selected_classes=selected_labels)

        valset = SemanticShiftBenchmark_Filter(os.path.join(dataset_root, "train"), transform=transform_test)
        valset.__Filter__(selected_classes=selected_labels)

        self.test_split = SemanticShiftBenchmark_Filter(os.path.join(dataset_root, "test"), transform=transform_test)

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

    def get_dataset_split(self, split):
        if split == "train":
            return self.train_split
        elif split == "val":
            return self.val_split
        elif split == "test": 
            return self.test_split
        else:
            raise NotImplementedError
