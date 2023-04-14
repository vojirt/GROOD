from torchvision import datasets


class SimpleTestImageFolder():
    def __init__(self, cfg, dataset_root, transform_train=None, transform_test=None):
        self.dataset_root = dataset_root
        self.transform_train = transform_train
        self.transform_test = transform_test

    def get_split(self, split):
        if split == "test":
            return datasets.ImageFolder(self.dataset_root, transform=self.transform_test)
        else:
            raise NotImplementedError

