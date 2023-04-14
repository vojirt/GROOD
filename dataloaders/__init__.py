import numpy
import importlib
import random
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import Compose, Grayscale

from dataloaders.torchvision_dataset_wrapper import TVDatasetWrapper, MNISTRGB, KMNISTRGB
from dataloaders.m3sda import M3SDA
from dataloaders.tinyimagenet import TinyImageNet
from dataloaders.simple_test_imagefolder import SimpleTestImageFolder   
from dataloaders.semantic_shift_benchmark import SemanticShiftBenchmark


def make_datasets(cfg, **kwargs):
    # load the specified augmentation
    augment_module = importlib.import_module("dataloaders.augmentations")
    # aug_maker = getattr(augment_module, cfg.DATASET.AUGMENT)(**kwargs)
    aug_maker = getattr(augment_module, cfg.DATASET.AUGMENT)()
    augment = aug_maker.train(cfg)
    test_augment = aug_maker.test(cfg)

    if "mnist" in [cfg.DATASET.TRAIN, cfg.DATASET.VAL, cfg.DATASET.TEST]:
        if cfg.INPUT.RGB:
            mnist_wrapper = TVDatasetWrapper(cfg, MNISTRGB, "MNISTRGB", use_split_keyword=False, transform_train=augment, transform_test=test_augment)
        else:
            mnist_wrapper = TVDatasetWrapper(cfg, datasets.MNIST, "MNIST", use_split_keyword=False, transform_train=augment, transform_test=test_augment)
    if "kmnist" in [cfg.DATASET.TRAIN, cfg.DATASET.VAL, cfg.DATASET.TEST]:
        if cfg.INPUT.RGB:
            kmnist_wrapper = TVDatasetWrapper(cfg, KMNISTRGB, "KMNISTRGB", use_split_keyword=False, transform_train=augment, transform_test=test_augment)
        else:
            kmnist_wrapper = TVDatasetWrapper(cfg, datasets.KMNIST, "KMNIST", use_split_keyword=False, transform_train=augment, transform_test=test_augment)
    if "cifar10" in [cfg.DATASET.TRAIN, cfg.DATASET.VAL, cfg.DATASET.TEST]:
        if cfg.INPUT.RGB:
            cifar10_wrapper = TVDatasetWrapper(cfg, datasets.CIFAR10, "CIFAR10", use_split_keyword=False, transform_train=augment, transform_test=test_augment)
        else:
            augment = Compose([augment, Grayscale()])
            test_augment = Compose([test_augment, Grayscale()])
            cifar10_wrapper = TVDatasetWrapper(cfg, datasets.CIFAR10, "CIFAR10", use_split_keyword=False, transform_train=augment, transform_test=test_augment)
    if "cifar100" in [cfg.DATASET.TRAIN, cfg.DATASET.VAL, cfg.DATASET.TEST]:
        cifar100_wrapper = TVDatasetWrapper(cfg, datasets.CIFAR100, "CIFAR100", use_split_keyword=False, transform_train=augment, transform_test=test_augment)
    if "svhn" in [cfg.DATASET.TRAIN, cfg.DATASET.VAL, cfg.DATASET.TEST]:
        svhn_wrapper = TVDatasetWrapper(cfg, datasets.SVHN, "SVHN", transform_train=augment, transform_test=test_augment)
    if "tinyimagenet" in [cfg.DATASET.TRAIN, cfg.DATASET.VAL, cfg.DATASET.TEST]:
        tinyimagenet_wrapper = TinyImageNet(cfg, "./_data/tinyimagenet/", "tinyimagenet", transform_train=augment, transform_test=test_augment)
    if not set(["clipart", "infograph", "painting", "quickdraw", "real", "sketch"]).isdisjoint([cfg.DATASET.TRAIN, cfg.DATASET.VAL, cfg.DATASET.TEST]):
        m3sda_wrapper = M3SDA(cfg, "./_data/M3SDA/", transform_train=augment, transform_test=test_augment)
    if "texture" in [cfg.DATASET.TRAIN, cfg.DATASET.VAL, cfg.DATASET.TEST]:
        texture_wrapper = SimpleTestImageFolder(cfg, "./_data/OpenOOD/data/images_classic/texture", transform_train=augment, transform_test=test_augment)
    if "places365" in [cfg.DATASET.TRAIN, cfg.DATASET.VAL, cfg.DATASET.TEST]:
        places365_wrapper = SimpleTestImageFolder(cfg, "./_data/OpenOOD/data/images_classic/places365", transform_train=augment, transform_test=test_augment)
    if "inaturalist" in [cfg.DATASET.TRAIN, cfg.DATASET.VAL, cfg.DATASET.TEST]:
        inaturalist_wrapper = SimpleTestImageFolder(cfg, "./_data/OpenOOD/data/images_largescale/inaturalist", transform_train=augment, transform_test=test_augment)
    if "lsun" in [cfg.DATASET.TRAIN, cfg.DATASET.VAL, cfg.DATASET.TEST]:
        lsun_wrapper = SimpleTestImageFolder(cfg, "./_data/LSUN/lsun/export/", transform_train=augment, transform_test=test_augment)
    if "ssb_cub" in [cfg.DATASET.TRAIN, cfg.DATASET.VAL, cfg.DATASET.TEST]:
        ssb_cub_wrapper = SemanticShiftBenchmark(cfg, "./_data/SemanticShiftBenchmark/cub-200-2011/CUB_200_2011", "ssb_cub", transform_train=augment, transform_test=test_augment)
    if "ssb_scars" in [cfg.DATASET.TRAIN, cfg.DATASET.VAL, cfg.DATASET.TEST]:
        ssb_scars_wrapper = SemanticShiftBenchmark(cfg, "./_data/SemanticShiftBenchmark/StanfordCars", "ssb_scars", transform_train=augment, transform_test=test_augment)
    if "ssb_aircraft" in [cfg.DATASET.TRAIN, cfg.DATASET.VAL, cfg.DATASET.TEST]:
        ssb_aircraft_wrapper = SemanticShiftBenchmark(cfg, "./_data/SemanticShiftBenchmark/FGVC-Aircraft/fgvc-aircraft-2013b/data", "ssb_aircraft", transform_train=augment, transform_test=test_augment)

    if cfg.DATASET.TRAIN == "mnist":
        train_set = mnist_wrapper.get_split("train")
        inverse_class_map = mnist_wrapper.inverse_class_map
    elif cfg.DATASET.TRAIN == "kmnist":
        train_set = kmnist_wrapper.get_split("train")
        inverse_class_map = kmnist_wrapper.inverse_class_map
    elif cfg.DATASET.TRAIN == "cifar10":
        train_set = cifar10_wrapper.get_split("train")
        inverse_class_map = cifar10_wrapper.inverse_class_map
    elif cfg.DATASET.TRAIN == "cifar100":
        train_set = cifar100_wrapper.get_split("train")
        inverse_class_map = cifar100_wrapper.inverse_class_map
    elif cfg.DATASET.TRAIN == "svhn":
        train_set = svhn_wrapper.get_split("train")
        inverse_class_map = svhn_wrapper.inverse_class_map
    elif cfg.DATASET.TRAIN == "tinyimagenet":
        train_set = tinyimagenet_wrapper.get_split("train")
        inverse_class_map = tinyimagenet_wrapper.inverse_class_map
    elif cfg.DATASET.TRAIN in ["clipart", "infograph", "painting", "quickdraw", "real", "sketch"]:
        train_set = m3sda_wrapper.get_dataset_split(cfg.DATASET.TRAIN, "train")
        inverse_class_map = dict(zip(range(cfg.MODEL.NUM_CLASSES), range(cfg.MODEL.NUM_CLASSES)))
    elif cfg.DATASET.TRAIN in ["texture", "places365", "inaturalist", "lsun"]:
        raise NotImplementedError
    elif cfg.DATASET.TRAIN == 'ssb_cub':
        train_set = ssb_cub_wrapper.get_dataset_split("train")
        inverse_class_map = ssb_cub_wrapper.inverse_class_map
    elif cfg.DATASET.TRAIN == 'ssb_scars':
        train_set = ssb_scars_wrapper.get_dataset_split("train")
        inverse_class_map = ssb_scars_wrapper.inverse_class_map
    elif cfg.DATASET.TRAIN == 'ssb_aircraft':
        train_set = ssb_aircraft_wrapper.get_dataset_split("train")
        inverse_class_map = ssb_aircraft_wrapper.inverse_class_map
    else:
        raise NotImplementedError
 
    if cfg.DATASET.VAL == "mnist":
        val_set = mnist_wrapper.get_split("val")
    elif cfg.DATASET.VAL == "kmnist":
        val_set = kmnist_wrapper.get_split("val")
    elif cfg.DATASET.VAL == "cifar10":
        val_set = cifar10_wrapper.get_split("val")
    elif cfg.DATASET.VAL == "cifar100":
        val_set = cifar100_wrapper.get_split("val")
    elif cfg.DATASET.VAL == "svhn":
        val_set = svhn_wrapper.get_split("val")
    elif cfg.DATASET.VAL == "tinyimagenet":
        val_set = tinyimagenet_wrapper.get_split("val")
    elif cfg.DATASET.VAL in ["clipart", "infograph", "painting", "quickdraw", "real", "sketch"]:
        val_set = m3sda_wrapper.get_dataset_split(cfg.DATASET.VAL, "val")
    elif cfg.DATASET.VAL in ["texture", "places365", "inaturalist", "lsun"]:
        raise NotImplementedError
    elif cfg.DATASET.VAL == 'ssb_cub':
        val_set = ssb_cub_wrapper.get_dataset_split("val")
    elif cfg.DATASET.VAL == 'ssb_scars':
        val_set = ssb_scars_wrapper.get_dataset_split("val")
    elif cfg.DATASET.VAL == 'ssb_aircraft':
        val_set = ssb_aircraft_wrapper.get_dataset_split("val")
    else:
        raise NotImplementedError

    if cfg.DATASET.TEST == "mnist":
        test_set = mnist_wrapper.get_split("test")
    elif cfg.DATASET.TEST == "kmnist":
        test_set = kmnist_wrapper.get_split("test")
    elif cfg.DATASET.TEST == "cifar10":
        test_set = cifar10_wrapper.get_split("test")
    elif cfg.DATASET.TEST == "cifar100":
        test_set = cifar100_wrapper.get_split("test")
    elif cfg.DATASET.TEST == "svhn":
        test_set = svhn_wrapper.get_split("test")
    elif cfg.DATASET.TEST == "tinyimagenet":
        test_set = tinyimagenet_wrapper.get_split("test")
    elif cfg.DATASET.TEST in ["clipart", "infograph", "painting", "quickdraw", "real", "sketch"]:
        test_set = m3sda_wrapper.get_dataset_split(cfg.DATASET.TEST, "test")
    elif cfg.DATASET.TEST == "texture":
        test_set = texture_wrapper.get_split("test")
    elif cfg.DATASET.TEST == "places365":
        test_set = places365_wrapper.get_split("test")
    elif cfg.DATASET.TEST == "inaturalist":
        test_set = inaturalist_wrapper.get_split("test")
    elif cfg.DATASET.TEST == 'ssb_cub':
        test_set = ssb_cub_wrapper.get_dataset_split("test")
    elif cfg.DATASET.TEST == 'ssb_scars':
        test_set = ssb_scars_wrapper.get_dataset_split("test")
    elif cfg.DATASET.TEST == 'ssb_aircraft':
        test_set = ssb_aircraft_wrapper.get_dataset_split("test")
    elif cfg.DATASET.TEST == "lsun":
        test_set = lsun_wrapper.get_split("test")
    else:
        raise NotImplementedError

    return train_set, val_set, test_set, inverse_class_map


def make_data_loader(cfg, **kwargs):
    train_set, val_set, test_set, inverse_class_map = make_datasets(cfg, **kwargs)

    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        numpy.random.seed(worker_seed)
        random.seed(worker_seed)

    train_loader = DataLoader(train_set, batch_size=cfg.INPUT.BATCH_SIZE, drop_last=True, shuffle=True, worker_init_fn=seed_worker, **kwargs)
    val_loader = DataLoader(val_set, batch_size=cfg.INPUT.BATCH_SIZE, shuffle=False, worker_init_fn=seed_worker, **kwargs)
    test_loader = DataLoader(test_set, batch_size=cfg.INPUT.BATCH_SIZE, shuffle=False, worker_init_fn=seed_worker, **kwargs)

    return train_loader, val_loader, test_loader, inverse_class_map


def get_data_loader_num_classes(cfg, **kwargs):
    if cfg.DATASET.TRAIN in ["mnist", "kmnist", "cifar10", "svhn"]:
        train_num_classes = 10
    elif cfg.DATASET.TRAIN == 'cifar100':
        train_num_classes = 100 
    elif cfg.DATASET.TRAIN == 'tinyimagenet':
        train_num_classes = 200 
    elif cfg.DATASET.TRAIN in ['clipart', 'infograph', 'quickdraw', 'real', 'sketch']:
        train_num_classes = 345 
    elif cfg.DATASET.TRAIN in ['painting']:
        train_num_classes = 344 
    elif cfg.DATASET.TRAIN == 'ssb_cub':
        train_num_classes = 200 
    elif cfg.DATASET.TRAIN == 'ssb_scars':
        train_num_classes = 196 
    elif cfg.DATASET.TRAIN == 'ssb_aircraft':
        train_num_classes = 50 
    else:
        raise NotImplementedError

    if cfg.DATASET.VAL in ["mnist", "kmnist", "cifar10", "svhn"]:
        val_num_classes = 10
    elif cfg.DATASET.VAL == 'cifar100':
        val_num_classes = 100 
    elif cfg.DATASET.VAL == 'tinyimagenet':
        val_num_classes = 200 
    elif cfg.DATASET.VAL in ['clipart', 'infograph', 'quickdraw', 'real', 'sketch']:
        val_num_classes = 345 
    elif cfg.DATASET.VAL in ['painting']:
        val_num_classes = 344 
    elif cfg.DATASET.VAL == 'ssb_cub':
        val_num_classes = 200 
    elif cfg.DATASET.VAL == 'ssb_scars':
        train_num_classes = 196 
    elif cfg.DATASET.VAL == 'ssb_aircraft':
        train_num_classes = 50 
    else:
        raise NotImplementedError

    if cfg.DATASET.TEST in ["mnist", "kmnist", "cifar10", "svhn"]:
        test_num_classes = 10
    elif cfg.DATASET.TEST == 'cifar100':
        test_num_classes = 100 
    elif cfg.DATASET.TEST == 'tinyimagenet':
        test_num_classes = 200 
    elif cfg.DATASET.TEST in ['clipart', 'infograph', 'quickdraw', 'real', 'sketch']:
        test_num_classes = 345 
    elif cfg.DATASET.TEST in ['painting']:
        test_num_classes = 344 
    elif cfg.DATASET.TEST == 'ssb_cub':
        test_num_classes = 200 
    elif cfg.DATASET.TEST == 'ssb_aircraft':
        train_num_classes = 50 
    else:
        raise NotImplementedError

    return train_num_classes, val_num_classes, test_num_classes
