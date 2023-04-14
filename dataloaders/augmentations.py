from torchvision.transforms import ToTensor, AutoAugment, Compose, RandomCrop, RandomHorizontalFlip, Resize, CenterCrop, Normalize
from torchvision.transforms import InterpolationMode


class NoAugmentation():
    def train(self, cfg):
        augment = Compose([
            Resize((cfg.INPUT.IMG_SZ, cfg.INPUT.IMG_SZ)),
            ToTensor()
        ])

        return augment
    
    def test(self, cfg):
        augment = Compose([
            Resize((cfg.INPUT.IMG_SZ, cfg.INPUT.IMG_SZ)), 
            ToTensor()
        ])

        return augment


class BasicAugmentation():
    def train(self, cfg):
        augment = Compose([
            Resize((cfg.INPUT.IMG_SZ, cfg.INPUT.IMG_SZ)),  
            RandomCrop(cfg.INPUT.IMG_SZ, padding=4), 
            RandomHorizontalFlip(), 
            ToTensor()
        ])

        return augment
    
    def test(self, cfg):
        augment = Compose([
            Resize((cfg.INPUT.IMG_SZ, cfg.INPUT.IMG_SZ)), 
            ToTensor()
        ])

        return augment


class AutoAugmentation():
    def train(self, cfg):
        augment = Compose([
            Resize((cfg.INPUT.IMG_SZ, cfg.INPUT.IMG_SZ)), 
            AutoAugment(),
            ToTensor()
        ])

        return augment
    
    def test(self, cfg):
        augment = Compose([
            Resize((cfg.INPUT.IMG_SZ, cfg.INPUT.IMG_SZ)), 
            ToTensor()
        ])

        return augment


class CLIPAugmentation():
    def train(self, cfg):
        augment = Compose([
            Resize(size=cfg.INPUT.IMG_SZ, interpolation=InterpolationMode.BICUBIC, max_size=None, antialias=None),
            CenterCrop(size=(cfg.INPUT.IMG_SZ, cfg.INPUT.IMG_SZ)),
            ToTensor(),
            Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
        ])

        return augment
    
    def test(self, cfg):
        augment = Compose([
            Resize(size=cfg.INPUT.IMG_SZ, interpolation=InterpolationMode.BICUBIC, max_size=None, antialias=None),
            CenterCrop(size=(cfg.INPUT.IMG_SZ, cfg.INPUT.IMG_SZ)),
            ToTensor(),
            Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
        ])

        return augment


class ResNetImageNetAugmentation():
    def train(self, cfg):
        augment = Compose([
            Resize(224),
            CenterCrop(size=(cfg.INPUT.IMG_SZ, cfg.INPUT.IMG_SZ)),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        return augment
    
    def test(self, cfg):
        augment = Compose([
            Resize(224),
            CenterCrop(size=(cfg.INPUT.IMG_SZ, cfg.INPUT.IMG_SZ)),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        return augment


