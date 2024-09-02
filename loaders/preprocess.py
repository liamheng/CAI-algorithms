import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
import random
from util.info_generation import generate_class_map
from PIL import Image, ImageFilter


class ToTensor:
    def __init__(self):
        pass

    def __call__(self, sample):
        img, label = sample
        img = F.to_tensor(img)
        label = torch.from_numpy(np.asarray(label)).to(dtype=torch.long)
        return img, label
    
class Rescale:
    def __init__(self, size):
        self.size = size[::-1] # (H, W)
    
    def __call__(self, sample):
        img, label = sample
        img = F.resize(img, self.size, interpolation=transforms.InterpolationMode.BILINEAR)
        label = F.resize(label, self.size, interpolation=transforms.InterpolationMode.NEAREST)
        return img, label
    
class Normalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
    
    def __call__(self, sample):
        img, label = sample
        img = F.normalize(img, self.mean, self.std)
        return img, label

class RandomHorizontalFlip:
    def __init__(self, p):
        self.p = p
    
    def __call__(self, sample):
        img, label = sample
        if random.random() > self.p:
            img = F.hflip(img)
            label = F.hflip(label)
        return img, label

class RandomVerticalFlip:
    def __init__(self, p):
        self.p = p
    
    def __call__(self, sample):
        img, label = sample
        if random.random() > self.p:
            img = F.vflip(img)
            label = F.vflip(label)
        return img, label

class ReMap:
    def __init__(self, dataroot, mapping_file):
        mapping = generate_class_map(dataroot, mapping_file)
        self.vf = np.vectorize(lambda e: mapping[e]["newid"])
    
    def __call__(self, sample):
        img, label = sample
        label = np.asarray(label, dtype=np.uint8)
        label = self.vf(label)
        return img, label

class RandomRotate:
    def __init__(self, p, r_range=10, ignore_label=-1):
        self.p = p
        self.r_range = r_range
        self.ignore_label = ignore_label

    def __call__(self, sample):
        img, label = sample
        if random.random() > self.p:
            degree = - self.r_range + random.random() * 2 * self.r_range
            img = F.rotate(img, degree, interpolation=transforms.InterpolationMode.BILINEAR)
            label = F.rotate(label, degree, interpolation=transforms.InterpolationMode.NEAREST, fill=self.ignore_label)
        return img, label


def get_transforms(opt):
    transform_list = []
    if 'rescale' in opt.preprocess:
        transform_list.append(Rescale(opt.load_size))
    if 'flip' in opt.preprocess:
        transform_list.append(RandomHorizontalFlip(0.5))
        transform_list.append(RandomVerticalFlip(0.5))
    if 'rotate' in opt.preprocess:
        transform_list.append(RandomRotate(0.5, ignore_label=opt.ignore_label))
    # default transforms
    transform_list += [
        ReMap(opt.data_root, opt.mapping_file_name),
        ToTensor(),
        Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ]
    return transforms.Compose(transforms=transform_list)

     