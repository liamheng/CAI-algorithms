from abc import ABC, abstractmethod
from PIL import Image
from torchvision.transforms import InterpolationMode
from torchvision import transforms
import numpy as np
import torchvision.transforms.functional as F
import torch
import random

def sample_type_check(sample):
    assert isinstance(sample, tuple), "Sample must be a tuple"
    assert len(sample) == 2, "Sample must have 2 elements"
    img, lbl = sample
    assert isinstance(img, Image.Image), "Image of sample must be an Image"
    assert isinstance(lbl, Image.Image), "Label of sample must be a Image"

class Preprocess(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def __call__(self, sample):
        pass

class ToTensor(Preprocess):
    def __init__(self, **kwargs):
        super().__init__()

    def __call__(self, sample):
        img, lbl = sample
        img = F.to_tensor(img)
        lbl = F.to_tensor(lbl).long()
        return (img, lbl)

#? Whether sample-wise normalization would promote the performance?
class SampleWiseNormalize(Preprocess):
    def __init__(self, **kwargs):
        super().__init__()

    def __call__(self, sample):
        img, lbl = sample
        img = np.array(img)
        mean = img.mean(axis=(0, 1))
        std = img.std(axis=(0, 1))
        img = (img - mean) / std
        img = Image.fromarray(img)
        return (img, lbl)

class RandomHorizontalFlip(Preprocess):
    def __init__(self, prob=0.5, **kwargs):
        assert prob >= 0 and prob <= 1, "Probability must be in [0, 1]"
        self.prob = prob
        super().__init__()

    def __call__(self, sample):
        rand = random.random()
        img, lbl = sample
        if rand < self.prob:
            img = F.hflip(img)
            lbl = F.hflip(lbl)
            return (img, lbl)
        return (img, lbl)

class Resize(Preprocess):
    def __init__(self, size=[405, 720], **kwargs):
        assert len(size) == 2, "Size must be a tuple of 2 elements"
        self.size = size
        super().__init__()

    def __call__(self, sample):
        img, lbl = sample
        img = F.resize(img, self.size, interpolation=InterpolationMode.BILINEAR)
        lbl = F.resize(lbl, self.size, interpolation=InterpolationMode.NEAREST)
        return (img, lbl)

class RemapLabel(Preprocess):
    def __init__(self, mapping=None, **kwargs):
        assert isinstance(mapping, dict) or mapping is None, "Mapping must be a dictionary"
        self.mapping = mapping
        super().__init__()

    def __call__(self, sample):
        if self.mapping is None:
            return sample
        img, lbl = sample
        lbl = np.array(lbl).astype(np.uint8)
        vf = np.vectorize(lambda x: self.mapping[x])
        lbl = vf(lbl)
        lbl = Image.fromarray(lbl)
        return (img, lbl)

class PytorchNormalize(Preprocess):
    def __init__(self, mean, std):
        super().__init__()
        self.mean = mean
        self.std = std
    
    def __call__(self, sample):
        img, lbl = sample
        img = F.normalize(img, mean=self.mean, std=self.std)
        return (img, lbl)

preprocess_registery = {
    "to_tensor": ToTensor,
    "sample_wise_normalize": SampleWiseNormalize,
    "random_horizontal_flip": RandomHorizontalFlip,
    "resize": Resize,
    "remap_label": RemapLabel,
    "pytorch_normalize": PytorchNormalize
}

def get_preprocess(preprocess_list):
    preprocess = [] 
    for (p, args) in preprocess_list:
        preprocess.append(preprocess_registery[p](**args))
    preprocess = transforms.Compose(preprocess)
    return preprocess

