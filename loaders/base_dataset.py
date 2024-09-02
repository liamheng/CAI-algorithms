import torch.utils.data as data
from abc import ABC, abstractmethod

class BaseDataset(data.Dataset, ABC):

    def __init__(self, opt):
        self.opt = opt
        self.data_root = opt.data_root
        self.isTrain = self.opt.phase == 'training'

    @staticmethod
    def modify_commandline_options(parser, isTrain=True):
        return parser

    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def __getitem__(self, index):
        pass
    
    @abstractmethod
    def get_classes(self):
        pass
