import importlib
import torch
from torch.utils.data import DataLoader
from loaders.base_dataset import BaseDataset

def find_dataset(dataset_name):
    
    # import py module of dataset
    dataset_filename = "loaders.%s_dataset" % dataset_name
    dataset_lib = importlib.import_module(dataset_filename)

    dataset = None

    # get dataset
    dataset_classname = dataset_name.replace('_', '') + 'dataset'
    for name, cls in dataset_lib.__dict__.items():
        if name.lower() == dataset_classname.lower() and issubclass(cls, BaseDataset):
            dataset = cls
    if dataset is None:
        print("In %s.py, there should be a subclass of BaseDataset with class name that matches %s in lowercase." % (dataset_filename, dataset_classname))
        exit(0)
    
    return dataset

def create_dataset(opt, logger):
    loader = Loader(opt)
    logger.info("[Create Dataset] dataset %s was created" % type(loader.dataset).__name__)
    return loader

class Loader():

    def __init__(self, opt):
        self.opt = opt

        # create dataset
        dataset = find_dataset(opt.dataset)
        self.dataset = dataset(opt)
        self.device = torch.device('cuda:{}'.format(opt.gpu)) if opt.gpu != -1 else torch.device('cpu')
        # create multithread loader
        self.dataloader = DataLoader(
            dataset=self.dataset,
            batch_size=opt.batch_size,
            shuffle=opt.shuffle,
            num_workers=int(opt.num_threads),
            drop_last=opt.drop_last
        )
    
    def __len__(self,):
        return min(len(self.dataset), self.opt.max_dataset_size)
    
    def __iter__(self):
        for i, data in enumerate(self.dataloader):
            if i * self.opt.batch_size >= self.opt.max_dataset_size:
                break
            yield data
    
    def get_classes(self):
        return self.dataset.get_classes()
    

        