from .base_dataset import BaseDataset
from .preprocess import get_transforms
from util.info_generation import generate_info, generate_class_map
from PIL import Image
import os.path as osp
import torch 
class SingleDomainDataset(BaseDataset):
    
    def __init__(self, opt):
        """
        Initialization, generate sample information and load them into memory
        """
        BaseDataset.__init__(self, opt)
        self.isTrain = opt.phase == 'training'
        self.root = osp.join(opt.data_root, opt.phase)
        self.infos = generate_info(self.root) 
        self.transforms = get_transforms(opt)
        self.label_class_map = generate_class_map(opt.data_root, name=opt.mapping_file_name)
        
        # if len(self.infos) > self.opt.max_dataset_size:
        #     self.infos = self.infos[:self.opt.max_dataset_size]
        # if self.opt.max_dataset_size == 1:
        #     self.infos = self.infos + self.infos
            
    def __getitem__(self, index):
        # get infomation of both image and label
        info = self.infos[index]
        image_path = info['image_path']
        label_path = info['label_path']
        image_name = info['image_name'].replace(".png", "")
        assert osp.exists(image_path)
        assert osp.exists(label_path)
        
        # load image and label in memory
        label = Image.open(label_path)
        image = Image.open(image_path).convert('RGB')
        
        image, label = self.transforms((image,label))
        
        return {'image':image, 'label': label, 'name':image_name}

    def __len__(self):
        return len(self.infos)

    def get_classes(self):
        return {info["newid"]:info["newname"] for info in self.label_class_map.values()}