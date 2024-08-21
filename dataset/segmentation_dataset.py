from dataset.base_dataset import BaseDataset
from dataset.preprocess import get_preprocess
from utility.mallicious import lower_keys
from PIL import Image
from glob import glob
from os.path import join
from os import listdir

def get_preprocess_args(process_list):
    process_list = lower_keys(process_list)
    process_list = list(map(lambda x: (x['name'], x['args']), process_list))
    return process_list

class SegmentationDataset(BaseDataset):
    def __init__(self, mode):
        super().__init__(mode)
        
    def generate_infos(self):
        dataset_info=self.obtain_dataconfig()
        info = listdir(dataset_info['ROOT'])
        return info
    
    def generate_names(self):
        return self.obtain_dataconfig()['NAME']

    def generate_preprocess(self):
        args = get_preprocess_args(self.obtain_dataconfig()['PREPROCESS'])
        return args
    
    def __len__(self):
        lens = len(self.obtain_infos())
        return lens

    def __getitem__(self, idx):
        img = Image.open(join(self.obtain_dataconfig()['ROOT'], str(idx), 'image.png'))
        lbl = Image.open(join(self.obtain_dataconfig()['ROOT'], str(idx), 'label.png'))
        preprocess = get_preprocess(self.obtain_preprocess())
        img, lbl = preprocess((img, lbl))

        name = self.obtain_names()
        return {
            'image': img,
            'label': lbl.squeeze(),
            'name': name
        }

def segmentation(mode):
    return SegmentationDataset(mode)