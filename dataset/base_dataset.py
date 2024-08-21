# from 
from abc import ABC, abstractmethod
from torch.utils.data import Dataset
from utility.configuration import CONFIG

class BaseDataset(Dataset, ABC):
    def __init__(self, mode):
        super().__init__()
        self._data_config = CONFIG['DATALOADER'][mode]['DATASET']
        self._infos = self.generate_infos()
        self._preprocess = self.generate_preprocess()
        self._names = self.generate_names()

    @abstractmethod
    def generate_preprocess(self):
        pass

    @abstractmethod
    def generate_names(self):
        pass

    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def __getitem__(self, idx):
        pass

    @abstractmethod
    def generate_infos(self):
        pass

    def obtain_preprocess(self):
        return self._preprocess

    def obtain_names(self):
        return self._names

    def obtain_infos(self):
        return self._infos

    def obtain_dataconfig(self):
        return self._data_config