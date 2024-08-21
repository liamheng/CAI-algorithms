from utility.configuration import CONFIG
from utility.logger import LOGGER
from torch.utils.data import DataLoader
from utility.file_locating import obtain_class_from_file


registry = {
    'segmentation': 'dataset/segmentation_dataset.py'
}

TRAIN_CONFIG = CONFIG['DATALOADER']['TRAIN']
TRAIN_LOADER_NAME = TRAIN_CONFIG['NAME']
TRAIN_LOADER = obtain_class_from_file(registry[TRAIN_LOADER_NAME], TRAIN_LOADER_NAME)

VALID_CONFIG = CONFIG['DATALOADER']['VALID']
VALID_LOADER_NAME = VALID_CONFIG['NAME']
VALID_LOADER = obtain_class_from_file(registry[VALID_LOADER_NAME], VALID_LOADER_NAME)

TEST_CONFIG = CONFIG['DATALOADER']['TEST']
TEST_LOADER_NAME = TEST_CONFIG['NAME']
TEST_LOADER = obtain_class_from_file(registry[TEST_LOADER_NAME], TEST_LOADER_NAME)

TRAIN_LOADER = DataLoader(
    TRAIN_LOADER('TRAIN'), 
    batch_size=TRAIN_CONFIG['BATCH_SIZE'], 
    shuffle=TRAIN_CONFIG['SHUFFLE'],
    num_workers=TRAIN_CONFIG['NUM_WORKERS']
    )
VALID_LOADER = DataLoader(
    VALID_LOADER('VALID'), 
    batch_size=VALID_CONFIG['BATCH_SIZE'], 
    shuffle=VALID_CONFIG['SHUFFLE'],
    num_workers=VALID_CONFIG['NUM_WORKERS']
    )
TEST_LOADER = DataLoader(
    TEST_LOADER('TEST'), 
    batch_size=TEST_CONFIG['BATCH_SIZE'], 
    shuffle=TEST_CONFIG['SHUFFLE'],
    num_workers=TEST_CONFIG['NUM_WORKERS']
    )

LOGGER.info(f"Finish creating dataloaders")

