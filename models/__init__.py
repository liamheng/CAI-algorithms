from utility.configuration import CONFIG
from utility.logger import LOGGER
from utility.file_locating import obtain_class_from_file
from utility.mallicious import lower_keys
from torch import save, load
from os import makedirs
from os.path import join

SAVE_DIR = join(CONFIG['EXP_DIRS'], CONFIG['EXP_NAME'], 'checkpoints')
makedirs(SAVE_DIR, exist_ok=True)

registry = {
    'deeplabv3_resnet101': 'models/segmentation/deeplabv3.py',
    'deeplabv3_resnet50': 'models/segmentation/deeplabv3.py',
    'resnet50': 'models/backbone/resnet.py',
    'resnet101': 'models/backbone/resnet.py',
    'aspp': 'models/segmentation/deeplabv3.py'
}

def obtain_model(config):
    name = config["NAME"]
    args = lower_keys(config['ARGS'])
    device = f'cuda:{config['GPU']}'
    model = obtain_class_from_file(registry[name], name)(**args).to(device)
    return model

MODELS_CONFIG = CONFIG['MODELS']
MODELS = {c['NAME']: obtain_model(c) for c in MODELS_CONFIG}
MODEL_DEVICE = {m: next(v.parameters()).device for m, v in MODELS.items()}
LOGGER.info(f"Models loaded: {MODELS.keys()}")


def save_models(models, prefix):
    for name, model in models.items():
        save(model.state_dict(), join(SAVE_DIR, f'{prefix}_{name}.pth'))
    LOGGER.info(f"Models saved with prefix {prefix}")

def load_models(models, prefix):
    for name, model in models.items():
        sd = load(join(SAVE_DIR, f'{prefix}_{name}.pth'), map_location=next(model.parameters()).device)
        model.load_state_dict(sd)
    LOGGER.info(f"Models load with prefix {prefix}")



