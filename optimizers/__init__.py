from models import MODELS
from utility.logger import LOGGER
from utility.configuration import CONFIG
from utility.file_locating import obtain_class_from_file
from utility.mallicious import lower_keys

registry = {
    'adam': 'optimizers/preset.py',
    'sgd': 'optimizers/preset.py',
    'adadelta': 'optimizers/preset.py',
    'adagrad': 'optimizers/preset.py',
    'adamw': 'optimizers/preset.py',
    'adamax': 'optimizers/preset.py',
    'asgd': 'optimizers/preset.py',
}

def obtain_optimizer(config):
    name = config["NAME"]
    args = lower_keys(config['ARGS'])
    params = list(map(lambda x: {'params': MODELS[x['MODEL']].parameters(), 'lr': x['LR']}, config['PARAMS']))
    return obtain_class_from_file(registry[name], name)(params, **args)

OPTIMS_CONFIG = CONFIG['TRAIN_HYPERS']['OPTIMIZER']
OPTIMS = {c['NAME']: obtain_optimizer(c) for c in OPTIMS_CONFIG}
LOGGER.info(f"Optimizers loaded: {OPTIMS.keys()}")