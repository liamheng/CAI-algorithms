from utility.logger import LOGGER
from utility.configuration import CONFIG
from utility.file_locating import obtain_class_from_file
from utility.mallicious import lower_keys

registry = {
    'cross_entropy' : "losses/preset.py",
    'l1' : "losses/preset.py",
    'bce' : "losses/preset.py",
}

def obtain_losses(config):
    name = config["NAME"]
    args = lower_keys(config['ARGS'])
    return obtain_class_from_file(registry[name], name)(**args)

LOSSES_CONFIG = CONFIG['TRAIN_HYPERS']['LOSS']
LOSSES = {c['NAME']: obtain_losses(c) for c in LOSSES_CONFIG}
LOGGER.info(f"Optimizers loaded: {LOSSES.keys()}")