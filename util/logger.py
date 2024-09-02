import os
import os.path as osp
import logging

def define_logger(log_dir, log_name):
    os.makedirs(log_dir, exist_ok=True)
    log_path = osp.join(log_dir, log_name+'.log')
    logging.basicConfig(
        format='[%(asctime)s] [%(filename)s: %(lineno)4d]: %(message)s',
        datefmt="%y/%m/%d %H:%M:%S",
        level=logging.INFO,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_path)
        ]
    )