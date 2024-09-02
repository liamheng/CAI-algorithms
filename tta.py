import time
import os
import torch
import os.path as osp
import numpy as np
import logging
from functools import reduce
from options.train_options import TrainOptions
from options.test_options import TestOptions
from loaders import create_dataset
from models import create_model
from util.logger import define_logger
from metrics import cal_confusion_for_one_img, summarise_statistics, cal_iou, cal_dice
from PIL import Image


class Runner():
    def __init__(self):
        self.gather_options()
        self.environment_settings()
        define_logger(self.save_dir, 'tta')
        self.logger = logging.getLogger('tta')
        self.train_parser.print_options(self.opt, time.time())
        self.dataset_preparation()
        self.model_preparation()
        self.run()

    def environment_settings(self):
        # create directory
        os.makedirs(osp.join(self.opt.checkpoints_dir, self.opt.name), exist_ok=True)
        self.logger = logging.getLogger('tta')

        self.opt.checkpoints_dir = osp.join(self.opt.checkpoints_dir, self.opt.name)
        os.makedirs(osp.join(self.opt.checkpoints_dir, self.opt.name), exist_ok=True)
        torch.backends.cudnn.benchmark = True
        self.save_name = self.opt.name + ('_' + self.opt.save_suffix if self.opt.save_suffix != "" else '')
        self.save_dir = osp.join(self.opt.results_dir, self.save_name)
        os.makedirs(osp.join(self.save_dir, 'image'), exist_ok=True)

        # attributes initializeation

    def gather_options(self):
        self.train_parser = TrainOptions()
        train_opt = self.train_parser.parse()
        self.opt = train_opt

    def dataset_preparation(self):
        self.opt.phase = 'test'
        self.dataset = create_dataset(self.opt, self.logger)
        self.opt.total_batches = len(self.dataset.dataloader)
        self.logger.info("===================================================================")
        self.logger.info("number of tta samples: %d, number of batches per epoch %d" % (len(self.dataset), len(self.dataset.dataloader)))
        self.logger.info("===================================================================")

    def model_preparation(self):
        self.opt.phase = 'training'
        self.model = create_model(self.opt, self.logger)
        # set logger 
        self.model.initialize()
        self.model.print_infos()

    def save_visuals(self, visuals, idx):
        for name, image in visuals.items():
            if name != 'rgb' and name != 'color_pred':
                continue
            # convert to PIL Image
            if image.ndim == 3:
                image = image.transpose((1, 2, 0)).squeeze()   
            image = image.astype(np.uint8)         
            image = Image.fromarray(image)

            # save to results dir
            path = osp.join(self.save_dir, 'image', "{}_{}.png".format(idx, name))
            image.save(path)
            
    def run(self):
        """Core function to run the training process of the method"""
        n_iters = 0
        dices = []
        ious = []
        
        for idx, data in enumerate(self.dataset):
            self.model.train()
            for _ in range(self.opt.tta_steps):
                self.model.set_inputs(data)
                self.model.optimize()
                n_iters += 1
                self.model.n_iters = n_iters
            visuals = self.model.get_visuals(batch_index=0)
            self.save_visuals(visuals, idx)
            self.model.eval()
            with torch.no_grad():
                self.model.forward()
            # self.model.eval()
            self.model.visualization_preprocess()

            mask = self.model.mask.cpu().numpy().astype(np.uint32)
            gt = self.model.gt.cpu().numpy().astype(np.uint32)
                
            tp, fp, fn = cal_confusion_for_one_img(self.opt.output_nc, gt, mask)
            iou = cal_iou(tp, fp, fn)
            dice = cal_dice(tp, fp, fn)
            ious.append(iou)
            dices.append(dice)
            self.logger.info(f"{data['name']}: iou{np.mean(iou)}: {iou}, dice{np.mean(dice)}: {dice}" )

            
        ious = np.mean(ious, axis=0)
        dices = np.mean(dices, axis=0)
        anatomy_ious = np.mean(ious[:4], keepdims=True)
        anatomy_dics = np.mean(dices[:4], keepdims=True)
        tools_ious = np.mean(ious[4:], keepdims=True)
        tools_dices = np.mean(dices[4:], keepdims=True)
        overall_ious = np.mean(ious, keepdims=True)
        overall_dices = np.mean(dices, keepdims=True)
        ious = np.concatenate([ious, anatomy_ious, tools_ious, overall_ious], axis=0)
        dices = np.concatenate([dices, anatomy_dics, tools_dices, overall_dices], axis=0)
        res_ious = str(reduce(lambda x, y: x + "%.2f\t"%y, ious, ""))
        res_dices = str(reduce(lambda x, y: x + "%.2f\t"%y, dices, ""))
        self.logger.info(f"ious: {res_ious}, dices: {res_dices}")

if __name__ == '__main__':
    runner = Runner()
