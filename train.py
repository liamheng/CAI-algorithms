import time
import os
import torch
import os.path as osp
import numpy as np
import logging
from options.train_options import TrainOptions
from loaders import create_dataset
from models import create_model
from util.visualizer import Visualizer
from util.logger import define_logger
from copy import deepcopy
from metrics import cal_confusion_for_one_img, cal_iou

class Runner():
    def __init__(self):
        self.gather_options()
        os.makedirs(osp.join(self.opt.checkpoints_dir, self.opt.name), exist_ok=True)
        self.root_name = self.opt.name
        self.opt.checkpoints_dir = osp.join(self.opt.checkpoints_dir, self.opt.name)
        self.run_rounds()

    def environment_settings(self):
        # create directory
        os.makedirs(osp.join(self.opt.checkpoints_dir, self.opt.name), exist_ok=True)
        torch.backends.cudnn.benchmark = True
        # attributes initializeation
        self.epoch_start_time = 0  
        self.curr_epoch = self.opt.epoch_start
        self.n_iters_per_epoch = 0
        self.n_iters = 0
        self.best = 0

    def gather_options(self):
        self.parser = TrainOptions()
        self.opt = self.parser.parse()
        

    def dataset_preparation(self):
        self.training_dataset = create_dataset(self.opt, self.logger)
        self.opt.total_batches = len(self.training_dataset.dataloader)
        self.logger.info("===================================================================")
        self.logger.info("number of training samples: %d, number of batches per epoch %d" % (len(self.training_dataset), len(self.training_dataset.dataloader)))
        self.logger.info("===================================================================")

        if self.opt.validation:
            validation_args = deepcopy(self.opt)
            if self.opt.validation_root != "None":
                validation_args.data_root = self.opt.validation_root
            validation_args.phase = 'validation'
            validation_args.preprocess = 'rescale'
            validation_args.drop_last = False
            validation_args.shuffle = False
            validation_args.max_dataset_size = 50
            validation_args.dataset = self.opt.validation_dataset
            validation_args.batch_size = 1
            self.validate_X = []
            self.validate_Y = []
            self.validation_dataset = create_dataset(validation_args, self.logger)
            cls_name_mapping = self.validation_dataset.get_classes()
            self.validate_legend = [cls_name_mapping[i] for i in range(self.opt.output_nc)] + ['mean']

    def model_preparation(self):
        self.model = create_model(self.opt, self.logger)
        # set logger 
        self.model.logger = self.logger
        self.model.initialize()
        self.model.print_infos()


    def visualizer_preparation(self):
        self.visualizer = Visualizer(self.opt)
        self.visualizer.logger = self.logger
    
    def prepare_logger(self):
        log_name = self.opt.phase
        log_dir = osp.join(self.opt.checkpoints_dir, self.opt.name)
        define_logger(log_dir, log_name)
        self.logger = logging.getLogger(log_name)
    
    def run_rounds(self):
        "run sequentially to get mean and std"
        for r in range(self.opt.rounds):
            self.opt.name = self.root_name + "_round%d" % r
            self.prepare_logger()
            self.logger.info("################################################Start training round: %d################################################" % r )
            now = time.time()
            self.parser.print_options(self.opt, now)
            self.environment_settings()
            self.dataset_preparation()
            self.model_preparation()
            self.visualizer_preparation()
            self.run()
            self.logger.info("###############################################Finish training round: %d################################################" % r )

    def run(self):
        """Core function to run the training process of the method"""
        import time
        for epoch in range(self.opt.epoch_start, self.opt.epoch_end + 1):
                        
            # update epoch information
            self.update_epoch_info('begining')

            # enumerate dataset to train
            for i, training_data in enumerate(self.training_dataset):
                # train model
                self.model.set_inputs(training_data)
                self.model.optimize()
                # update update_learning rate for networks in model(by iter)
                self.model.update_lr('iteration')

                # update iteration information
                self.update_iter_info()
                
                if self.n_iters % self.opt.display_freq == 0:
                    # visualize results
                    losses = self.model.get_losses()
                    losses['lr1000x'] = self.model.optimizers[0].param_groups[0]['lr']*1000 # append lr to visualize
                    self.visualizer.visualize(
                        epoch, 
                        self.n_iters_per_epoch,
                        self.n_iters,
                        losses,
                        self.model.get_visuals(),
                        )
                # save networks
                self.save_model("iteration")
            
            if self.opt.validation:
                self.eval_at_one_epoch()
            # save networks
            self.save_model("epoch")

            # update epoch information
            self.update_epoch_info('end')

            # update update_learning rate for networks in model(by epoch)
            self.model.update_lr('epoch')
            
        # save networks at the end of training
        self.save_model("final")
    def update_epoch_info(self, stage='begining'):
        if stage=='begining':
            self.epoch_start_time = time.time()
            self.n_iters_per_epoch = 0
            self.model.n_iters_per_epoch = 0
        
        elif stage=='end':
            self.curr_epoch += 1
            self.model.curr_epoch = self.curr_epoch
    
    def update_iter_info(self):
        self.n_iters_per_epoch += 1
        self.n_iters += 1
        self.model.n_iters_per_epoch = self.n_iters_per_epoch
        self.model.n_iters = self.n_iters

    def save_model(self, phase, value=0):
        """Method to save model at the end of epoch or at the end of each iteration"""
        
        # save by iterations
        if phase == "iteration" and self.opt.save_by == "iteration":
            if self.n_iters % self.opt.save_freq == 0:
                self.logger.info("saving the model at the end of iter %d" % (self.n_iters))
                self.model.save_networks("iter_%d" % (self.n_iters))

        # save by epochs
        if phase == "epoch" and self.opt.save_by == "epoch":
            if self.curr_epoch % self.opt.save_freq == 0:
                self.logger.info("saving the model at the end of epoch %d" % (self.curr_epoch))
                self.model.save_networks("epoch_%d" % (self.curr_epoch))
        
        # save at the end of training 
        if phase == 'final' and self.opt.save_by == "epoch":
            self.logger.info("saving the model at the end of training")
            self.model.save_networks("final")
        
        if phase == 'best':
            self.logger.info("saving the best model %.2f" % value)
            self.model.save_networks("best")


    def eval_at_one_epoch(self):
        # validate on each epoch
        self.model.eval()
        ious = []
        import time
        for i, validate_data in enumerate(self.validation_dataset):
            with torch.no_grad():
                self.model.set_inputs(validate_data)
                self.model.forward()
                self.model.visualization_preprocess()

                mask = self.model.mask.cpu().numpy().astype(np.uint32)
                gt = self.model.gt.cpu().numpy().astype(np.uint32)
                
                tp, fp, fn = cal_confusion_for_one_img(self.opt.output_nc, gt, mask)
                iou = cal_iou(tp, fp, fn)
                ious.append(iou)
        ious = np.stack(ious).mean(axis=0)
        mean = ious.mean()


        self.validate_X.append([self.curr_epoch for _ in range(self.opt.output_nc + 1)])
        self.validate_Y.append([*ious, mean])
        self.visualizer.vis.line(
            X = self.validate_X,
            Y = self.validate_Y,
            win = hash(self.opt.name) % 1000 + 999,
            opts={
                'title':'validation_' + self.opt.name,
                'xlabel':'epoch',
                'ylabel':'metric',
                'legend':self.validate_legend
            }
        )

        self.model.train()
        format_results = f"epoch {self.curr_epoch} mean ious: {mean:.2f}\t"
        self.logger.info(format_results)
        if mean > self.best:
            self.best = mean
            self.save_model("best", mean)


if __name__ == '__main__':
    runner = Runner()
