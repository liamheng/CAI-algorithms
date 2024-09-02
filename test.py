import os
import os.path as osp
import csv
import torch
import logging
import time
import numpy as np
from options.test_options import TestOptions 
from loaders import create_dataset
from models import create_model
from PIL import Image
from metrics import SegMetrics
from util.HTML import HTML
from util.logger import define_logger

class Runner():
    def __init__(self):
        self.gather_options()
        self.root_name = self.opt.name
        self.opt.checkpoints_dir = osp.join(self.opt.checkpoints_dir, self.opt.name)

        self.run_rounds()

    def run_rounds(self):
        "run sequentially to get mean and std"
        for r in range(self.opt.rounds):
            print("################################################Start testing round: %d################################################" % r )
            self.opt.name = self.root_name + "_round%d" % r
            self.save_name = self.opt.name
            self.save_name += ('_' + self.opt.save_suffix if self.opt.save_suffix != "" else '')
            self.save_name += '_' + self.opt.load_suffix + ('_eval' if self.opt.eval else '')
            self.prepare_logger()
            now = time.time()
            self.parser.print_options(self.opt, now)
            self.environment_settings()
            self.dataset_preparation()
            self.model_preparation()
            self.run()
            print("###############################################Finish testing round: %d################################################" % r )

    def prepare_logger(self):
        log_name = self.opt.phase
        log_dir = osp.join(self.opt.results_dir, self.save_name)
        define_logger(log_dir, log_name)
        self.logger = logging.getLogger(log_name)

    def environment_settings(self):
        # create directory
        self.save_dir = osp.join(self.opt.results_dir, self.save_name)
        self.names = []
        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(osp.join(self.save_dir, 'image'), exist_ok=True)


    def gather_options(self):
        self.parser = TestOptions()
        self.opt = self.parser.parse()
    
    def dataset_preparation(self):
        self.test_dataset = create_dataset(self.opt, self.logger)
        # store class information
        classes = self.test_dataset.get_classes()
        classes = [[key, value] for key, value in classes.items()]
        with open(osp.join(self.save_dir, 'classes.csv'), 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['id', 'name'])
            writer.writerows(classes)
        print("===================================================================")
        print("number of test samples: %d" % len(self.test_dataset))
        print("===================================================================")
    
    def model_preparation(self):
        self.model = create_model(self.opt, self.logger)
        self.model.initialize()
        self.model.print_infos()
    
    def run(self):
        with torch.no_grad(): # torch no grad is needed or the computation graph will be reserved and takes up a lot of GPU memory
            for i, test_data in enumerate(self.test_dataset):
                # infer output through models
                self.model.set_inputs(test_data)
                self.model.forward()
                self.model.visualization_preprocess()

                # visualize results and append name information for further metric evalutation
                for batch_index, name in enumerate(test_data["name"]):
                    self.names.append(name+'\n') 

                    visuals = self.model.get_visuals(batch_index=batch_index)

                    self.save_visuals(visuals, name)
                if i % 5 ==0 and i != 0:
                    print("Finished inferencing batch %d" % i)
        # generate name.txt
        with open(osp.join(self.save_dir, 'names.txt'), 'w') as f:
            f.writelines(self.names)
        print("Finished inferencing all data")

        # create metrics evaluator
        self.seg_metrics = SegMetrics(self.save_dir)
        print("Calculating metrics......")
        self.seg_metrics.run()

        # create html
        html = HTML(self.model.visual_names, self.save_dir, osp.join(self.save_dir, 'index.html'))
        html.write2html()
    
    def save_visuals(self, visuals, idx):
        for name, image in visuals.items():
            # convert to PIL Image
            if image.ndim == 3:
                image = image.transpose((1, 2, 0)).squeeze()   
            image = image.astype(np.uint8)         
            image = Image.fromarray(image)

            # save to results dir
            path = osp.join(self.save_dir, 'image', "{}_{}.png".format(idx, name))
            image.save(path)
            

if __name__ == '__main__':
    runner = Runner()

        
    

