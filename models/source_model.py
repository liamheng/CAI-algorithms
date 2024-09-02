# -*- coding: UTF-8 -*-
import torch
from models.base_model import BaseModel
from models.networks import define_network

class SourceModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, isTrain=True):
        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)
            
    def register_nets(self,):
        # define networks
        self.net_names = ['Task']
        self.netTask = define_network(self.opt.input_nc, self.opt.output_nc, self.opt.netTask, self.opt.gpu, self.opt.initialization, self.opt.init_gain, dict(pretrained=True))

    def register_optimizers(self):
        # define optimizers
        self.optimizer_Task = torch.optim.Adam(self.netTask.parameters(), lr=self.opt.lr)
    
    def register_losses(self):
        self.loss_names = ['seg']
        self.criterion_seg = torch.nn.CrossEntropyLoss(ignore_index=self.ignore_label) 
   
    def register_visuals(self):
        self.visual_names = ['rgb','color_gt', 'color_pred']
        self.visual_names += ['gt', 'mask'] if not self.isTrain else []
    
    def set_inputs(self, data):
        self.rgb = data['image'].to(self.device)
        self.gt = data['label'].to(self.device, dtype=torch.long)
    
    def optimize(self):
        self.optimizer_Task.zero_grad()
        self.forward()
        self.backward()
        self.optimizer_Task.step()

    def forward(self):
        self.pred = self.netTask(self.rgb)
        self.mask = self.get_hard_label(self.pred)

    def backward(self):
        self.loss_seg = self.criterion_seg(self.pred, self.gt)
        self.loss_seg.backward()

    def visualization_preprocess(self):
        with torch.no_grad():
            self.color_gt = self.expand(self.gt)
            self.color_pred = self.expand(self.mask)