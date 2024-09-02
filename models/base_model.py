import torch
import numpy as np
import os.path as osp
import torch.nn as nn
import logging
from .networks import get_scheduler
from abc import ABC
from PIL.ImageColor import getrgb

class BaseModel(ABC):

    def __init__(self, opt):
        """Base model for all implemented methods.
        Note that:
        1. optimizers should be named in the form of "optimizer_*"
        """

        # environment attributes
        self.opt = opt
        self.device = torch.device('cuda:{}'.format(opt.gpu)) if opt.gpu != -1 else torch.device('cpu')
        self.save_dir = osp.join(opt.checkpoints_dir, opt.name)
        self.palette = self.get_palette("./static/palette.txt")
        self.isTrain = opt.phase == 'training'
        self.logger = logging.getLogger(opt.name)
        
        # attributes for recording data
        self.loss_names = []
        self.net_names = []
        self.visual_names = []

        # auxilary training attributes
        if self.isTrain:
            self.ignore_label = opt.ignore_label
            self.optimizers = []
            self.schedulers = []
            self.curr_epoch = 0
            self.n_iters = 0
            self.n_iters_per_epoch = 0

    @staticmethod
    def modify_commandline_options(parser, isTrain=True):
        return parser
    
    def register_nets(self):
        raise NotImplementedError

    def register_optimizers(self):
        raise NotImplementedError

    def register_losses(self):
        raise NotImplementedError
    
    def register_visuals(self):
        raise NotImplementedError

    def register_scheduler(self):
        raise NotImplementedError
    
    def set_inputs(self, data):
        raise NotImplementedError
    
    def optimize(self):
        raise NotImplementedError

    def forward(self):
        raise NotImplementedError

    def visualization_preprocess(self):
        raise NotImplementedError

    def get_palette(self, path):
        """
        Function to get palette from 'palette.txt' file located in './static/'.
        Palette is used for colorize the prediction output from the model,
        """
        palette = {}
        with open(path, 'r') as f:
            for index, line in enumerate(f):
                line = line.rstrip("\n")
                line = list(getrgb(line))
                palette[index] = line
        return palette
    
    def initialize(self):
        # register nets and visuals:
        self.register_nets()
        self.register_visuals()

        if self.isTrain:
            # collect optimizers
            self.register_optimizers()
            self.optimizers = [getattr(self, name) for name in self.__dict__.keys() if 'optimizer_' in name]
            self.schedulers = [get_scheduler(optimizer, self.opt) for optimizer in self.optimizers]
            self.logger.info('[Init Learning Rate] learning rate %.7f ' % (self.optimizers[0].param_groups[0]['lr']))
            # collect losses
            self.register_losses()
        
        elif not self.isTrain and self.opt.eval:
            # collect nets and set them to eval mode
            for name in self.net_names:
                net = getattr(self, 'net' + name)
                net.eval()
            self.logger.info("[Switch to eval mode]")

        if not self.isTrain or self.opt.continue_train :
            suffix = self.opt.load_suffix
            self.load_networks(suffix)
    
    def print_infos(self):
        self.logger.info("=================== Network Infos ===================")
        for name in self.net_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                n_params = sum([param.numel() for param in net.parameters()])
                mem_param = sum([param.nelement() * param.element_size() for param in net.parameters()])
                mem_buf = sum([buf.nelement() * buf.element_size() for buf in net.buffers()])
                mem = mem_param + mem_buf
                self.logger.info('[Network %s] Total number of parameters : %.3f M, total memory network occupies : %.3f MB' % (name, n_params / 1e6, mem / 1e6))
        self.logger.info("=====================================================")
    
    def update_lr(self, update_position):
        for scheduler in self.schedulers:
            scheduler.step(self.curr_epoch, self.n_iters, update_position)
    
    def save_networks(self, suffix):
        for name in self.net_names:
            
            file_name = "%s_net_%s.pth" % (suffix, name)
            file_path = osp.join(self.save_dir, file_name)
            net = getattr(self, 'net' + name)

            if self.device != -1 and torch.cuda.is_available():
                torch.save(net.cpu().state_dict(), file_path)
                net.cuda(self.device)
            else:
                torch.save(net.state_dict(), file_path)

    def load_one_network(self, name, suffix, net=None):
        file_name = "%s_net_%s.pth" % (suffix, name)
        file_path = osp.join(self.save_dir, file_name)
        if net is None:
            net = getattr(self, 'net' + name)
        if isinstance(net, nn.DataParallel):
            net = net
        
        state_dict = torch.load(file_path, map_location=str(self.device))
        if hasattr(state_dict, '_metatdata'):
            del state_dict.metadata
        net.load_state_dict(state_dict)
            

        self.logger.info("[Load Networks] load network %s from %s" % (name, file_path))
        
    def load_networks(self, suffix):
        for name in self.net_names:
            self.load_one_network(name, suffix)

    def colorize_one_img(self, img):
        img = img.data
        img = img.cpu().numpy()[0]
        res = np.zeros((*img.shape, 3)) # in the shape of HxWx3
        for key in self.palette.keys():
            res[img==key] = self.palette[key]
        res = res.astype(np.uint8)
        res = res.transpose((2, 0, 1))
        return res
    
    def get_visuals(self, batch_index=0):
        self.visualization_preprocess()
        visuals = {}
        for name in self.visual_names:
            image = getattr(self, name)

            if self.device != -1 and torch.cuda.is_available():
                image = image.cpu()
            
            image = image[batch_index]
            
            if 'color' in name:
                image = self.colorize_one_img(image)
            elif 'gt' == name or 'mask' == name:
                image = image.numpy()
            elif 'heatmap' in name:
                image = np.flip(image.numpy(), axis=0)
            else:
                image = image.numpy()*255
            visuals[name] = image
        return visuals
    
    def get_losses(self, ):
        losses = {}
        for name in self.loss_names:
            loss = getattr(self, 'loss_{}'.format(name))
            losses[name] = loss.detach().cpu()
        return losses

    # following are just utilities only for convenience
    def expand(self, X):
        return torch.unsqueeze(X, dim=1)

    def get_hard_label(self, X):
        return torch.argmax(X, dim=1)
    
    def softmax(self, x):
        return torch.softmax(x, dim=1)

    def eval(self):
        self.isTrain = False
        for name in self.net_names:
            net = getattr(self, 'net' + name)
            net.eval()
    
    def train(self):
        self.isTrain = True
        for name in self.net_names:
            net = getattr(self, 'net' + name)
            net.train()
            




                
    


