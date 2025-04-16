# -*- coding: UTF-8 -*-
import torch
from models import BaseModel
from losses import SoftmaxEntropyLoss
from models.networks import define_network

def model_initialize(model, file_path, device='cpu'):
    model_dict = model.state_dict()
    state_dict = torch.load(file_path, map_location=str(device))
    if hasattr(state_dict, '_metatdata'):
        del state_dict.metadata
    overlap_dict = {k: v for k, v in state_dict.items() if k in model_dict}
    model_dict.update(overlap_dict)
    model.load_state_dict(model_dict)
    return model

class DecoderBlock(torch.nn.Module):
    def __init__(self, opt):
        super(DecoderBlock, self).__init__()
        self.opt = opt
        self.netD1 = define_network(self.opt.input_nc, self.opt.output_nc, 'aspp', self.opt.gpu, self.opt.initialization, self.opt.init_gain)
        self.netD2 = define_network(self.opt.input_nc, self.opt.output_nc, 'aspp', self.opt.gpu, self.opt.initialization, self.opt.init_gain)

    def initialize(self, file_path, device='cpu', ):
        self.netD1 = model_initialize(self.netD1, file_path.replace('net_D', 'net_D1'), device)
        self.netD2 = model_initialize(self.netD2, file_path.replace('net_D', 'net_D2'), device)
    
    def forward(self, x):
        x1 = self.netD1(x)
        x2 = self.netD2(x)
        return x1 + x2

class MUTASourceModel(BaseModel):

    @staticmethod
    def modify_commandline_options(parser, isTrain=True):
        parser.add_argument("--dropout_rates", type=float, nargs='+', default=[0.5])
        parser.add_argument("--dropout_weights", type=float, nargs='+', default=[1])
        if isTrain:
            parser.add_argument("--ema_freq", type=int, default=10)
            parser.add_argument("--momentum", type=float, default=0.9)
            parser.add_argument("--mu_ssl", type=float, default=1)
            parser.add_argument("--mu_ent", type=float, default=1)
            parser.add_argument("--mu_con", type=float, default=0.001)
            parser.add_argument("--threshold", type=float, default=0.95)
            parser.add_argument("--file_path", type=str, default="checkpoints/deeplabv3_res101/100_net_Task.pth")
        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)
    
    def register_nets(self,):
        # define networks
        self.net_names = ['ES', 'ET', 'D']
        assert self.opt.netTask == 'res50'
        self.netES = define_network(self.opt.input_nc, self.opt.output_nc, 'res50_encoder', self.opt.gpu, self.opt.initialization, self.opt.init_gain)
        self.netET = define_network(self.opt.input_nc, self.opt.output_nc, 'res50_encoder', self.opt.gpu, self.opt.initialization, self.opt.init_gain)

        self.netD = DecoderBlock(self.opt)
        
        if self.isTrain:
        # initialize networks
            encoder_file_path = self.opt.file_path.replace('net_Task', 'net_E')
            decoder_file_path = self.opt.file_path.replace('net_Task', 'net_D')

            self.netES = model_initialize(self.netES, encoder_file_path, self.device)
            self.netET = model_initialize(self.netET, encoder_file_path, self.device)
            self.netD.initialize(decoder_file_path, self.device)
            self.netES.requires_grad_(False)
            print("[Load Networks] load network %s from %s" % ("name", self.opt.file_path))
            
        # collect bn params of encoders
        self.ES_params = self.collect_ema_params(self.netES)

        # define dropout module
        self.dropouts = [torch.nn.Dropout(rate) for rate in self.opt.dropout_rates]
        self.dropout_weights = self.opt.dropout_weights

    def register_optimizers(self):
        param = list(self.netET.parameters())
        # define optimizers
        self.optimizer_ET = torch.optim.Adam(param, lr=self.opt.lr)

    def register_losses(self):
        self.loss_names = [
            'ssl', 
            'con', 
            'ent',
            ]
        self.criterion_ssl = torch.nn.CrossEntropyLoss(ignore_index=self.opt.ignore_label)
        self.criterion_con = torch.nn.MSELoss()
        self.criterion_ent = SoftmaxEntropyLoss()

    def register_visuals(self):
        self.visual_names = [
            'rgb','color_gt', 
            'color_pred_S', 
            'color_pred_T', 
            'color_pred'
            ]
        for i in range(len(self.dropouts)):
            self.visual_names.append('color_pred_T%d' % (i+1))
           
        self.visual_names += ['gt', 'mask'] if not self.isTrain else [ 'color_pseudo']

    def set_inputs(self, data):
        self.rgb = data['image'].to(self.device)
        self.gt = data['label'].to(self.device, dtype=torch.long)

    def optimize(self):
        self.optimizer_ET.zero_grad()
        self.netD.zero_grad()
        self.forward()
        self.backward()
        self.optimizer_ET.step()

    def forward(self):
        size = (self.rgb.shape[2], self.rgb.shape[3])
        # extract feature
        self.feature_S = self.netES(self.rgb)
        self.feature_T = self.netET(self.rgb)

        # pertubate target feature
        for i in range(len(self.dropouts)):
            # set dropout featture using reflection
            setattr(self, 'netDrop%d' % (i+1), self.dropouts[i])
        
        # get source model prediction
        with torch.no_grad():
            self.pred_S = torch.nn.Upsample(size, mode='bilinear', align_corners=True)(self.netD(self.feature_S))

        # get vallina target model prediction
        self.pred_T = torch.nn.Upsample(size, mode='bilinear', align_corners=True)(self.netD(self.feature_T))

        # get dropout prediction
        self.pred_drops = []
        for i in range(len(self.dropouts)):
            # get dropout pred using reflection
            feature = getattr(self, 'netDrop%d' % (i+1))(self.feature_T)
            pred = torch.nn.Upsample(size, mode='bilinear', align_corners=True)(self.netD(feature))
            setattr(self, 'pred_T%d' % (i+1), pred)

        # get final target model prediction
        self.pred = self.pred_S

        # generate pesudo label
        if self.isTrain:
            prob, index = torch.max(torch.softmax(self.pred_S, dim=1), dim=1)
            mask = prob > self.opt.threshold
            self.pseudo = torch.where(mask, index, torch.ones_like(index)*self.ignore_label).long()
    
    def backward(self):
        # update student
        self.loss_ent = self.criterion_ent(self.pred_T) * self.opt.mu_ent

        self.loss_ssl = self.criterion_ssl(self.pred, self.pseudo) * self.opt.mu_ssl

        self.loss_con = 0
        for i in range(len(self.dropouts)):
            pred = getattr(self, 'pred_T%d' % (i+1))
            loss = self.criterion_con(torch.softmax(pred, dim=1), torch.softmax(self.pred_S, dim=1))
            self.loss_con += loss
        self.loss_con = self.loss_con / len(self.dropouts) * self.opt.mu_con

        self.loss = self.loss_con + self.loss_ssl + self.loss_ent    
        self.loss.backward()

        # update teacher
        if self.n_iters != 0 and self.n_iters % self.opt.ema_freq == 0:
            ES_state_dict = self.netES.state_dict()
            ET_state_dict = self.netET.state_dict()
            for k in self.ES_params:
                ES_state_dict[k] = self.opt.momentum * ES_state_dict[k] + (1 - self.opt.momentum) * ET_state_dict[k]
            self.netES.load_state_dict(ES_state_dict)
            # print("[Exponential Moving Average] update teacher encoder")

    def visualization_preprocess(self):
        with torch.no_grad():
            self.color_gt = self.expand(self.gt)
            self.color_pred_S = self.expand(self.get_hard_label(self.pred_S))
            self.color_pred_T = self.expand(self.get_hard_label(self.pred_T))
            self.color_pred = self.color_pred_T
            for i in range(len(self.dropouts)):
                pred = getattr(self, 'pred_T%d' % (i+1))
                setattr(self, 'color_pred_T%d' % (i+1), self.expand(self.get_hard_label(pred)))
            self.pred = self.expand(self.get_hard_label(self.pred))
            if self.isTrain:
                self.color_pseudo = self.expand(self.pseudo)
            self.mask = self.get_hard_label(self.pred_S)

    def collect_ema_params(self, model):
        params = []
        # encoder ema
        for nm, m in model.named_modules():
            for np, p in m.named_parameters():
                if np in ['weight', 'bias']:
                    params.append(nm + '.' + np)        
        return params
