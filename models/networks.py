import torch
from torch.nn import init
import schedulers

def get_scheduler(optimzier, opt):
    scheduler = schedulers.find_scheduler(opt.lr_policy)
    return scheduler(opt, optimzier)

def define_network(input_nc, output_nc, netTask, device, initialization='normal', init_gain=0.02, extra_configs=None):
    
    net = None
    import network_bank as bank
    
    if netTask == 'deeplabv3_resnet101':
        net = bank.resnet101(pretrained=True, num_classes=output_nc)
    elif netTask == 'deeplabv3_resnet50':
        net = bank.resnet50(pretrained=extra_configs["pretrained"] if extra_configs is not None else False, num_classes=output_nc)
    elif netTask == 'res101_encoder':
        net = bank.res101_encoder(pretrained=True, num_classes=output_nc)
    elif netTask == 'res50_encoder':
        net = bank.res50_encoder(pretrained=extra_configs["pretrained"] if extra_configs is not None else False, num_classes=output_nc)
    elif netTask == 'aspp':
        net = bank.ASPP(512 * bank.Bottleneck.expansion, 256, output_nc)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % netTask)
    return initialize_network(net, device, initialization, init_gain, )

def initialize_network(net, device, initialization='normal', init_gain=0.02):
    # load network into gpu
    if device != -1 and torch.cuda.is_available():
        net.to(device)
    
    # intialize weights:
    if not isinstance(initialization, str):
        net.apply(initialization)
    elif initialization == "in_model":
        return net
    else:
        def init_func(m):  # define the initialization function
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if initialization == 'normal':
                    init.normal_(m.weight.data, 0.0, init_gain)
                elif initialization == 'xavier':
                    init.xavier_normal_(m.weight.data, gain=init_gain)
                elif initialization == 'kaiming':
                    init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif initialization == 'orthogonal':
                    init.orthogonal_(m.weight.data, gain=init_gain)
                else:
                    raise NotImplementedError('initialization method [%s] is not implemented' % initialization)
                if hasattr(m, 'bias') and m.bias is not None:
                    init.constant_(m.bias.data, 0.0)
            elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
                init.normal_(m.weight.data, 1.0, init_gain)
                init.constant_(m.bias.data, 0.0)
        net.apply(init_func)  
    print('initialize module %s with %s' % (net.__class__.__name__, initialization))
    return net