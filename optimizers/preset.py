from torch import optim

def adam(params, lr=0.001, **kwargs):
    return optim.Adam(params, lr=lr, **kwargs)

def sgd(params, lr=0.001, **kwargs):
    return optim.SGD(params, lr=lr, **kwargs)

def adadelta(params, lr=0.001, **kwargs):
    return optim.Adadelta(params, lr=lr, **kwargs)

def adagrad(params, lr=0.001, **kwargs):
    return optim.Adagrad(params, lr=lr, **kwargs)

def adamw(params, lr=0.001, **kwargs):
    return optim.AdamW(params, lr=lr, **kwargs)

def adamax(params, lr=0.001, **kwargs):
    return optim.Adamax(params, lr=lr, **kwargs)

def asgd(params, lr=0.001, **kwargs):
    return optim.ASGD(params, lr=lr, **kwargs)

def rmsprop(params, lr=0.001, **kwargs):
    return optim.RMSprop(params, lr=lr, **kwargs)

def rprop(params, lr=0.001, **kwargs):
    return optim.Rprop(params, lr=lr, **kwargs)