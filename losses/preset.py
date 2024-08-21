from torch import nn

def cross_entropy(**kwargs):
    return nn.CrossEntropyLoss(**kwargs)

def l1(**kwargs):
    return nn.L1Loss(**kwargs)

def bce(**kwargs):
    return nn.BCELoss(**kwargs)