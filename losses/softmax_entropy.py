import torch.nn as nn
import torch

class SoftmaxEntropyLoss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        loss = -(x.softmax(1) * x.log_softmax(1)).sum(1)
        return loss.mean()
    
class MaxSquareloss(nn.Module):
    def __init__(self, ignore_index= -1):
        super().__init__()
        self.ignore_index = ignore_index
    
    def forward(self, pred, prob):
        """
        :param pred: predictions (N, C, H, W)
        :param prob: probability of pred (N, C, H, W)
        :return: maximum squares loss
        """
        # prob -= 0.5
        mask = (prob != self.ignore_index)    
        loss = -torch.mean(torch.pow(prob, 2)[mask]) / 2
        return loss