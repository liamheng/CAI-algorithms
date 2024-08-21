from abc import ABC, abstractmethod
from utility.configuration import CONFIG
import numpy as np

class BaseMetrics(ABC):
    def __init__(self):
        self.num_classes = CONFIG['NUM_CLASSES']

    @abstractmethod
    def __call__(self, preds, gts):
        pass

    def confusion_matrix(self, pred, gt):
        tps = np.zeros(self.num_classes)
        fps = np.zeros(self.num_classes)
        fns = np.zeros(self.num_classes)
        for label in range(self.num_classes):
            label_pred = (pred == label).astype(np.float64)
            label_gt = (gt == label).astype(np.float64)

            tp = (label_pred * label_gt).sum()
            diff = label_pred - label_gt
            fp = diff.clip(0).sum()
            fn = (-diff).clip(0).sum()
            assert(tp + fn == label_gt.sum() )
            assert(tp + fp == label_pred.sum())

            tps[label] += tp
            fps[label] += fp
            fns[label] += fn

        return tps, fps, fns