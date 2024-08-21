from metrics.base_metric import BaseMetrics
import numpy as np

def dice():
    return Dice()

class Dice(BaseMetrics):
    def __call__(self, preds, gts):
        dice = []
        for pred, gt in zip(preds, gts):
            tp, fp, fn = self.confusion_matrix(pred, gt)
            dice.append(self.cal_dice(tp, fp, fn))
        dice = np.stack(dice, axis=0)
        return (dice.mean(axis=0), dice.std(axis=0))

    def cal_dice(self, tps, fps, fns):
        denominator = 2 * tps + fps + fns
        denominator[denominator==0] = 1e-3
        dice = 2 * tps / denominator
        return dice*100