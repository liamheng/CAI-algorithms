from metrics.base_metric import BaseMetrics
import numpy as np

def iou():
    return IoU()

class IoU(BaseMetrics):
    def __call__(self, preds, gts):
        iou = []
        for pred, gt in zip(preds, gts):
            tp, fp, fn = self.confusion_matrix(pred, gt)
            iou.append(self.cal_iou(tp, fp, fn))
        iou = np.stack(iou, axis=0)
        return (iou.mean(axis=0), iou.std(axis=0))
                
    def cal_iou(self, tps, fps, fns):
        ps = fns + fps + tps
        ps[ps==0] = 1e-3 
        iou = tps / ps
        return iou*100