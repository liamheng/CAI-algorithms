import os.path as osp
import csv
import numpy as np
import argparse
import pandas as pd

from PIL import Image

# TODO 写IoU metrics
# TODO 测试的时候添加类别信息
# TODO 测试的时候添加namelist
# TODO 写Dice

#===========================================Metrics===========================================
def cal_dice(tps, fps, fns):
    denominator = 2 * tps + fps + fns
    denominator[denominator==0] = 1e-3
    dice = 2 * tps / denominator
    return dice*100

def cal_iou(tps, fps, fns):
    ps = fns + fps + tps
    ps[ps==0] = 1e-3 
    iou = tps / ps
    return iou*100
#===========================================Metrics===========================================

def get_name_list(names_path):
    name_list = []
    with open(names_path, 'r') as f:
        for line in f:
            line = line.rstrip('\n')
            name_list.append(line)
    return name_list

def get_classes(classes_path):
    classes = {}
    with open(classes_path, 'r') as f:
        lines = list(csv.reader(f,delimiter=','))[1:]
        for line in lines:
            index = int(line[0])
            name = line[1]
            classes[index] = name
    return classes

def cal_confusion_for_one_img(num_classes, gt, mask):
    tps = np.zeros(num_classes)
    fps = np.zeros(num_classes)
    fns = np.zeros(num_classes)
    # print(gt.shape, mask.shape)
    for label in range(num_classes):
        label_pred = (mask == label).astype(np.float64)
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

def summarise_statistics(matrix, index):
    matrix = matrix[:, index]
    mean = np.mean(matrix, axis=1, keepdims=True).round(4)
    return mean

class SegMetrics(object):
    
    def __init__(self, results_dir, per_cls=False):
        self.results_dir = results_dir
        self.per_sample = per_cls

        self.images_dir = osp.join(self.results_dir, 'image')
        self.classes = get_classes(osp.join(self.results_dir, 'classes.csv'))

        self.num_classes = len(self.classes)
        self.names = get_name_list(osp.join(self.results_dir, 'names.txt'))

    def load_image_and_convert(self, path):
        gt = Image.open(osp.join(self.images_dir, path+'_gt.png'))
        gt = np.asarray(gt, dtype=np.uint32).flatten()
        mask = Image.open(osp.join(self.images_dir, path+'_mask.png'))
        mask = np.asarray(mask, dtype=np.uint32).flatten()
        return gt, mask
    
    def output_results(self, metrics):
        for metric_name, metric in metrics.items():
            names, values = list(metric.keys()), list(metric.values())

            # post-process
            metric = np.stack(values, axis=0)
            names += ['overall']
            names = np.expand_dims(np.stack(names, axis=0), axis=1)
            # conclusion
            column_name = ['name'] + list(self.classes.values()) + ["anatomy", "instrument", "overall"]

            # calculate anatomy, instrument, overall metrics results
            selects = [[0, 1, 2, 3], [4, 5], range(self.num_classes)]
            for select in selects:
                mean = summarise_statistics(metric, select)
                metric = np.concatenate([metric, mean], axis=1)
            
            # add average score
            metric_mean = metric.mean(axis=0, keepdims=True)
            metric = np.concatenate([metric, metric_mean], axis=0)

            # add names 
            metric = np.concatenate([names, metric], axis=1)
            
            with open(osp.join(self.results_dir, 'metrics_%s.csv' % metric_name), 'w') as fi:
                csv_writer = csv.writer(fi, delimiter=',')
                csv_writer.writerow(column_name)
                csv_writer.writerows(metric)
    
    def cal_metrics_sample_wise(self):
        metrics = dict(ious=dict(), dices=dict())
        for file_name in self.names:
            gt, mask = self.load_image_and_convert(file_name)
            tp, fp, fn = cal_confusion_for_one_img(self.num_classes, gt, mask)
            iou = cal_iou(tp, fp, fn)
            dice = cal_dice(tp, fp, fn)
            metrics["ious"][file_name] = iou
            metrics["dices"][file_name] = dice
        return metrics
    
    def run(self):
        metrics = self.cal_metrics_sample_wise()
        self.output_results(metrics)
                
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', type=str, required=True)
    args = parser.parse_args()
    metrics = SegMetrics(args.results_dir)
    metrics.run()


# deprecated
    # def cal_metrtics_dataset_wise(self):
    #     metrics = {}
    #     tps, fps, fns = [], [], []
    #     for file_name in self.names:
    #         gt, mask = self.load_image_and_convert(file_name)
    #         tp, fp, fn = cal_confusion_for_one_img(self.num_classes, gt, mask)
    #         tps.append(tp)
    #         fps.append(fp)
    #         fns.append(fn)    
    #     tps = np.asarray(tps).sum(axis=0)
    #     fps = np.asarray(fps).sum(axis=0)
    #     fns = np.asarray(fns).sum(axis=0)
    #     ious = cal_iou(tps, fps, fns)
    #     metrics["iou"] = ious
    #     dices = cal_dice(tps, fps, fns)
    #     metrics["dice"] = dices
    #     return metrics