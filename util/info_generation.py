import glob
import csv
from os import path as osp
import numpy as np
from PIL import Image
from tqdm import tqdm


def generate_info(root, dtype='.png'):
    """Iterate over the directory of dataset and generate image informatino for each sample
    
    Parameters:
        root                -- directory of the dataset
        dtype               -- type of image sample
    
    Returns:
        infos               -- information dictionary of each image
    """
    infos = []
    for path in glob.glob(osp.join(root, 'image', '*' + dtype)):
        name = path.split('/')[-1]
        info = {
            'image_path': path,
            'label_path': osp.join(root, 'label', name),
            'image_name': name
        }
        infos.append(info)
    return infos

def generate_class_map(root, name='mapping.csv'):
    """Class Mapping format is located under root directory of the dataset, with the format of "Class id, Class name, New id, New name"
    """
    mapping = {}
    with open(osp.join(root, name+'.csv'), 'r') as f:
        text = csv.reader(f, delimiter='\t')
        text = list(text)[1:]
        for item in text:
            [cid, cname, newid, newname] = item
            mapping[int(cid)] = {"name":cname, "newid":int(newid), "newname":newname}
    return mapping

def calculate_statistics(dataroot, phase):
    infos = generate_info(osp.join(dataroot, phase))
    p_sum = np.zeros(3, dtype=np.ulonglong)
     
    p_sqt_sum = np.zeros(3, dtype=np.ulonglong)
    p_count = np.zeros(3, dtype=np.ulonglong)
    for info in tqdm(infos, desc="Calculating mean and std for dataset:%s(phase:%s)" % (dataroot, phase)):
        img_path = info["image_path"]
        img = np.asarray(Image.open(img_path).convert("RGB"))
        img = img.transpose(2, 0, 1).reshape(3, -1)
        print(img.sum(1), (img ** 2).sum(1))
        break
        p_count += img.shape[1]
        p_sum += img.sum(1)
        p_sqt_sum += (np.power(img, 2)).sum(1)
    mean = p_sum / p_count
    print(p_sqt_sum , p_sum)
    std = np.sqrt((p_sqt_sum / p_count) - (mean ** 2))
    # print("[Finish Calculating mean and std]: mean: %d, std: %d" % (mean, std))
    return mean, std

        