import argparse
import os
import torch
import numpy as np
import nibabel as nib
from medpy.metric.binary import hd95  

parser = argparse.ArgumentParser()
parser.add_argument("--test_image_path", type=str, required=True, 
                    help="path to the prediction/image files")
parser.add_argument("--test_gt_path", type=str, required=True,
                    help="path to the mask files")
args = parser.parse_args()


def load_nii_data(image_path):
    images = sorted(os.listdir(image_path))
    images_list = []
    for image in images:
        image_path_ = os.path.join(image_path, image)
        img = nib.load(image_path_).get_fdata()  
        
        for z in range(img.shape[2]):
            slice_ = img[:, :, z]
            images_list.append(slice_)
        
    return images_list


def iou_score(output, target):
    smooth = 1e-5

    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
        
    # 生成二值化掩码
    output_ = output > 0.5
    target_ = target > 0.5
    
    intersection = (output_ & target_).sum()
    union = (output_ | target_).sum()
    iou = (intersection + smooth) / (union + smooth)
    dice = (2 * iou) / (iou + 1)
    
    return iou, dice, output_, target_


def f1_score(output_, target_):
    """计算F1 Score"""
    true_positive = (output_ & target_).sum()
    false_positive = ((output_ == 1) & (target_ == 0)).sum()
    false_negative = ((output_ == 0) & (target_ == 1)).sum()

    precision = true_positive / (true_positive + false_positive + 1e-5)
    recall = true_positive / (true_positive + false_negative + 1e-5)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-5)

    return f1, precision, recall


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# 初始化记录器
iou_avg_meter = AverageMeter()
dice_avg_meter = AverageMeter()
precision_avg_meter = AverageMeter()
recall_avg_meter = AverageMeter()
hd95_avg_meter = AverageMeter()  


images_dict = load_nii_data(args.test_image_path)
labels_dict = load_nii_data(args.test_gt_path)


for image, gt in zip(images_dict, labels_dict):

    gt_has_content = False
    if torch.is_tensor(gt):
        gt_has_content = torch.sum(gt) > 0
    else:
        gt_has_content = np.sum(gt) > 0
    
    if gt_has_content:
        iou, dice, output_, target_ = iou_score(image, gt)
        f1, precision, recall = f1_score(output_, target_)

        iou_avg_meter.update(iou, 1)
        dice_avg_meter.update(dice, 1)
        precision_avg_meter.update(precision, 1)
        recall_avg_meter.update(recall, 1)
 
        if np.sum(output_) > 0:
            hd_val = hd95(output_, target_)
            hd95_avg_meter.update(hd_val, 1)
        else:
            pass 
        
    else:
        continue

print('IoU: %.4f' % iou_avg_meter.avg)
print('Dice: %.4f' % dice_avg_meter.avg) 
print('Precision: %.4f' % precision_avg_meter.avg)
print('Recall: %.4f' % recall_avg_meter.avg)
print('HD95: %.4f' % hd95_avg_meter.avg)