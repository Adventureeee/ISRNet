import argparse
import os
from typing_extensions import Required
import torch
import imageio
import numpy as np
import torch.nn.functional as F
from scipy.ndimage import label
from ISFNet import ISFNet
from dataset import TestDataset
import nibabel as nib
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str, required=True, 
                help="path to the checkpoint of ISFNet")
parser.add_argument("--test_image_path", type=str, required=True,
                    help="path to the image files for testing")
parser.add_argument("--save_path", type=str, required=True,
                    help="path to save the predicted masks")
args = parser.parse_args()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
test_loader = TestDataset(args.test_image_path, 352)
model = ISFNet().to(device)
model.load_state_dict(torch.load(args.checkpoint), strict=True)
model.eval()
model.cuda()
os.makedirs(args.save_path, exist_ok=True)

def save_images_as_nii(pred_image, save_path, name):


    reference_nii_path = os.path.join(args.test_image_path, name)
    reference_nii = nib.load(reference_nii_path)
    reference_affine = reference_nii.affine
    reference_dtype = reference_nii.get_data_dtype()


    # 获取当前批次的图像
    img_data = np.array(pred_image, dtype=reference_dtype)  # 转换为 NumPy 数组
    img_data = np.transpose(img_data, (1, 2, 0))  # 变为 (H, W, D)
    
    # 生成 NIfTI 图像对象
    nii_image = nib.Nifti1Image(img_data, affine=reference_affine)  # 创建nii对象
    

    save_path = os.path.join(args.save_path, f'{name}')
    nib.save(nii_image, save_path)

for i in range(test_loader.size):
    # print(test_loader.size)
    with torch.no_grad():
        image, image_name, image_size = test_loader.load_data()
        # print(image[0].dtype)
        image = torch.stack(image, dim=0)
        # print(image.shape)
        image = image.to(device)
        
        prev_image = None
        prev_mask = None
        
        pred_image = []
        for frame_idx in range(image.shape[0]): 
                
                single_frame = image[frame_idx:frame_idx+1, :, :, :] 
                out1, out2,  out = model(single_frame, prev_image, prev_mask)  # 每次只传入一帧
                
                prev_mask = out
                prev_image = single_frame
                
                res = out
                res = F.interpolate(res, size=image_size, mode='bilinear', align_corners=False)
                res = res.sigmoid().data.cpu()
                res = res.numpy().squeeze()
                res = (res - res.min()) / (res.max() - res.min() + 1e-8)
                res = (res * 255).astype(np.uint8)
                res[res>210] = 255
                res[res<=210] = 0
                
                res[res!=0] = 1
                
                pred_image.append(res)

        pred_image = np.stack(pred_image, axis=0)
        save_images_as_nii(pred_image, args.save_path, image_name)





   





 


