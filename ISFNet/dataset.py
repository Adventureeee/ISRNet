import torchvision.transforms.functional as F
import numpy as np
import random
import os
import imageio
import matplotlib.pyplot as plt
from PIL import Image
from torchvision.transforms import InterpolationMode
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import nibabel as nib
from scipy.ndimage import zoom


class ToTensor(object):

    def __call__(self, data):
        image, label = data['image'], data['label']
        return {'image': F.to_tensor(image), 'label': F.to_tensor(label)}


class Resize(object):

    def __init__(self, size):
        self.size = size

    def __call__(self, data):
        image, label = data['image'], data['label']

        return {'image': F.resize(image, self.size), 'label': F.resize(label, self.size, interpolation=InterpolationMode.BICUBIC)}


class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, data):
        image, label = data['image'], data['label']

        if random.random() < self.p:
            return {'image': F.hflip(image), 'label': F.hflip(label)}

        return {'image': image, 'label': label}


class RandomVerticalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, data):
        image, label = data['image'], data['label']

        if random.random() < self.p:
            return {'image': F.vflip(image), 'label': F.vflip(label)}

        return {'image': image, 'label': label}


class Normalize(object):
    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        image = F.normalize(image, self.mean, self.std)
        return {'image': image, 'label': label}

    

class FullDataset(Dataset):
    def __init__(self, image_root, gt_root, size, mode):
        
        self.videos = []
        self.masks = []

        vid_list = sorted(os.listdir(image_root))
        mask_list = sorted(os.listdir(gt_root))
        # Pre-reading
        for vid in vid_list:
            videos_path = os.path.join(image_root, vid)
            self.videos.append(videos_path)
        
        for mask in mask_list:
            masks_path = os.path.join(gt_root, mask)
            self.masks.append(masks_path)

        if mode == 'train':
            self.transform = transforms.Compose([
                Resize((size, size)),
                ToTensor(),
                Normalize()
            ])
        else:
            self.transform = transforms.Compose([
                Resize((size, size)),
                ToTensor(),
                Normalize()
            ])

    def __getitem__(self, idx):
        
        video_data = load_nii_data(self.videos[idx])
        label_data = load_nii_data(self.masks[idx])
        image = []
        label = []
        
        for depth in range(video_data.shape[2]):
            slice_ = self.rgb_loader(video_data[:, :, depth])
            mask_ = self.binary_loader(label_data[:, :, depth])
            data = {'image': slice_, 'label': mask_}
            data = self.transform(data)
            image.append(data['image'])
            label.append(data['label'])
        
        data = {'image': image, 'label': label}
        return data

    def __len__(self):
        return len(self.videos)

    def rgb_loader(self, image_data):
        if isinstance(image_data, np.ndarray):
            img = Image.fromarray(image_data.astype(np.uint8))
            return img.convert('RGB')  
        else:
            with open(image_data, 'rb') as f:
                img = Image.open(f)
                return img.convert('RGB')  

    def binary_loader(self, gt_data):
        if isinstance(gt_data, np.ndarray):
            img = Image.fromarray(gt_data.astype(np.uint8))
            return img.convert('L') 
        else:
            with open(gt_data, 'rb') as f:
                img = Image.open(f)
                return img.convert('L')
        

class TestDataset:
    def __init__(self, image_root, size):
        self.image = []
        self.images = sorted(os.listdir(image_root))
        for image in self.images:
            image_path = os.path.join(image_root, image)
            self.image.append(image_path)
        
        self.transform = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])

        self.size = len(self.images)
        self.index = 0

    def load_data(self):
        image = load_nii_data(self.image[self.index])

        h,w,d = image.shape
        size = (h, w)
        image_data = []

        
        for depth in range(image.shape[2]):
            slice_ = self.rgb_loader(image[:, :, depth])
            data = self.transform(slice_)
            image_data.append(data)

        image_name = self.images[self.index]
        self.index += 1

        return image_data, image_name, size
    

    def rgb_loader(self, image_data):
        if isinstance(image_data, np.ndarray):
            img = Image.fromarray(image_data.astype(np.uint8))
            return img.convert('RGB') 
        else:
            with open(image_data, 'rb') as f:
                img = Image.open(f)
                return img.convert('RGB') 

    def binary_loader(self, gt_data):
        if isinstance(gt_data, np.ndarray):
            img = Image.fromarray(gt_data.astype(np.uint8))
            return img.convert('L')  
        else:
            with open(gt_data, 'rb') as f:
                img = Image.open(f)
                return img.convert('L')
            
def load_nii_data(image_path):

    img = nib.load(image_path).get_fdata() 
    normalized_slice = np.zeros_like(img)
    for z in range(img.shape[2]):
        slice_ = img[:, :, z]
    
        if np.max(slice_) > np.min(slice_):
            normalized_slice[:, :, z] = ((slice_ - np.min(slice_)) / (np.max(slice_) - np.min(slice_)) * 255).astype(np.uint8)
        else:
            normalized_slice[:, :, z] = np.zeros_like(slice_, dtype=np.uint8)  
        

    return normalized_slice

