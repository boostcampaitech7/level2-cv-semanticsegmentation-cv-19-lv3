import torchvision.transforms.functional as F
import numpy as np
import random
import os
import cv2
import json
import torch
import albumentations as A
from PIL import Image
from torchvision.transforms import InterpolationMode
from torch.utils.data import Dataset
from sklearn.model_selection import GroupKFold

CLASSES = [
    'finger-1', 'finger-2', 'finger-3', 'finger-4', 'finger-5',
    'finger-6', 'finger-7', 'finger-8', 'finger-9', 'finger-10',
    'finger-11', 'finger-12', 'finger-13', 'finger-14', 'finger-15',
    'finger-16', 'finger-17', 'finger-18', 'finger-19', 'Trapezium',
    'Trapezoid', 'Capitate', 'Hamate', 'Scaphoid', 'Lunate',
    'Triquetrum', 'Pisiform', 'Radius', 'Ulna',
]
CLASS2IND = {v: i for i, v in enumerate(CLASSES)}

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
    def __init__(self, image_root, gt_root, is_train=True, transforms=None, kfold=5, k=0):
        pngs = {
            os.path.relpath(os.path.join(root, fname), start=image_root)
            for root, _dirs, files in os.walk(image_root)
            for fname in files
            if os.path.splitext(fname)[1].lower() == ".png"
        }
        jsons = {
            os.path.relpath(os.path.join(root, fname), start=gt_root)
            for root, _dirs, files in os.walk(gt_root)
            for fname in files
            if os.path.splitext(fname)[1].lower() == ".json"
        }
        self.image_root = image_root
        self.gt_root = gt_root
        self.is_train = is_train
        pngs = sorted(pngs)
        jsons = sorted(jsons)

        _filenames = np.array(pngs)
        _labelnames = np.array(jsons)
        
        groups = [os.path.dirname(fname) for fname in _filenames]
        
        ys = [0 for fname in _filenames]
        
        gkf = GroupKFold(n_splits=kfold)
        
        filenames = []
        labelnames = []
        for i, (x, y) in enumerate(gkf.split(_filenames, ys, groups)):
            if is_train:
                if i == k:
                    continue
                filenames += list(_filenames[y])
                labelnames += list(_labelnames[y])
            
            else:
                if i != k:
                    continue
                filenames = list(_filenames[y])
                labelnames = list(_labelnames[y])
                break

        self.images = filenames
        self.labels = labelnames
        self.transform = transforms

    def __getitem__(self, idx):
        image_name = self.images[idx]
        image_path = os.path.join(self.image_root, image_name)
        
        image = cv2.imread(image_path)
        image = image / 255.
        
        label_name = self.gts[idx]
        label_path = os.path.join(self.gt_root, label_name)
        
        image_height, image_width = image.shape[:2]
        label_shape = (image_height, image_width, len(CLASSES),)
        label = np.zeros(label_shape, dtype=np.uint8)
        
        try:
            with open(label_path, "r") as f:
                annotations = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Label file not found at path: {label_path}")
        
        annotations = annotations.get("annotations", [])
        
        # Fill polygons in label
        for ann in annotations:
            c = ann.get("label")
            if c not in CLASS2IND:
                continue
            
            class_ind = CLASS2IND[c]
            points = np.array(ann.get("points", []), dtype=np.int32)
            
            if points.size == 0:
                continue
            
            class_label = np.zeros((image_height, image_width), dtype=np.uint8)
            cv2.fillPoly(class_label, [points], 1)
            label[..., class_ind] = class_label
        
        if self.transform is not None:
            inputs = {"image": image, "mask": label} if self.is_train else {"image": image}
            result = self.transform(**inputs)
            
            image = result["image"]
            label = result["mask"] if self.is_train else label

        image = image.transpose(2, 0, 1)
        label = label.transpose(2, 0, 1)
        
        image = torch.from_numpy(image).float()
        label = torch.from_numpy(label).float()
        
        return image, label

    def __len__(self):
        return len(self.images)

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

class TestDataset:
    def __init__(self, image_root, transform=None):
        pngs = {
            os.path.relpath(os.path.join(root, fname), start=image_root)
            for root, _dirs, files in os.walk(image_root)
            for fname in files
            if os.path.splitext(fname)[1].lower() == ".png"
        }
        pngs = sorted(pngs)

        self.image_root = image_root
        self.images = np.array(pngs)
        self.transform = transform
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image_name = self.images[idx]
        image_path = os.path.join(self.image_root, image_name)
        
        image = cv2.imread(image_path)
        image = image / 255.
        
        if self.transform is not None:
            inputs = {"image": image}
            result = self.transform(**inputs)
            image = result["image"]

        image = image.transpose(2, 0, 1)
        
        image = torch.from_numpy(image).float()
            
        return image, image_name