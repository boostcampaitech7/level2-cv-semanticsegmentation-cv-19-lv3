import numpy as np
import os
import cv2
import json
import torch
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

def get_sorted_files_by_type(root_path, file_type='json'):
    current_dir = os.getcwd()
    files = {
        os.path.relpath(os.path.join(root, fname), start=current_dir)
        for root, _dirs, files in os.walk(root_path)
        for fname in files
        if os.path.splitext(fname)[1].lower() == '.' + file_type
    }
    files = sorted(files)

    return files

def split_data(pngs, jsons, kfold=5, k=0):
    assert k < kfold

    _filenames = np.array(pngs)
    _labelnames = np.array(jsons)

    groups = [os.path.dirname(fname) for fname in _filenames]

    ys = [0 for fname in _filenames]

    gkf = GroupKFold(n_splits=kfold)

    train_datalist, valid_datalist = dict(filenames = [], labelnames = []), dict(filenames = [], labelnames = [])

    for idx, (x, y) in enumerate(gkf.split(_filenames, ys, groups)):
        if idx != k:
            train_datalist['filenames'] += list(_filenames[y])
            train_datalist['labelnames'] += list(_labelnames[y])
        if (k < 0 and idx == 0) or (idx == k):
            valid_datalist['filenames'] += list(_filenames[y])
            valid_datalist['labelnames'] += list(_labelnames[y])

    return train_datalist, valid_datalist

class FullDataset(Dataset):
    def __init__(self, image_files, label_files=None, transforms=None):
        self.image_files = image_files
        self.label_files = label_files
        self.transforms = transforms
    
    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, item):
        image_path = self.image_files[item]
        image_name = os.path.basename(image_path)

        image = cv2.imread(image_path)
        image = image.astype(np.float32) / 255.0
        
        if self.label_files:
            label_path = self.label_files[item]
            label_shape = tuple(image.shape[:2]) + (len(CLASSES), )
            label = np.zeros(label_shape, dtype=np.uint8)
            
            with open(label_path, "r") as f:
                annotations = json.load(f)["annotations"]

            for ann in annotations:
                c = ann["label"]
                class_ind = CLASS2IND[c]
                points = np.array(ann["points"])
                
                class_label = np.zeros(image.shape[:2], dtype=np.uint8)
                cv2.fillPoly(class_label, [points], 1)
                label[..., class_ind] = class_label
        else:
            label = np.zeros((len(CLASSES), *image.shape[:2]), dtype=np.uint8)
        
        if self.transforms:
            inputs = {"image": image, "mask": label} if self.label_files else {"image": image}
            result = self.transforms(**inputs)
            image = result["image"]
            label = result["mask"] if self.label_files else label

        image = torch.from_numpy(image.transpose(2, 0, 1)).float()
        label = torch.from_numpy(label.transpose(2, 0, 1)).float() if self.label_files else None

        return (image_name, image, label) if label is not None else (image_name, image)