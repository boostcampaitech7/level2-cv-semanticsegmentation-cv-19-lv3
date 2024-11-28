# python native
import os
import json
import cv2
import numpy as np
import torch
import pandas as pd
from torch.utils.data import Dataset
from sklearn.model_selection import StratifiedGroupKFold

CLASSES = [
    'finger-1', 'finger-2', 'finger-3', 'finger-4', 'finger-5',
    'finger-6', 'finger-7', 'finger-8', 'finger-9', 'finger-10',
    'finger-11', 'finger-12', 'finger-13', 'finger-14', 'finger-15',
    'finger-16', 'finger-17', 'finger-18', 'finger-19', 'Trapezium',
    'Trapezoid', 'Capitate', 'Hamate', 'Scaphoid', 'Lunate',
    'Triquetrum', 'Pisiform', 'Radius', 'Ulna',
]

class XRayDataset(Dataset):
    def __init__(self, fnames, labels, image_root, label_root, kfold=0, transforms=None, is_train=True):
        self.is_train = is_train
        self.transforms = transforms
        self.image_root = image_root
        self.label_root = label_root
        self.kfold = kfold
        self.class2ind = {v: i for i, v in enumerate(CLASSES)}
        self.ind2class = {v: k for k, v in self.class2ind.items()}
        self.classes = CLASSES
        
        groups = [os.path.dirname(fname) for fname in fnames]
        
        meta_data = pd.read_excel('update_meta_data.xlsx')
        
        # dummy label
        ys = []
        for fname in fnames:
            folder_name = os.path.dirname(fname)
            id_number = int(folder_name[2:])
            
            row = meta_data.iloc[id_number - 1]
            ys.append(row['rotate'])
        

        gkf = StratifiedGroupKFold(n_splits=5)
        
        filenames = []
        labelnames = []
        for i, (x, y) in enumerate(gkf.split(fnames, ys, groups)):
            if self.is_train: # k번 빼고 학습
                if i == self.kfold:
                    continue
                filenames += list(fnames[y])
                labelnames += list(labels[y])
            
            else:  # k번은 검증
                if i == self.kfold:
                    filenames = list(fnames[y])
                    labelnames = list(labels[y])
                    break
                
        self.filenames = filenames
        self.labelnames = labelnames
    
    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, item):
        image_name = self.filenames[item]
        image_path = os.path.join(self.image_root, image_name)
        
        image = cv2.imread(image_path)
        image = image / 255.
        
        label_name = self.labelnames[item]
        label_path = os.path.join(self.label_root, label_name)
        
        # (H, W, NC) 모양의 label을 생성합니다.
        label_shape = tuple(image.shape[:2]) + (len(CLASSES), ) # (2048, 2048, 29)
        label = np.zeros(label_shape, dtype=np.uint8)
        
        # label 파일을 읽습니다.
        with open(label_path, "r") as f:
            annotations = json.load(f)
        annotations = annotations["annotations"]
        
        # 클래스 별로 처리합니다.
        for ann in annotations:
            c = ann["label"] # class name
            class_ind = self.class2ind[c] # to index
            points = np.array(ann["points"]) # mask
            
            # polygon 포맷을 dense한 mask 포맷으로 바꿉니다.
            class_label = np.zeros(image.shape[:2], dtype=np.uint8) # (2048, 2048)
            cv2.fillPoly(class_label, [points], 1) # points 좌표의 점을 1로 채우기
            label[..., class_ind] = class_label
        
        if self.transforms is not None:
            inputs = {"image": image, "mask": label} if self.is_train else {"image": image}
            result = self.transforms(**inputs)
            
            image = result["image"]
            label = result["mask"] if self.is_train else label

        # to tenser will be done later
        image = image.transpose(2, 0, 1)    # channel first 포맷으로 변경합니다.
        label = label.transpose(2, 0, 1)
        
        image = torch.from_numpy(image).float()
        label = torch.from_numpy(label).float()
        
        return image, label

class XRayDatasetWithMixup(XRayDataset):
    def __init__(self, fnames, labels, image_root, label_root, kfold=0, transforms=None, 
                is_train=True, mixup_prob=0.5, mixup_alpha=0.5):
        super().__init__(fnames, labels, image_root, label_root, kfold, transforms, is_train)
        self.mixup_prob = mixup_prob  # mixup을 적용할 확률
        self.mixup_alpha = mixup_alpha  # Beta 분포의 알파 값
        
    def __getitem__(self, item):
        image, label = super().__getitem__(item)
        
        # 학습 시에만 mixup을 적용하고, mixup_prob 확률로 적용
        if self.is_train and torch.rand(1) < self.mixup_prob:
            # 랜덤하게 다른 인덱스 선택
            other_idx = torch.randint(len(self), size=(1,)).item()
            other_image, other_label = super().__getitem__(other_idx)
            
            # Beta 분포에서 혼합 비율 샘플링
            lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
            
            # 이미지와 라벨 혼합
            mixed_image = lam * image + (1 - lam) * other_image
            mixed_label = lam * label + (1 - lam) * other_label
            
            return mixed_image, mixed_label
        
        return image, label

class XRayInferenceDataset(Dataset):
    def __init__(self, fnames, image_root, transforms=None):
        self.fnames = np.array(sorted(fnames))
        self.image_root = image_root
        self.transforms = transforms
        self.ind2class = {i: v for i, v in enumerate(CLASSES)}

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, item):
        image_name = self.fnames[item]
        image_path = os.path.join(self.image_root, image_name)

        image = cv2.imread(image_path)
        image = image / 255.

        if self.transforms is not None:
            inputs = {"image": image}
            result = self.transforms(**inputs)
            image = result["image"]

        # to tenser will be done later
        image = image.transpose(2, 0, 1)    # make channel first

        image = torch.from_numpy(image).float()

        return image, image_name