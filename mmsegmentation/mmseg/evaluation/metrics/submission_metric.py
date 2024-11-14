from collections import defaultdict
import os

import numpy as np
import pandas as pd
from prettytable import PrettyTable

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from mmseg.registry import METRICS

from mmengine.evaluator import BaseMetric
from mmengine.logging import MMLogger, print_log
from tqdm import tqdm



CLASSES = [
    'finger-1', 'finger-2', 'finger-3', 'finger-4', 'finger-5',
    'finger-6', 'finger-7', 'finger-8', 'finger-9', 'finger-10',
    'finger-11', 'finger-12', 'finger-13', 'finger-14', 'finger-15',
    'finger-16', 'finger-17', 'finger-18', 'finger-19', 'Trapezium',
    'Trapezoid', 'Capitate', 'Hamate', 'Scaphoid', 'Lunate',
    'Triquetrum', 'Pisiform', 'Radius', 'Ulna',
]
num_classes = len(CLASSES) + 1


@METRICS.register_module()
class SubmissionMetric(BaseMetric):
    def __init__(self,
                 collect_device='cpu',
                 prefix=None,
                 save_path='',
                 **kwargs):
        super().__init__(collect_device=collect_device, prefix=prefix)
        self.save_path = save_path
        if self.save_path == '':
            self.save_path = '/data/ephemeral/home/submission'
        
        os.makedirs(self.save_path, exist_ok=True)
        self.cnt = 0
        
        # self.rles = []
        # self.filename_and_class = []
        
            
    def encode_mask_to_rle(self, mask):
        '''
        mask: numpy array binary mask 
        1 - mask 
        0 - background
        Returns encoded run length 
        '''
        pixels = mask.flatten()
        pixels = np.concatenate([[0], pixels, [0]])
        runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
        runs[1::2] -= runs[::2]
        return ' '.join(str(x) for x in runs)
    
    
    def convert_to_class_masks(self, mask):
        """
        Convert a mask of shape (1, H, W) with values in range [0, 29] to (29, H, W) masks.
        
        Parameters:
        - mask: numpy array of shape (1, H, W) with integer values from 0 to 29 representing class labels.
        
        Returns:
        - class_masks: numpy array of shape (29, H, W) where each slice corresponds to a class mask.
        """
        # Get the height and width from the input mask shape
        _, H, W = mask.shape

        # Initialize an empty array to hold the class masks (29, H, W)
        class_masks = np.zeros((29, H, W), dtype=np.uint8)

        # Iterate over each class (1 to 29)
        for class_id in range(1, 30):  # Classes 1 to 29
            class_masks[class_id - 1] = (mask[0] == class_id).astype(np.uint8)

        return class_masks
    

    def decode_rle_to_mask(self, rle, height, width):
        s = rle.split()
        starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
        starts -= 1
        ends = starts + lengths
        img = np.zeros(height * width, dtype=np.uint8)
        
        for lo, hi in zip(starts, ends):
            img[lo:hi] = 1
        
        return img.reshape(height, width)
    
    
    def process(self, data_batch, data_samples):
        for data_sample in data_samples:
            img_path = data_sample['img_path']
            img_name = os.path.basename(img_path).split('_')[1]
            pred_label = data_sample['pred_sem_seg']['data']
            img_shape = data_sample['img_shape']
            ori_shape = data_sample['ori_shape']
            
            # if img_shape != ori_shape:
            #     pred_label = F.interpolate(pred_label, size=ori_shape, mode="bilinear")
            pred_label = pred_label.cpu().numpy()
            
            
            pred_label = self.convert_to_class_masks(pred_label)
            for i, pred in enumerate(pred_label):
                rle = self.encode_mask_to_rle(pred)
                self.results.append((rle, f"{CLASSES[i]}_{img_name}"))
        
            
    def compute_metrics(self, results):
        classes, filename = zip(*[x[1].split("_") for x in self.results])
        image_name = [f for f in filename]
        
        df = pd.DataFrame({
            "image_name": image_name,
            "class": classes,
            "rle": [x[0] for x in self.results]
        })
        print(df.head(30))
        df.to_csv(os.path.join(self.save_path, 'output.csv'), index=False)
        
        return {"status": 1}