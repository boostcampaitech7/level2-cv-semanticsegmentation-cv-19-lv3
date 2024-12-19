import os
import json

import numpy as np
import cv2
from sklearn.model_selection import GroupKFold

from mmseg.registry import DATASETS, TRANSFORMS, MODELS, METRICS
from mmseg.datasets import BaseSegDataset

from mmcv.transforms import BaseTransform

# 데이터 경로를 입력하세요

CLASSES = [
    'finger-1', 'finger-2', 'finger-3', 'finger-4', 'finger-5',
    'finger-6', 'finger-7', 'finger-8', 'finger-9', 'finger-10',
    'finger-11', 'finger-12', 'finger-13', 'finger-14', 'finger-15',
    'finger-16', 'finger-17', 'finger-18', 'finger-19', 'Trapezium',
    'Trapezoid', 'Capitate', 'Hamate', 'Scaphoid', 'Lunate',
    'Triquetrum', 'Pisiform', 'Radius', 'Ulna',
]

CLASS2IND = {v: i for i, v in enumerate(CLASSES)}
IND2CLASS = {v: k for k, v in CLASS2IND.items()}
val_fold_num = 0


@DATASETS.register_module()
class XRayDataset2(BaseSegDataset):
    """ Multi Label Segmentation task를 위해 Xray Dataset을 불러오기 위한 Dataset Class
        Args:
            is_train (bool) : train mode와 val mode일 때 group fold의  fold_num을 다르게 가져가기 위한 설정값
            image_root (str) : 환자별 DCM이미지가 들어있는 directory
            label_root (str) : 환자별 Segmentation mask json 파일이 들어있는 directory
    """
    def __init__(self, 
                 is_train, 
                 image_root='/data/ephemeral/home/data/train/DCM',
                 label_root='/data/ephemeral/home/data/train/outputs_json',
                 **kwargs):
        
        self.image_root = image_root
        self.label_root = label_root
        self.is_train = is_train
        
        
        pngs = {
            os.path.relpath(os.path.join(root, fname), start=self.image_root)
            for root, _dirs, files in os.walk(self.image_root)
            for fname in files
            if os.path.splitext(fname)[1].lower() == ".png"
        }

        jsons = {
            os.path.relpath(os.path.join(root, fname), start=self.label_root)
            for root, _dirs, files in os.walk(self.label_root)
            for fname in files
            if os.path.splitext(fname)[1].lower() == ".json"
        }

        jsons_fn_prefix = {os.path.splitext(fname)[0] for fname in jsons}
        pngs_fn_prefix = {os.path.splitext(fname)[0] for fname in pngs}

        assert len(jsons_fn_prefix - pngs_fn_prefix) == 0
        assert len(pngs_fn_prefix - jsons_fn_prefix) == 0

        self.pngs = sorted(pngs)
        self.jsons = sorted(jsons)
        
        super().__init__(**kwargs)
        
    def load_data_list(self):
        _filenames = np.array(self.pngs)
        _labelnames = np.array(self.jsons)

        groups = [os.path.dirname(fname) for fname in _filenames]

        # dummy label
        ys = [0 for fname in _filenames]

        gkf = GroupKFold(n_splits=5)

        filenames = []
        labelnames = []
        for i, (x, y) in enumerate(gkf.split(_filenames, ys, groups)):
            if self.is_train:
                # if i == val_fold_num:
                #     continue

                filenames += list(_filenames[y])
                labelnames += list(_labelnames[y])

            else:
                if i != val_fold_num:
                    continue
                
                filenames = list(_filenames[y])
                labelnames = list(_labelnames[y])

                break

        data_list = []
        for i, (img_path, ann_path) in enumerate(zip(filenames, labelnames)):
            data_info = dict(
                img_path=os.path.join(self.image_root, img_path),
                seg_map_path=os.path.join(self.label_root, ann_path),
            )
            data_list.append(data_info)

        return data_list
        

@TRANSFORMS.register_module()
class LoadXRayAnnotations(BaseTransform):
    """ 
    label json file path로부터 mask를 load해오기 위한 Transform Class
    """
    def transform(self, result):
        label_path = result["seg_map_path"]

        image_size = (2048, 2048)

        # process a label of shape (H, W, NC)
        label_shape = image_size + (len(CLASSES), )
        label = np.zeros(label_shape, dtype=np.uint8)

        # read label file
        with open(label_path, "r") as f:
            annotations = json.load(f)
        annotations = annotations["annotations"]

        # iterate each class
        for ann in annotations:
            c = ann["label"]
            class_ind = CLASS2IND[c]
            points = np.array(ann["points"])

            # polygon to mask
            class_label = np.zeros(image_size, dtype=np.uint8)
            cv2.fillPoly(class_label, [points], 1)
            label[..., class_ind] = class_label

        result["gt_seg_map"] = label
        result['seg_fields'] = ['gt_seg_map']

        return result
    
@TRANSFORMS.register_module()
class TransposeAnnotations(BaseTransform):
    """ 
    segmentation map은 image가 tensor로 변환될 때 채널이 앞에 오도록 자동으로 바뀌지 않으므로, 
    이를 적용해주기 위한 Transform Class 정의
    """
    def transform(self, result):
        result["gt_seg_map"] = np.transpose(result["gt_seg_map"], (2, 0, 1))
        
        return result