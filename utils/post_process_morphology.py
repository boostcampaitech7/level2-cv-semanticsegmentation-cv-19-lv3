import numpy as np
import os
import cv2
import pandas as pd
from tqdm import tqdm

########################## settings ########################### 
CSV_PATH = "/data/ephemeral/home/post_process/sam2_double_vote3.csv"
TARGET_PATH = "/data/ephemeral/home/post_process/new_output.csv"
CLASSES = [
    'finger-1', 'finger-2', 'finger-3', 'finger-4', 'finger-5',
    'finger-6', 'finger-7', 'finger-8', 'finger-9', 'finger-10',
    'finger-11', 'finger-12', 'finger-13', 'finger-14', 'finger-15',
    'finger-16', 'finger-17', 'finger-18', 'finger-19', 'Trapezium',
    'Trapezoid', 'Capitate', 'Hamate', 'Scaphoid', 'Lunate',
    'Triquetrum', 'Pisiform', 'Radius', 'Ulna',
]
TARGETS = ['finger-8', 'finger-3', 'finger-4', 'finger-16']
########################################################################################################


num_classes = len(CLASSES)
CLASS2IND = {v: i for i, v in enumerate(CLASSES)}
IND2CLASS = {v: k for k, v in CLASS2IND.items()}


def rle2mask(rle, shape):
    if pd.isna(rle):
        return np.zeros(shape, dtype=np.uint8)
    try:
        s = np.array(rle.split(), dtype=int)
        starts, lengths = s[0::2] - 1, s[1::2]
        ends = starts + lengths
        img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
        for lo, hi in zip(starts, ends):
            img[lo:hi] = 1
        return img.reshape(shape)
    except AttributeError:
        # RLE가 예상치 못한 형식일 경우 빈 마스크 반환
        return np.zeros(shape, dtype=np.uint8)

def encode_mask_to_rle(mask):
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

def main():
    df = pd.read_csv(CSV_PATH)
    for idx, row in tqdm(df.iterrows()):
        cls_name = row['class']
        
        if cls_name not in TARGETS:
            continue
        
        mask = rle2mask(row['rle'], (2048, 2048))
        # 구조화 요소 커널, 사각형 (5x5) 생성 ---①
        
        if cls_name == 'Ulna':
            kernel_size = (30, 30)
        else:
            kernel_size = (10, 10)
        k = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
        # 열림 연산 적용 ---②
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
        # # 닫힘 연산 적용 ---③
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
        
        new_rle = encode_mask_to_rle(mask)
        df.loc[idx, 'rle'] = new_rle
    
    df.to_csv(TARGET_PATH)
    
if __name__ == "__main__":
    main()