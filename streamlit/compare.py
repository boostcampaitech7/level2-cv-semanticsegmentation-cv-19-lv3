import os
import cv2
import json
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

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
        return np.zeros(shape, dtype=np.uint8)

def show(info, df, image_name, anno="All"):
    cols = st.columns(2)
    cols[0].markdown("<h4 style='text-align: center;'>Ground Truth</h4>", unsafe_allow_html=True)
    cols[1].markdown("<h4 style='text-align: center;'>Prediction</h4>", unsafe_allow_html=True)

    df = df[df['image_name'] == image_name]
    image_path = df.iloc[0]['image_path']
    label_path = df.iloc[0]['label_path']

    image_path = os.path.join(info['image_root'].format('train'), image_path)
    image = Image.open(image_path).convert("RGB")

    label_path = os.path.join(info['label_root'], label_path)
    label = json.load(open(label_path))
    
    mask_gt = np.zeros((*image.size[::-1], 3), dtype=np.uint8)
    contour_mask_gt = np.zeros((*image.size[::-1], 3), dtype=np.uint8)

    for annot in label['annotations']:
        points = np.array(annot['points'], dtype=np.int32)
        class_name = annot['label']
        if anno != 'All' and anno != class_name:
            continue
        if class_name.startswith('finger'):
            color = info['color']['finger']
        elif class_name in ['Trapezoid', 'Pisiform', 'Radius', 'Ulna', 'Trapezium', 'Capitate', 'Hamate', 'Scaphoid', 'Lunate', 'Triquetrum']:
            color = info['color'][class_name]
        
        single_channel_mask = np.zeros(image.size[::-1], dtype=np.uint8)
        cv2.fillPoly(single_channel_mask, [points], 1)

        mask_gt[single_channel_mask == 1] = color

        contours, _ = cv2.findContours(single_channel_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(contour_mask_gt, contours, -1, color, 2)
            
    overlay_image = cv2.addWeighted(np.array(image), 0.5, mask_gt, 0.5, 0)
    final_image = cv2.addWeighted(overlay_image, 1, contour_mask_gt, 1, 0)
    with cols[0]:
        st.image(final_image, caption=image, use_container_width=True)

    mask = np.zeros((*image.size[::-1], 3), dtype=np.uint8)
    contour_mask = np.zeros((*image.size[::-1], 3), dtype=np.uint8)
    
    for _, row in df.iterrows():
        class_mask = rle2mask(row['rle'], image.size[::-1])
        class_name = row['class']

        if class_name.startswith('finger'):
            color = info['color']['finger']
        elif class_name in ['Trapezoid', 'Pisiform', 'Radius', 'Ulna', 'Trapezium', 'Capitate', 'Hamate', 'Scaphoid', 'Lunate', 'Triquetrum']:
            color = info['color'][class_name]

        mask[class_mask == 1] = color

        contours, _ = cv2.findContours(class_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        cv2.drawContours(contour_mask, contours, -1, color, 2)

    overlay_image = cv2.addWeighted(np.array(image), 0.5, mask, 0.5, 0)
    final_image = cv2.addWeighted(overlay_image, 1, contour_mask, 1, 0)
    with cols[1]:
        st.image(final_image, caption=image_name, use_container_width=True)

