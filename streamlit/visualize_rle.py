import os
import cv2
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
        # RLE가 예상치 못한 형식일 경우 빈 마스크 반환
        return np.zeros(shape, dtype=np.uint8)

def show(info, df, image_name, anno='All'):
    cols = st.columns(len(image_name))
    for i in range(len(image_name)):
        current_df = df[df['image_name'] == image_name[i]]
        image_path = current_df.iloc[0]['image_path']

        image_path = os.path.join(info['image_root'].format('test'), image_path)
        image = Image.open(image_path).convert("RGB")
        
        # Create a mask for the segmentation
        mask = np.zeros((*image.size[::-1], 3), dtype=np.uint8)
        contour_mask = np.zeros((*image.size[::-1], 3), dtype=np.uint8)
        
        for _, row in current_df.iterrows():
            class_mask = rle2mask(row['rle'], image.size[::-1])
            class_name = row['class']
            if anno != 'All' and anno != class_name:
                continue
            # 색상 선택 로직
            if class_name.startswith('finger'):
                color = info['color']['finger']
            elif class_name in ['Trapezoid', 'Pisiform', 'Radius', 'Ulna', 'Trapezium', 'Capitate', 'Hamate', 'Scaphoid', 'Lunate', 'Triquetrum']:
                color = info['color'][class_name]  # 특정 클래스의 경우 개별 색상 적용

            mask[class_mask == 1] = color
            
            # Find contours
            contours, _ = cv2.findContours(class_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Draw contours on the contour mask
            cv2.drawContours(contour_mask, contours, -1, color, 2)
            
        # Overlay the mask on the original image
        overlay_image = cv2.addWeighted(np.array(image), 0.5, mask, 0.5, 0)

        # Add contours to the overlayed image
        final_image = cv2.addWeighted(overlay_image, 1, contour_mask, 1, 0)

        # Display the overlayed image
        with cols[i]:
            st.image(final_image, caption=image_name[i], use_container_width=True)

