import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st

def rle2mask(rle: pd.Series, shape: tuple[int, int]) -> np.ndarray:
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

def show(info: dict, df: pd.DataFrame, images: list, anno: str='All') -> None:
    cols = st.columns(len(images))
    for i, img in enumerate(images):
        current_df = df[df['image_name'] == img]
        image_path = current_df.iloc[0]['image_path']

        image_path = os.path.join(info['image_root'].format('test'), image_path)
        image = Image.open(image_path).convert("RGB")
        mask = np.zeros((*image.size[::-1], 3), dtype=np.uint8)
        contour_mask = np.zeros((*image.size[::-1], 3), dtype=np.uint8)
        for _, row in current_df.iterrows():
            class_mask = rle2mask(row['rle'], image.size[::-1])
            class_name = row['class']
            if anno not in ('All', class_name):
                continue
            if class_name.startswith('finger'):
                color = info['color']['finger']
            elif class_name in info['color'].keys():
                color = info['color'][class_name]
            mask[class_mask == 1] = color
            contours, _ = cv2.findContours(
                image=class_mask.astype(np.uint8),
                mode=cv2.RETR_EXTERNAL,
                method=cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(
                image=contour_mask,
                contours=contours,
                contourIdx=-1,
                color=color,
                thickness=2
            )
        overlay_image = cv2.addWeighted(np.array(image), 0.5, mask, 0.5, 0)
        final_image = cv2.addWeighted(overlay_image, 1, contour_mask, 1, 0)

        with cols[i]:
            st.image(final_image, caption=img)
