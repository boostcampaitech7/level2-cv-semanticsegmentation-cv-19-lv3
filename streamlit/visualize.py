import os
import json
import cv2
import numpy as np
import streamlit as st
from PIL import Image

def show(info, images, labels, ids, anno='All'):
    cols = st.columns(2 * len(ids))
    mode = 'train'
    if len(labels) == 0:
        mode = 'test'

    for i in range(len(ids)):
        image_path_left = os.path.join(info['image_root'].format(mode), images[ids[i] + 1])
        image_left = Image.open(image_path_left).convert("RGB")
        image_path_right = os.path.join(info['image_root'].format(mode), images[ids[i]])
        image_right = Image.open(image_path_right).convert("RGB")
        if mode == 'test':
            with cols[2 * i]:
                st.image(image_left, caption=images[ids[i] + 1])
            with cols[2 * i + 1]:
                st.image(image_right, caption=images[ids[i] + 1])
            continue

        label_path_left = os.path.join(info['label_root'], labels[ids[i] + 1])
        label_left = json.load(open(label_path_left))
        label_path_right = os.path.join(info['label_root'], labels[ids[i]])
        label_right = json.load(open(label_path_right))

        mask_left = np.zeros((*image_left.size[::-1], 3), dtype=np.uint8)
        contour_mask_left = np.zeros((*image_left.size[::-1], 3), dtype=np.uint8)

        for annot in label_left['annotations']:
            points = np.array(annot['points'], dtype=np.int32)
            class_name = annot['label']
            if anno != 'All' and anno != class_name:
                continue
            if class_name.startswith('finger'):
                color = info['color']['finger']
            elif class_name in ['Trapezoid', 'Pisiform', 'Radius', 'Ulna', 'Trapezium', 'Capitate', 'Hamate', 'Scaphoid', 'Lunate', 'Triquetrum']:
                color = info['color'][class_name]
            
            single_channel_mask = np.zeros(image_left.size[::-1], dtype=np.uint8)
            cv2.fillPoly(single_channel_mask, [points], 1)

            mask_left[single_channel_mask == 1] = color

            contours, _ = cv2.findContours(single_channel_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(contour_mask_left, contours, -1, color, 2)
            
        overlay_image = cv2.addWeighted(np.array(image_left), 0.5, mask_left, 0.5, 0)
        final_image = cv2.addWeighted(overlay_image, 1, contour_mask_left, 1, 0)
        with cols[2 * i]:
            st.image(final_image, caption=images[ids[i] + 1])

        mask_right = np.zeros((*image_right.size[::-1], 3), dtype=np.uint8)
        contour_mask_right = np.zeros((*image_right.size[::-1], 3), dtype=np.uint8)

        for annot in label_right['annotations']:
            points = np.array(annot['points'], dtype=np.int32)
            class_name = annot['label']
            if anno != 'All' and anno != class_name:
                continue
            if class_name.startswith('finger'):
                color = info['color']['finger']
            elif class_name in ['Trapezoid', 'Pisiform', 'Radius', 'Ulna', 'Trapezium', 'Capitate', 'Hamate', 'Scaphoid', 'Lunate', 'Triquetrum']:
                color = info['color'][class_name]
            
            single_channel_mask = np.zeros(image_right.size[::-1], dtype=np.uint8)
            cv2.fillPoly(single_channel_mask, [points], 1)

            mask_right[single_channel_mask == 1] = color

            contours, _ = cv2.findContours(single_channel_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(contour_mask_right, contours, -1, color, 2)

        overlay_image = cv2.addWeighted(np.array(image_right), 0.5, mask_right, 0.5, 0)
        final_image = cv2.addWeighted(overlay_image, 1, contour_mask_right, 1, 0)
        with cols[2 * i + 1]:
            st.image(final_image, caption=images[ids[i]])