import os
import json
import cv2
import numpy as np
from PIL import Image
import streamlit as st

def show(info: dict, images: list, labels: list, ids: list, anno: str='All') -> None:
    cols = st.columns(2 * len(ids))
    mode = 'train'
    if len(labels) == 0:
        mode = 'test'

    for i, id in enumerate(ids):
        image_path_left = os.path.join(info['image_root'].format(mode), images[id + 1])
        image_left = Image.open(image_path_left).convert("RGB")
        image_path_right = os.path.join(info['image_root'].format(mode), images[id])
        image_right = Image.open(image_path_right).convert("RGB")
        if mode == 'test':
            with cols[2 * i]:
                st.image(image_left, caption=images[id + 1])
            with cols[2 * i + 1]:
                st.image(image_right, caption=images[id + 1])
            continue

        label_path_left = os.path.join(info['label_root'], labels[id + 1])
        with open(label_path_left, "r", encoding="utf8") as file:
            label_left = json.load(file)
        label_path_right = os.path.join(info['label_root'], labels[id])
        with open(label_path_right, "r", encoding="utf8") as file:
            label_right = json.load(file)

        mask_left = np.zeros((*image_left.size[::-1], 3), dtype=np.uint8)
        contour_mask_left = np.zeros((*image_left.size[::-1], 3), dtype=np.uint8)

        for annot in label_left['annotations']:
            points = np.array(annot['points'], dtype=np.int32)
            class_name = annot['label']
            if anno not in ('All', class_name):
                continue
            if class_name.startswith('finger'):
                color = info['color']['finger']
            elif class_name in info['color'].keys():
                color = info['color'][class_name]
            single_channel_mask = np.zeros(image_left.size[::-1], dtype=np.uint8)
            cv2.fillPoly(single_channel_mask, [points], 1)

            mask_left[single_channel_mask == 1] = color

            contours, _ = cv2.findContours(
                image=single_channel_mask.astype(np.uint8),
                mode=cv2.RETR_EXTERNAL,
                method=cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(
                image=contour_mask_left,
                contours=contours,
                contourIdx=-1,
                color=color,
                thickness=2)
        overlay_image = cv2.addWeighted(np.array(image_left), 0.5, mask_left, 0.5, 0)
        final_image = cv2.addWeighted(overlay_image, 1, contour_mask_left, 1, 0)
        with cols[2 * i]:
            st.image(final_image, caption=images[id + 1])

        mask_right = np.zeros((*image_right.size[::-1], 3), dtype=np.uint8)
        contour_mask_right = np.zeros((*image_right.size[::-1], 3), dtype=np.uint8)

        for annot in label_right['annotations']:
            points = np.array(annot['points'], dtype=np.int32)
            class_name = annot['label']
            if anno not in ('All', class_name):
                continue
            if class_name.startswith('finger'):
                color = info['color']['finger']
            elif class_name in info['color'].keys():
                color = info['color'][class_name]
            single_channel_mask = np.zeros(image_right.size[::-1], dtype=np.uint8)
            cv2.fillPoly(single_channel_mask, [points], 1)

            mask_right[single_channel_mask == 1] = color

            contours, _ = cv2.findContours(
                image=single_channel_mask.astype(np.uint8),
                mode=cv2.RETR_EXTERNAL,
                method=cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(
                image=contour_mask_right,
                contours=contours,
                contourIdx=-1,
                color=color,
                thickness=2
            )
        overlay_image = cv2.addWeighted(np.array(image_right), 0.5, mask_right, 0.5, 0)
        final_image = cv2.addWeighted(overlay_image, 1, contour_mask_right, 1, 0)
        with cols[2 * i + 1]:
            st.image(final_image, caption=images[id])
