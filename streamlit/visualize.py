import os
import json
import cv2
import numpy as np
import streamlit as st
from PIL import Image

def show(info, images, labels, id, text='None', anno='All'):
    cols = st.columns(2)

    image_path_left = os.path.join(info['image_root'].format('train'), images[id + 1])
    image_left = Image.open(image_path_left).convert("RGB")
    label_path_left = os.path.join(info['label_root'], labels[id + 1])
    label_left = json.load(open(label_path_left))

    image_path_right = os.path.join(info['image_root'].format('train'), images[id])
    image_right = Image.open(image_path_right).convert("RGB")
    label_path_right = os.path.join(info['label_root'], labels[id])
    label_right = json.load(open(label_path_right))

    mask_left = np.zeros((*image_left.size[::-1], 3), dtype=np.uint8)
    for annot in label_left['annotations']:
        points = np.array(annot['points'], dtype=np.int32)
        class_name = annot['label']
        if anno != 'All' and anno != class_name:
            continue
        if class_name.startswith('finger'):
            color = info['color']['finger']
        elif class_name in ['Trapezoid', 'Pisiform', 'Radius', 'Ulna']:
            color = info['color'][class_name]
        else:
            color = info['color']['wrist']
        
        single_channel_mask = np.zeros(image_left.size[::-1], dtype=np.uint8)
        cv2.fillPoly(single_channel_mask, [points], 1)

        mask_left[single_channel_mask == 1] = color

        if text == 'Text':
            centroid_x = int(np.mean(points[:, 0]))
            centroid_y = int(np.mean(points[:, 1]))
            cv2.putText(
                mask_left,
                class_name,
                (centroid_x, centroid_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                4,
                cv2.LINE_AA  # Line type
            )
        
    overlay_image = cv2.addWeighted(np.array(image_left), 0.5, mask_left, 0.5, 0)
    with cols[0]:
        st.image(overlay_image, caption=images[id], use_container_width=True)

    mask_right = np.zeros((*image_right.size[::-1], 3), dtype=np.uint8)
    for annot in label_right['annotations']:
        points = np.array(annot['points'], dtype=np.int32)
        class_name = annot['label']
        if anno != 'All' and anno != class_name:
            continue
        if class_name.startswith('finger'):
            color = info['color']['finger']
        elif class_name in ['Trapezoid', 'Pisiform', 'Radius', 'Ulna']:
            color = info['color'][class_name]
        else:
            color = info['color']['wrist']
        
        single_channel_mask = np.zeros(image_right.size[::-1], dtype=np.uint8)
        cv2.fillPoly(single_channel_mask, [points], 1)

        mask_right[single_channel_mask == 1] = color

        if text == 'Text':
            centroid_x = int(np.mean(points[:, 0]))
            centroid_y = int(np.mean(points[:, 1]))

            cv2.putText(
                mask_right,
                class_name,
                (centroid_x, centroid_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                4,
                cv2.LINE_AA  # Line type
            )

    overlay_image = cv2.addWeighted(np.array(image_right), 0.5, mask_right, 0.5, 0)
    with cols[1]:
        st.image(overlay_image, caption=images[id], use_container_width=True)