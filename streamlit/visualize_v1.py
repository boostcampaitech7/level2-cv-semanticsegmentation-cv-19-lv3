import os
import cv2
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

def label2rgb(label, palette):
    image_size = label.shape[1:] + (3, )
    image = np.zeros(image_size, dtype=np.uint8)

    for i, class_label in enumerate(label):
        image[class_label == 1] = palette[i]

    return image

def show(info, images, labels, id):
    try:
        cols = st.columns(2)
        CLASS2IND = {v: i for i, v in enumerate(info['classes'])}
        IND2CLASS = {v: k for k, v in CLASS2IND.items()}

        image_path = os.path.join(info['image_root'], images[id])
        image_left = cv2.imread(image_path)
        image_left = cv2.cvtColor(image_left, cv2.COLOR_BGR2RGB)
        image_left = image_left / 255.

        image_path = os.path.join(info['image_root'], images[id + 1])
        image_right = cv2.imread(image_path)
        image_right = cv2.cvtColor(image_right, cv2.COLOR_BGR2RGB)
        image_right = image_right / 255.

        label_path_left = os.path.join(info['label_root'], labels[id])
        label_shape = tuple(image_left.shape[:2]) + (len(info['classes']), )
        label_left = np.zeros(label_shape, dtype=np.uint8)

        label_path_right = os.path.join(info['label_root'], labels[id + 1])
        label_shape = tuple(image_right.shape[:2]) + (len(info['classes']), )
        label_right = np.zeros(label_shape, dtype=np.uint8)

        with open(label_path_left, "r") as f:
            annotations_left = json.load(f)
        annotations_left = annotations_left["annotations"]

        with open(label_path_right, "r") as f:
            annotations_right = json.load(f)
        annotations_right = annotations_right["annotations"]

        for ann in annotations_left:
            c = ann["label"]
            class_ind = CLASS2IND[c]
            points = np.array(ann["points"])

            class_label_left = np.zeros(image_left.shape[:2], dtype=np.uint8)
            cv2.fillPoly(class_label_left, [points], 1)
            label_left[..., class_ind] = class_label_left
        
        for ann in annotations_right:
            c = ann["label"]
            class_ind = CLASS2IND[c]
            points = np.array(ann["points"])
            
            class_label_right = np.zeros(image_right.shape[:2], dtype=np.uint8)
            cv2.fillPoly(class_label_right, [points], 1)
            label_right[..., class_ind] = class_label_right
        
        label_left = label_left.transpose(2, 0, 1)
        label_right = label_right.transpose(2, 0, 1)

        label_left = torch.from_numpy(label_left).float()
        label_right = torch.from_numpy(label_right).float()

        fig_left, ax_left = plt.subplots()
        ax_left.imshow(label2rgb(label_left, info['palette']))
        ax_left.set_axis_off()

        fig_right, ax_right = plt.subplots()
        ax_right.imshow(label2rgb(label_right, info['palette']))
        ax_right.set_axis_off()

        with cols[0]:
            st.pyplot(fig_left)
        with cols[1]:
            st.pyplot(fig_right)
    except Exception as e:
        st.write(e)
