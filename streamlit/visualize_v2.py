import os
import json
import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

COLOR = {
    'finger': (120,203,228),
    'Trapezoid': (145,42,177),
    'Pisiform': (145,42,177),
    'Radius': (210,71,77),
    'Ulna': (210,71,77),
    'wrist': (193,223,159)
}

def get_poly(points, label):
    if label.startswith('finger'):
        label = 'finger'
    elif label not in COLOR:
        label = 'wrist'
        
    poly = patches.Polygon(
        points, 
        closed=True, 
        facecolor=[ck/255 for ck in COLOR[label]], 
        edgecolor='black',
        alpha=0.7
    )
    return poly

def show(info, images, labels, id, text='None', anno='All'):
    cols = st.columns(2)

    image_path_right = os.path.join(info['image_root'], images[id])
    label_path_right = os.path.join(info['label_root'], labels[id])

    image_path_left = os.path.join(info['image_root'], images[id + 1])
    label_path_left = os.path.join(info['label_root'], labels[id + 1])
        
    image_right = Image.open(image_path_right).convert("RGB")
    label_right = json.load(open(label_path_right))

    image_left = Image.open(image_path_left).convert("RGB")
    label_left = json.load(open(label_path_left))
    
    fig_left, ax_left = plt.subplots()
    
    ax_left.imshow(image_left)
    ax_left.set_title(images[id + 1])
    ax_left.axis('off')
    for annot in label_left['annotations']:
        points = [tuple(pts) for pts in annot['points']]
        orin_label = annot['label'] 
        label = orin_label
        if anno == 'All':
            poly = get_poly(points, label)
            ax_left.add_patch(poly)
            cx, cy = sum([p[0] for p in points]) / len(points), sum([p[1] for p in points]) / len(points)
            if text == 'Text':
                ax_left.text(cx, cy, orin_label, fontsize=5, color='white')
        else:
            if label == anno:
                poly = get_poly(points, label)
                ax_left.add_patch(poly)
                cx, cy = sum([p[0] for p in points]) / len(points), sum([p[1] for p in points]) / len(points)
                if text == 'Text':
                    ax_left.text(cx, cy, orin_label, fontsize=5, color='white')
    with cols[0]:
        st.pyplot(fig_left)

    fig_right, ax_right = plt.subplots()

    ax_right.imshow(image_right)
    ax_right.set_title(images[id])
    ax_right.axis('off')
    for annot in label_right['annotations']:
        points = [tuple(pts) for pts in annot['points']]
        orin_label = annot['label'] 
        label = orin_label
        if anno == 'All':
            poly = get_poly(points, label)
            ax_right.add_patch(poly)
            cx, cy = sum([p[0] for p in points]) / len(points), sum([p[1] for p in points]) / len(points)
            if text == 'Text':
                ax_right.text(cx, cy, orin_label, fontsize=5, color='white')
        else:
            if label == anno:
                poly = get_poly(points, label)
                ax_right.add_patch(poly)
                cx, cy = sum([p[0] for p in points]) / len(points), sum([p[1] for p in points]) / len(points)
                if text == 'Text':
                    ax_right.text(cx, cy, orin_label, fontsize=5, color='white')
    with cols[1]:
        st.pyplot(fig_right)