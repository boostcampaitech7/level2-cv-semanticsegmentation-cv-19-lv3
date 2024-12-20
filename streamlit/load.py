import os
import numpy as np
import pandas as pd

def load(info, mode='train'):
    images = {
        os.path.relpath(os.path.join(root, fname), start=info['image_root'].format(mode))
        for root, _dirs, files in os.walk(info['image_root'].format(mode))
        for fname in files
        if os.path.splitext(fname)[1].lower() == ".png"
    }

    labels = {
        os.path.relpath(os.path.join(root, fname), start=info['label_root'])
        for root, _dirs, files in os.walk(info['label_root'])
        for fname in files
        if os.path.splitext(fname)[1].lower() == ".json"
    }

    images = sorted(images)
    labels = sorted(labels)
    images = np.array(images)
    labels = np.array(labels)

    return images, labels

def load_csv(info, csv_path, mode='train'):
    df = pd.read_csv(csv_path)
    image_path_list = {
        os.path.relpath(os.path.join(root, fname), start=info['image_root'].format(mode))
        for root, _dirs, files in os.walk(info['image_root'].format(mode))
        for fname in files
        if os.path.splitext(fname)[1].lower() == ".png"
    }
    image_path_list = sorted(image_path_list)
    if mode == 'train':
        label_path_list = {
            os.path.relpath(os.path.join(root, fname), start=info['label_root'])
            for root, _dirs, files in os.walk(info['label_root'])
            for fname in files
            if os.path.splitext(fname)[1].lower() == ".json"
        }
        label_path_list = sorted(label_path_list)

    image_paths = []
    label_paths = []

    for image_name in df['image_name']:
        image_path = None
        label_path = None
        for i in range(len(image_path_list)):
            if image_name in image_path_list[i]:
                image_path = image_path_list[i]
                if mode == 'train' and i < len(label_path_list):
                    label_path = label_path_list[i]
                break
        image_paths.append(image_path)
        if mode == 'train':
            label_paths.append(label_path)

    df['image_path'] = image_paths
    if mode == 'train':
        df['label_path'] = label_paths

    return df
