import os
import numpy as np

def load(info):
    images = {
        os.path.relpath(os.path.join(root, fname), start=info['image_root'])
        for root, _dirs, files in os.walk(info['image_root'])
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