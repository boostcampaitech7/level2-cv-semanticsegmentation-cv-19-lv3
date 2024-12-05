import os
import cv2
import json
import shutil
import argparse
import numpy as np
import pandas as pd

def decode_rle_to_polygon(rle, img_width=2048, img_height=2048):
    s = rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    mask = np.zeros(img_height * img_width, dtype=np.uint8)
    
    for lo, hi in zip(starts, ends):
        mask[lo:hi] = 1
    
    mask = mask.reshape(img_height, img_width)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    polygons = []
    for contour in contours:
        for point in contour:
            polygons.append([int(point[0][0]), int(point[0][1])])

    return polygons

def main():
    parser = argparse.ArgumentParser(description='Pseudo Labeling')
    parser.add_argument('--result', required=True, help='result csv file path')
    parser.add_argument('--src_image_path', default='/data/ephemeral/home/data_v3/test/DCM', help='from pngs directory')
    parser.add_argument('--dst_image_path', default='/data/ephemeral/home/data_v3/train/DCM', help='to pngs directory')
    parser.add_argument('--dst_label_path', default='/data/ephemeral/home/data_v3/train/outputs_json', help='to jsons directory')
    args = parser.parse_args()

    pngs = {
        os.path.relpath(os.path.join(root, fname), start=args.src_image_path)
        for root, _dirs, files in os.walk(args.src_image_path)
        for fname in files
        if os.path.splitext(fname)[1].lower() == ".png"
    }
    pngs = sorted(pngs)
    image_dict = {png.split('/')[-1]: png.split('.')[0] for png in pngs}
    
    df = pd.read_csv(args.result)

    annotations = []

    for image_name, group in df.groupby('image_name'):
        for idx, row in group.iterrows():
            class_name = row['class']
            rle = row['rle']

            points = decode_rle_to_polygon(rle)

            annotation = {
                "id": f"{idx}-{image_name}",
                "type": "poly_seg",
                "attributes": {},
                "points": points,
                "label": class_name
            }

            annotations.append(annotation)

        from_image_path = os.path.join(args.src_image_path, image_dict[image_name] + '.png')
        to_image_path = os.path.join(args.dst_image_path, image_dict[image_name] + '.png')

        os.makedirs(os.path.dirname(to_image_path), exist_ok=True)
        shutil.copy(from_image_path, to_image_path)
        print(f"save image '{to_image_path}'")

        to_label_path = os.path.join(args.dst_label_path, image_dict[image_name] + '.json')
        os.makedirs(os.path.dirname(to_label_path), exist_ok=True)
        with open(to_label_path, 'w') as json_file:
            json.dump({"annotations": annotations, "filename": image_name,}, json_file)
        print(f"save label '{to_label_path}'")

if __name__ == "__main__":
    main()