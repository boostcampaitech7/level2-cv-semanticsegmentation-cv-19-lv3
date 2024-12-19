import os
import cv2
import json

CLASSES = [
    'finger-1', 'finger-2', 'finger-3', 'finger-4', 'finger-5',
    'finger-6', 'finger-7', 'finger-8', 'finger-9', 'finger-10',
    'finger-11', 'finger-12', 'finger-13', 'finger-14', 'finger-15',
    'finger-16', 'finger-17', 'finger-18', 'finger-19', 'Trapezium',
    'Trapezoid', 'Capitate', 'Hamate', 'Scaphoid', 'Lunate',
    'Triquetrum', 'Pisiform', 'Radius', 'Ulna',
]
CLASS2IND = {v: i for i, v in enumerate(CLASSES)}

def flip_annotation(label_path, image_size):
    with open(label_path, 'r') as f:
        data = json.load(f)
    
    for ann in data['annotations']:
        if ann['type'] == 'poly_seg':
            flipped_points = []
            for point in ann['points']:
                x, y = point
                flipped_x = image_size - x - 1
                flipped_points.append([flipped_x, y])
            
            ann['points'] = flipped_points
    
    return data

def save_annotation(data, output_file_path):
    with open(output_file_path, 'w') as f:
        json.dump(data, f, indent=4)

def create_offline_horizantal_flip(image_root, label_root):
    images = {
        os.path.relpath(os.path.join(root, fname), start=image_root)
        for root, _dirs, files in os.walk(image_root)
        for fname in files
        if os.path.splitext(fname)[1].lower() == ".png"
    }
    labels = {
        os.path.relpath(os.path.join(root, fname), start=label_root)
        for root, _dirs, files in os.walk(label_root)
        for fname in files
        if os.path.splitext(fname)[1].lower() == ".json"
    }
    
    images = sorted(images)
    labels = sorted(labels)

    for image_name, label_name in zip(images, labels):
        image_path = os.path.join(image_root, image_name)
        image = cv2.imread(image_path)
        flipped_image = cv2.flip(image, 1)
        flipped_image_path = os.path.join(image_root, ''.join(image_name.split('.')[:-1]) + '_flip.png')
        cv2.imwrite(flipped_image_path, flipped_image)
        print(f"save '{flipped_image_path}'")

        label_path = os.path.join(label_root, label_name)
        flipped_label = flip_annotation(label_path, image.shape[1])
        flipped_label_path = os.path.join(label_root, ''.join(label_name.split('.')[:-1]) + '_flip.json')
        save_annotation(flipped_label, flipped_label_path)
        print(f"save '{flipped_label_path}'")
        
if __name__ == "__main__":
    image_root = "/data/ephemeral/home/data_v1/train/DCM"
    label_root = "/data/ephemeral/home/data_v1/train/outputs_json"
    create_offline_horizantal_flip(image_root, label_root)