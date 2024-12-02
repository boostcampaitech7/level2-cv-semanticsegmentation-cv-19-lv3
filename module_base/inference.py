import os
import random
import numpy as np
import pandas as pd
import albumentations as A
import torch
from tqdm.auto import tqdm
import torch.nn.functional as F
from torch.utils.data import DataLoader
from argparse import ArgumentParser
from dataset import XRayInferenceDataset
from transform import TransformSelector

def parse_args():
    parser = ArgumentParser()
    
    parser.add_argument('--image_root', type=str, default='본인 이미지 경로 추가',
                        help='Path to the root directory containing images')
    parser.add_argument('--save_dir', type=str, default="csv 저장 경로 추가",
                        help='Path to the root directory containing save direction')
    parser.add_argument('--random_seed', type=int, default=2024)
    parser.add_argument('--model_type', type=str, default='smp')
    args = parser.parse_args()
    
    return args

CLASSES = [
    'finger-1', 'finger-2', 'finger-3', 'finger-4', 'finger-5',
    'finger-6', 'finger-7', 'finger-8', 'finger-9', 'finger-10',
    'finger-11', 'finger-12', 'finger-13', 'finger-14', 'finger-15',
    'finger-16', 'finger-17', 'finger-18', 'finger-19', 'Trapezium',
    'Trapezoid', 'Capitate', 'Hamate', 'Scaphoid', 'Lunate',
    'Triquetrum', 'Pisiform', 'Radius', 'Ulna',
]

def set_seed(random_seed):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)

def encode_mask_to_rle(mask):
    '''
    mask: numpy array binary mask 
    1 - mask 
    0 - background
    Returns encoded run length 
    '''
    pixels = mask.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

def decode_rle_to_mask(rle, height, width):
    s = rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(height * width, dtype=np.uint8)
    
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    
    return img.reshape(height, width)

def test(model, test_loader, model_type, thr=0.5):
    model = model.cuda()
    model.eval()

    rles = []
    filename_and_class = []
    with torch.no_grad():
        for step, (images, image_names) in tqdm(enumerate(test_loader), total=len(test_loader)):
            images = images.cuda()  
            if model_type == 'torchvision':
                outputs = model(images)['out']
            elif model_type == 'smp':
                outputs = model(images)  
            
            outputs = F.interpolate(outputs, size=(2048, 2048), mode="bilinear")
            outputs = torch.sigmoid(outputs)
            outputs = (outputs > thr).detach().cpu().numpy()
            
            for output, image_name in zip(outputs, image_names):
                for c, segm in enumerate(output):
                    rle = encode_mask_to_rle(segm)
                    rles.append(rle)
                    filename_and_class.append(f"{test_loader.dataset.ind2class[c]}_{image_name}")
                    
    return rles, filename_and_class

def do_inference(image_root, save_dir, random_seed, model_type):
    set_seed(random_seed)
    fnames = {
        os.path.relpath(os.path.join(root, fname), start=image_root)
        for root, _dirs, files in os.walk(image_root)
        for fname in files
        if os.path.splitext(fname)[1].lower() == ".png"
    }
    model = torch.load(os.path.join(save_dir, "best_model.pt"))
    
    test_trans = TransformSelector('albumentation')
    test_tf = test_trans.get_transform(False, 1024)
    test_dataset = XRayInferenceDataset(fnames, image_root,transforms=test_tf)
    
    test_loader = DataLoader(
        dataset=test_dataset, 
        batch_size=2,
        shuffle=False,
        num_workers=2,
        drop_last=False
    )
    
    rles, filename_and_class = test(model, test_loader, model_type)
    
    classes, filename = zip(*[x.split("_") for x in filename_and_class])
    
    image_name = [os.path.basename(f) for f in filename]
    
    df = pd.DataFrame({
        "image_name": image_name,
        "class": classes,
        "rle": rles,
    })
    
    df.to_csv("output.csv", index=False)

def main(args):
    do_inference(**args.__dict__)

if __name__ == '__main__':
    args = parse_args()
    main(args)