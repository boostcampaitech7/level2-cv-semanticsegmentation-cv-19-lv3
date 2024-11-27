import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from utils.dataset import *
from utils.transform import TransformSelector
from tqdm.auto import tqdm
from SAM2UNet import SAM2UNet
from omegaconf import OmegaConf
from argparse import ArgumentParser

def set_seed(RANDOM_SEED):
    torch.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed_all(RANDOM_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)

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

def test(model, args, thr=0.5):
    classes = [
        'finger-1', 'finger-2', 'finger-3', 'finger-4', 'finger-5',
        'finger-6', 'finger-7', 'finger-8', 'finger-9', 'finger-10',
        'finger-11', 'finger-12', 'finger-13', 'finger-14', 'finger-15',
        'finger-16', 'finger-17', 'finger-18', 'finger-19', 'Trapezium',
        'Trapezoid', 'Capitate', 'Hamate', 'Scaphoid', 'Lunate',
        'Triquetrum', 'Pisiform', 'Radius', 'Ulna',
    ]
    class2ind = {v: i for i, v in enumerate(classes)}
    ind2class = {v: k for k, v in class2ind.items()}


    image_root = os.path.join(args.test_data_path, 'DCM')
    pngs = get_sorted_files_by_type(image_root, 'png')

    transform_selector = TransformSelector(args.image_size)
    transform = transform_selector.get_transform(is_train=False)

    test_dataset = FullDataset(
        image_files=np.array(pngs), 
        transforms=transform)

    test_loader = DataLoader(
        dataset=test_dataset, 
        batch_size=cfg.test_batch_size,
        num_workers=cfg.test_workers,
        shuffle=False)

    rles = []
    filename_and_class = []

    model.eval()
    with torch.no_grad():
        for step, (image_names, images) in tqdm(enumerate(test_loader), total=len(test_loader)):
            images = images.cuda()    
            outputs, _, _ = model(images)
            
            outputs = F.interpolate(outputs, size=(2048, 2048), mode="bilinear")
            outputs = torch.sigmoid(outputs)
            outputs = (outputs > thr).detach().cpu().numpy()
            
            for output, image_name in zip(outputs, image_names):
                for c, segm in enumerate(output):
                    rle = encode_mask_to_rle(segm)
                    rles.append(rle)
                    filename_and_class.append(f"{ind2class[c]}_{image_name}")
                    
    return rles, filename_and_class

def main(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SAM2UNet().to(device)
    checkpoint = torch.load(os.path.join(cfg.checkpoint))
    model.load_state_dict(checkpoint['model_state_dict'])
    rles, filename_and_class = test(model, cfg)
    classes, filename = zip(*[x.split("_") for x in filename_and_class])
    image_name = [os.path.basename(f) for f in filename]
    df = pd.DataFrame({
        "image_name": image_name,
        "class": classes,
        "rle": rles,
    })
    
    os.makedirs("./outputs", exist_ok=True)
    save_file = cfg.save_file
    if len(save_file) < 5 or save_file[-4:] != '.csv':
        save_file += '.csv'
    save_path = os.path.join('./outputs', save_file)
    df.to_csv(save_path, index=False)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml')
    parser.add_argument('--checkpoint', type=str, required=True)
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        cfg = OmegaConf.load(f)
    main(cfg)
