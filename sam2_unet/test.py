import argparse
import os
import torch
import imageio
import numpy as np
import pandas as pd
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from SAM2UNet import SAM2UNet
from utils.dataset import TestDataset


parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str, required=True,
                help="path to the checkpoint of sam2-unet")
parser.add_argument("--test_image_path", type=str, required=True, 
                    help="path to the image files for testing")
parser.add_argument("--test_gt_path", type=str, required=True,
                    help="path to the mask files for testing")
parser.add_argument("--save_path", type=str, required=True,
                    help="path to save the predicted masks")
# args = parser.parse_args()

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
    CLASSES = [
        'finger-1', 'finger-2', 'finger-3', 'finger-4', 'finger-5',
        'finger-6', 'finger-7', 'finger-8', 'finger-9', 'finger-10',
        'finger-11', 'finger-12', 'finger-13', 'finger-14', 'finger-15',
        'finger-16', 'finger-17', 'finger-18', 'finger-19', 'Trapezium',
        'Trapezoid', 'Capitate', 'Hamate', 'Scaphoid', 'Lunate',
        'Triquetrum', 'Pisiform', 'Radius', 'Ulna',
    ]
    CLASS2IND = {v: i for i, v in enumerate(CLASSES)}
    IND2CLASS = {v: k for k, v in CLASS2IND.items()}

    test_dataset = TestDataset(args.test_image_path, 512)
    test_loader = DataLoader(
        dataset=test_dataset, 
        batch_size=2,
        shuffle=False,
        num_workers=2,
        drop_last=False
    )
    model.eval()
    model.cuda()
    os.makedirs(args.save_path, exist_ok=True)

    rles = []
    filename_and_class = []
    with torch.no_grad():
        n_class = len(CLASSES)

        for step, (images, image_names) in tqdm(enumerate(test_loader), total=len(test_loader)):
            images = images.cuda()    
            outputs, _, _ = model(images)
            
            outputs = F.interpolate(outputs, size=(2048, 2048), mode="bilinear")
            outputs = torch.sigmoid(outputs)
            outputs = (outputs > thr).detach().cpu().numpy()
            
            for output, image_name in zip(outputs, image_names):
                for c, segm in enumerate(output):
                    rle = encode_mask_to_rle(segm)
                    rles.append(rle)
                    filename_and_class.append(f"{IND2CLASS[c]}_{image_name}")
                    
    return rles, filename_and_class

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SAM2UNet().to(device)
    model.load_state_dict(torch.load(args.checkpoint), strict=True)
    rles, filename_and_class = test(model, args)
    classes, filename = zip(*[x.split("_") for x in filename_and_class])
    image_name = [os.path.basename(f) for f in filename]
    df = pd.DataFrame({
        "image_name": image_name,
        "class": classes,
        "rle": rles,
    })
    df.to_csv("output1.csv", index=False)

if __name__ == "__main__":
    args = argparse.Namespace(
        checkpoint="/data/ephemeral/home/jongmin/level2-cv-semanticsegmentation-cv-19-lv3/sam2_unet/checkpoints/SAM2-UNet-30.pth", 
        test_image_path="/data/ephemeral/home/data/test/DCM", 
        test_gt_path="/data/ephemeral/home/data/test/DCM",
        save_path="./outputs", 
    )
    main(args)

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# test_loader = TestDataset(args.test_image_path, args.test_gt_path, 352)
# model = SAM2UNet().to(device)
# model.load_state_dict(torch.load(args.checkpoint), strict=True)
# model.eval()
# model.cuda()
# os.makedirs(args.save_path, exist_ok=True)
# for i in range(test_loader.size):
#     with torch.no_grad():
#         image, gt, name = test_loader.load_data()
#         gt = np.asarray(gt, np.float32)
#         image = image.to(device)
#         res, _, _ = model(image)
#         # fix: duplicate sigmoid
#         # res = torch.sigmoid(res)
#         res = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
#         res = res.sigmoid().data.cpu()
#         res = res.numpy().squeeze()
#         res = (res - res.min()) / (res.max() - res.min() + 1e-8)
#         res = (res * 255).astype(np.uint8)
#         print("Saving " + name)
#         imageio.imsave(os.path.join(args.save_path, name[:-4] + ".png"), res)
