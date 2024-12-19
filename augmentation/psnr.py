import os
import cv2
import math
import numpy as np
import albumentations as A

image_root = "/data/ephemeral/home/data/train/DCM"
pngs = {
    os.path.relpath(os.path.join(root, fname), start=image_root)
    for root, _dirs, files in os.walk(image_root)
    for fname in files
    if os.path.splitext(fname)[1].lower() == ".png"
}
pngs = sorted(pngs)

modes = ['NEAREST', 'LINEAR', 'CUBIC', 'AREA', 'LANCZOS4']

def compare_interpolation(image_path, image_size: int = 512, interpolation: int = 0):
    ori_image = cv2.imread(image_path)
    ori_image = ori_image.astype(np.float32) / 255.0

    original_height, original_width = ori_image.shape[:2]

    transformer = A.Resize(height=image_size, width=image_size, interpolation=interpolation, always_apply=True)
    resized_image = transformer(image=ori_image)['image']

    revert_transformer = A.Resize(height=original_height, width=original_width, interpolation=interpolation, always_apply=True)
    reverted_image = revert_transformer(image=resized_image)['image']
    
    psnr_value = psnr(ori_image, reverted_image)
    return psnr_value
    

def psnr(ori_img, con_img):
    max_pixel = 255.0
    mse = np.mean((ori_img - con_img)**2)

    if mse == 0:
        return 100
    
    psnr = 20 * math.log10(max_pixel / math.sqrt(mse))

    return psnr

if __name__ == "__main__":
    # image_size = 512
    # for i in range(len(modes)):
    #     psnrs = []
    #     interpolation = i
    #     for png in pngs:
    #         image_path = os.path.join(image_root, png)
    #         psnrs.append(compare_interpolation(image_path, image_size, interpolation))
    #     print(f"PSNR mean {modes[i]}: {sum(psnrs) / len(psnrs)}")

    same = []
    for i in range(len(pngs)):
        image1_path = os.path.join(image_root, pngs[i])
        image1 = cv2.imread(image1_path)
        print(f"{pngs[i]} start")
        for j in range(i + 1, len(pngs)):
            image2_path = os.path.join(image_root, pngs[j])
            image2 = cv2.imread(image2_path)
            if psnr(image1, image2) == 100:
                same.append((pngs[i], pngs[j]))
                print(f"same {pngs[i]} & {pngs[j]}")
    print(same)