import os
import numpy as np
import pandas as pd
import copy
from tqdm import tqdm
import argparse


# 비교할 파일들의 경로 리스트
file_paths = [
    '/data/ephemeral/home/ensemble/0.9729_vgg.csv',
    '/data/ephemeral/home/ensemble/hrnet_0.9734.csv',
    '/data/ephemeral/home/ensemble/hrnet_2048_alltrain.csv',
    '/data/ephemeral/home/ensemble/mmseg_segformer_0.9690.csv',
    '/data/ephemeral/home/ensemble/sam2unet_0.9749.csv',
    ]

CLASSES = [
    'finger-1', 'finger-2', 'finger-3', 'finger-4', 'finger-5',
    'finger-6', 'finger-7', 'finger-8', 'finger-9', 'finger-10',
    'finger-11', 'finger-12', 'finger-13', 'finger-14', 'finger-15',
    'finger-16', 'finger-17', 'finger-18', 'finger-19', 'Trapezium',
    'Trapezoid', 'Capitate', 'Hamate', 'Scaphoid', 'Lunate',
    'Triquetrum', 'Pisiform', 'Radius', 'Ulna',
]

def parse_argument():
    parser = argparse.ArgumentParser(description="19조가~ 좋아하는~ 앙상블")
    parser.add_argument('save_root', type=str, help='output.csv를 저장할 위치')
    parser.add_argument('image_root', type=str, help='image가 위치한 DCM폴더 경로')
    parser.add_argument('--threshold', type=int, help='동의를 요구하는 모델 갯수', default=2)
    args = parser.parse_args()
    return args
    

def encode_mask_to_rle(mask):
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


def main(SAVE_ROOT, IMAGE_ROOT, threshold):
    # threshold를 초과한 개수의 픽셀을 앙상블
    print(f'{threshold} 초과를 앙상블합니다.')

    # 파일들을 담을 빈 리스트
    dfs = []

    # 각 파일을 순회하면서 DataFrame 생성 및 리스트에 추가
    pbar = tqdm(desc="loading data...")
    for i, file_path in enumerate(file_paths):
        pbar.set_postfix({"iteration": i, "status": "in progress"})
        pbar.update(1)
        if os.path.exists(file_path):  # 파일이 존재하는지 확인
            df = pd.read_csv(file_path)
            dfs.append(df)
        else:
            print(f"{file_path} 파일이 존재하지 않습니다.")
    pbar.set_postfix({"status": "done"})
    pbar.close()

    pngs = {
        os.path.relpath(os.path.join(root, fname), start=IMAGE_ROOT)
        for root, _dirs, files in os.walk(IMAGE_ROOT)
        for fname in files
        if os.path.splitext(fname)[1].lower() == ".png"
    }
    
    remove_list = []
    for k in pngs:
        if k.startswith('all_pngs'):
            remove_list.append(k)
    
    for k in remove_list:
        pngs.remove(k)

    ensemble = {}
    class_dict = {}

    height = 2048
    width = 2048

    for bone in CLASSES:
        class_dict[bone] = np.zeros(height * width, dtype=np.uint8).reshape(height, width)

    for png in pngs:
        ensemble[png[6:]] = copy.deepcopy(class_dict)

    pbar = tqdm(desc="loading rle info...")
    for fold, df in enumerate(dfs):
        # 모든 행 순회
        pbar.set_postfix({"iteration": fold, "status": "in progress"})
        pbar.update(1)
        for index, row in df.iterrows():
            # 각 행에 대해 작업 수행
            if not pd.isna(row['rle']):
                mask_img = decode_rle_to_mask(row['rle'], height, width)
                ensemble[row['image_name']][row['class']] += mask_img
            else:
                print(f'{fold}fold의 {index}번에 문제 발생!')
                print(row)
    pbar.set_postfix({"status": "done"})
    pbar.close()

    pbar = tqdm(desc="앙상블을 진행중입니다!")
    for i, png in enumerate(pngs):
        pbar.set_postfix({"iteration": i, "status": "in progress"})
        pbar.update(1)
        for bone in CLASSES:
            binary_arr = np.where(ensemble[png[6:]][bone] > threshold, 1, 0)
            ensemble[png[6:]][bone] = encode_mask_to_rle(binary_arr)
    pbar.set_postfix({"status": "done"})
    pbar.close()
    # encode 과정이 오래걸립니다. (test set 기준 약 10분)

    image_name = []
    classes = []
    rles = []

    for png in pngs:
        for bone in CLASSES:
            image_name.append(png[6:])
            classes.append(bone)
            rles.append(ensemble[png[6:]][bone])

    df = pd.DataFrame({
        "image_name": image_name,
        "class": classes,
        "rle": rles,
    })

    df.to_csv(os.path.join(SAVE_ROOT, "output.csv"), index=False)


if __name__=="__main__":
    args = parse_argument()
    main(args.save_root, args.image_root, args.threshold)