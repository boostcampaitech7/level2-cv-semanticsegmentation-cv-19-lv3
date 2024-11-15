# python native
import os
import random
import datetime
import numpy as np
from tqdm.auto import tqdm
import albumentations as A
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models
from dataset import XRayDataset
from model import ModelSelector
from argparse import ArgumentParser

def parse_args():
    parser = ArgumentParser()

    # Conventional args
    parser.add_argument('--image_root', type=str, default='/data/ephemeral/home/data/train/DCM',
                        help='Path to the root directory containing images')
    parser.add_argument('--label_root', type=str, default='/data/ephemeral/home/data/train/outputs_json',
                        help='Path to the root directory containing labels')
    parser.add_argument('--save_dir', type=str, default="/data/ephemeral/home/data/result",
                        help='Path to the root directory containing save direction')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--max_epoch', type=int, default=1)
    parser.add_argument('--val_every', type=int, default=5)
    parser.add_argument('--random_seed', type=int, default=2024)
    parser.add_argument('--model_type', type=str, default='smp')
    parser.add_argument('--model_name', type=str, default='efficientnet-b0')
    parser.add_argument('--encoder_weights', type=str, default='imagenet')
    # parser.add_argument('--pretrained', type=str, default='True')
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

CLASS2IND = {v: i for i, v in enumerate(CLASSES)}
IND2CLASS = {v: k for k, v in CLASS2IND.items()}

def dice_coef(y_true, y_pred):
    y_true_f = y_true.flatten(2)
    y_pred_f = y_pred.flatten(2)
    intersection = torch.sum(y_true_f * y_pred_f, -1)
    eps = 0.0001
    return (2. * intersection + eps) / (torch.sum(y_true_f, -1) + torch.sum(y_pred_f, -1) + eps)

def set_seed(random_seed):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)

def save_model(model, save_dir, file_name='fcn_resnet50_best_model.pt'):
    output_path = os.path.join(save_dir, file_name)
    torch.save(model, output_path)

def validation(epoch, model, data_loader, criterion, model_type, thr=0.5):
    print(f'Start validation #{epoch:2d}')
    model.eval()

    dices = []
    with torch.no_grad():
        n_class = len(CLASSES)
        total_loss = 0
        cnt = 0

        for step, (images, masks) in tqdm(enumerate(data_loader), total=len(data_loader)):
            images, masks = images.cuda(), masks.cuda()         
            model = model.cuda()
            
            if model_type == 'torchvision':
                outputs = model(images)['out']
            elif model_type == 'smp':
                outputs = model(images)
            
            output_h, output_w = outputs.size(-2), outputs.size(-1)
            mask_h, mask_w = masks.size(-2), masks.size(-1)
            
            # gt와 prediction의 크기가 다른 경우 prediction을 gt에 맞춰 interpolation 합니다.
            if output_h != mask_h or output_w != mask_w:
                outputs = F.interpolate(outputs, size=(mask_h, mask_w), mode="bilinear")
            
            loss = criterion(outputs, masks)
            total_loss += loss
            cnt += 1
            
            outputs = torch.sigmoid(outputs)
            outputs = (outputs > thr).detach().cpu()
            masks = masks.detach().cpu()
            
            dice = dice_coef(outputs, masks)
            dices.append(dice)
                
    dices = torch.cat(dices, 0)
    dices_per_class = torch.mean(dices, 0)
    dice_str = [
        f"{c:<12}: {d.item():.4f}"
        for c, d in zip(CLASSES, dices_per_class)
    ]
    dice_str = "\n".join(dice_str) 
    print(dice_str)
    
    avg_dice = torch.mean(dices_per_class).item()
    
    return avg_dice

def train(model, train_loader, valid_loader, criterion, optimizer, save_dir, random_seed, max_epoch, val_every, model_type):
    print(f'Start training..')
    
    n_class = len(CLASSES)
    best_dice = 0.

    # GradScaler를 사용해 Mixed Precision Training을 설정
    scaler = torch.cuda.amp.GradScaler()
    
    for epoch in range(max_epoch):
        model.train()

        for step, (images, masks) in enumerate(train_loader):            
            # gpu 연산을 위해 device 할당합니다.
            images, masks = images.cuda(), masks.cuda()
            model = model.cuda()
            
            # if model_type == 'torchvision':
            #     outputs = model(images)['out']
            # elif model_type == 'smp':
            #     outputs = model(images)
            
            # loss를 계산합니다.
            # loss = criterion(outputs, masks)
            # optimizer.zero_grad()
            # loss.backward()
            # optimizer.step()

            # Mixed Precision Training 적용
            with torch.cuda.amp.autocast():
                if model_type == 'torchvision':
                    outputs = model(images)['out']
                elif model_type == 'smp':
                    outputs = model(images)
                loss = criterion(outputs, masks)
            
            # 스케일된 loss를 사용해 backward 및 optimizer step
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            # step 주기에 따라 loss를 출력합니다.
            if (step + 1) % 25 == 0:
                print(
                    f'{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")} | '
                    f'Epoch [{epoch+1}/{max_epoch}], '
                    f'Step [{step+1}/{len(train_loader)}], '
                    f'Loss: {round(loss.item(),4)}'
                )
                
        # validation 주기에 따라 loss를 출력하고 best model을 저장합니다.
        if (epoch + 1) % val_every == 0:
            # 캐시된 메모리를 해제하여 PyTorch의 메모리 누수를 방지
            torch.cuda.empty_cache()
            dice = validation(epoch + 1, model, valid_loader, criterion, model_type)
            
            if best_dice < dice:
                print(f"Best performance at epoch: {epoch + 1}, {best_dice:.4f} -> {dice:.4f}")
                print(f"Save model in {save_dir}")
                best_dice = dice
                save_model(model, save_dir)

def do_training(image_root, label_root, save_dir, batch_size, learning_rate, max_epoch, val_every, random_seed,
                model_type, model_name, encoder_weights):
    pngs = {
        os.path.relpath(os.path.join(root, fname), start=image_root)
        for root, _dirs, files in os.walk(image_root)
        for fname in files
        if os.path.splitext(fname)[1].lower() == ".png"
    }

    jsons = {
        os.path.relpath(os.path.join(root, fname), start=label_root)
        for root, _dirs, files in os.walk(label_root)
        for fname in files
        if os.path.splitext(fname)[1].lower() == ".json"
    }
    
    pngs = sorted(pngs)
    jsons = sorted(jsons)
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    tf = A.Resize(512, 512)
    '''
    ************************************** augmentatoin도 모듈화 할 것 **************************************
    train_tf = A.Compose([
        A.Resize(512, 512),      
        A.HorizontalFlip(p=0.3),
    ])
    '''
    
    train_dataset = XRayDataset(pngs, jsons, CLASS2IND, CLASSES, image_root, label_root, is_train=True, transforms=tf)
    valid_dataset = XRayDataset(pngs, jsons, CLASS2IND, CLASSES, image_root, label_root, is_train=False, transforms=tf)
    
    train_loader = DataLoader(
        dataset=train_dataset, 
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
        drop_last=True,
    )
    
    valid_loader = DataLoader(
        dataset=valid_dataset, 
        batch_size=8,
        shuffle=False,
        num_workers=0,
        drop_last=False
    )
    
    model_selector = ModelSelector(
        model_type=model_type,
        num_classes=len(CLASSES),
        model_name=model_name,
        encoder_weights=encoder_weights,
    )
    model = model_selector.get_model()
    
    # Loss function을 정의합니다.
    criterion = nn.BCEWithLogitsLoss()

    # Optimizer를 정의합니다.
    optimizer = optim.Adam(params=model.parameters(), lr=learning_rate, weight_decay=1e-6)

    # 시드를 설정합니다.
    set_seed(random_seed)
    
    train(model, train_loader, valid_loader, criterion, optimizer, save_dir, random_seed, max_epoch, val_every, model_type)

def main(args):
    do_training(**args.__dict__)

if __name__ == '__main__':
    args = parse_args()
    main(args)