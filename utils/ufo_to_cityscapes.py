import os
import shutil
import json
from PIL import Image, ImageDraw
from sklearn.model_selection import GroupKFold
from tqdm import tqdm

# 기본 경로 설정
input_dir = '/data/ephemeral/home/data'
output_dir = '/data/ephemeral/home/cityscapes_format_xlay_kfold'
n_splits = 5  # GroupKFold의 분할 개수

CLASSES = [
    'finger-1', 'finger-2', 'finger-3', 'finger-4', 'finger-5',
    'finger-6', 'finger-7', 'finger-8', 'finger-9', 'finger-10',
    'finger-11', 'finger-12', 'finger-13', 'finger-14', 'finger-15',
    'finger-16', 'finger-17', 'finger-18', 'finger-19', 'Trapezium',
    'Trapezoid', 'Capitate', 'Hamate', 'Scaphoid', 'Lunate',
    'Triquetrum', 'Pisiform', 'Radius', 'Ulna',
]
CLASS2IND = {v: i+1 for i, v in enumerate(CLASSES)}
IND2CLASS = {v: k for k, v in CLASS2IND.items()}
PALETTE = [
    (220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230), (106, 0, 228),
    (0, 60, 100), (0, 80, 100), (0, 0, 70), (0, 0, 192), (250, 170, 30),
    (100, 170, 30), (220, 220, 0), (175, 116, 175), (250, 0, 30), (165, 42, 42),
    (255, 77, 255), (0, 226, 252), (182, 182, 255), (0, 82, 0), (120, 166, 157),
    (110, 76, 0), (174, 57, 255), (199, 100, 0), (72, 0, 118), (255, 179, 240),
    (0, 125, 92), (209, 0, 151), (188, 208, 182), (0, 220, 176),
]

# 경로 생성
def create_cityscapes_dirs():
    for split in ['train', 'val', 'test']:
        
        if split == 'test':
            os.makedirs(os.path.join(output_dir, 'leftImg8bit', split), exist_ok=True)
            os.makedirs(os.path.join(output_dir, 'gtFine', split), exist_ok=True)
        
        else:
            for fold in range(1, n_splits+1):
                os.makedirs(os.path.join(output_dir, 'leftImg8bit', split, f"fold_{fold}"), exist_ok=True)
                os.makedirs(os.path.join(output_dir, 'gtFine', split, f"fold_{fold}"), exist_ok=True)

# JSON 파일을 읽어 label 및 instance 마스크 생성
def create_masks(json_file, img_shape, label_map):
    with open(json_file, 'r') as f:
        annotations = json.load(f)["annotations"]
    
    label_mask = Image.new("L", img_shape, 0)
    instance_mask = Image.new("I", img_shape, 0)  # "I" 모드는 32비트 정수 이미지를 생성합니다.
    color_mask = Image.new("RGB", img_shape, (0, 0, 0))
    instance_id = 1

    for ann in annotations:
        label = ann['label']
        if label not in label_map:
            continue  # 정의된 클래스에 없는 경우 무시
        points = ann['points']
        
        draw_label = ImageDraw.Draw(label_mask)
        draw_instance = ImageDraw.Draw(instance_mask)
        draw_color = ImageDraw.Draw(color_mask)
        
        # 폴리곤으로 그리기
        label_id = label_map[label]
        polygon = [tuple(p) for p in points]
        draw_label.polygon(polygon, fill=label_map[label])
        draw_instance.polygon(polygon, fill=instance_id)
        draw_color.polygon(polygon, fill=tuple(PALETTE[label_id-1]))
        instance_id += 1
        
    return label_mask, instance_mask, color_mask

# 데이터 수집 함수
def collect_data():
    img_paths = []
    json_paths = []
    groups = []

    train_image_dir = os.path.join(input_dir, 'train', 'DCM')
    train_json_dir = os.path.join(input_dir, 'train', 'outputs_json')
    
    for id_folder in tqdm(os.listdir(train_image_dir), desc="Collecting Train Data"):
        img_folder_path = os.path.join(train_image_dir, id_folder)
        json_folder_path = os.path.join(train_json_dir, id_folder)

        if not os.path.isdir(img_folder_path):
            continue

        for img_file in os.listdir(img_folder_path):
            if img_file.endswith('.png'):
                img_path = os.path.join(img_folder_path, img_file)
                json_path = os.path.join(json_folder_path, f"{os.path.splitext(img_file)[0]}.json")
                if os.path.exists(json_path):
                    img_paths.append(img_path)
                    json_paths.append(json_path)
                    groups.append(id_folder)  # 그룹 ID로 폴더명을 사용

    return img_paths, json_paths, groups

# 데이터 분할 및 마스크 생성
def split_and_create_masks(img_paths, json_paths, groups, label_map):
    gkf = GroupKFold(n_splits=n_splits)
    
    for fold, (train_idx, val_idx) in enumerate(gkf.split(img_paths, groups=groups), start=1):
        print(f"Processing fold {fold}")

        for idx, split in zip([train_idx, val_idx], ['train', 'val']):
            for i in tqdm(idx, desc=f"Creating masks for {split} in fold {fold}"):
                img_path = img_paths[i]
                json_path = json_paths[i]
                id_folder = groups[i]
                img_filename = os.path.basename(img_path)
                
                img = Image.open(img_path)
                img_shape = img.size

                # 마스크 생성
                label_mask, instance_mask, color_mask = create_masks(json_path, img_shape, label_map)

                # 저장 경로 설정
                base_name = f"{id_folder}_{os.path.splitext(img_filename)[0]}_gtFine"
                img_output_path = os.path.join(output_dir, 'leftImg8bit', split, f"fold_{fold}", f"{id_folder}_{img_filename}")
                label_output_path = os.path.join(output_dir, 'gtFine', split, f"fold_{fold}", f"{base_name}_labelIds.png")
                instance_output_path = os.path.join(output_dir, 'gtFine', split, f"fold_{fold}", f"{base_name}_instanceIds.png")
                color_output_path = os.path.join(output_dir, 'gtFine', split, f"fold_{fold}", f"{base_name}_color.png")

                # 파일 저장
                img.save(img_output_path)
                label_mask.save(label_output_path)
                instance_mask.save(instance_output_path)
                color_mask.save(color_output_path)
                

# 테스트 데이터 복사
def copy_test_files():
    test_image_dir = os.path.join(input_dir, 'test', 'DCM')
    
    for id_folder in tqdm(os.listdir(test_image_dir), desc="Copying Test Data"):
        img_folder_path = os.path.join(test_image_dir, id_folder)
        if not os.path.isdir(img_folder_path):
            continue
        
        for img_file in os.listdir(img_folder_path):
            if img_file.endswith('.png'):
                img_path = os.path.join(img_folder_path, img_file)
                
                # Cityscapes 형식에 맞는 저장 경로 설정
                img_output_path = os.path.join(output_dir, 'leftImg8bit', 'test', f"{id_folder}_{img_file}")
                
                # 파일 복사
                shutil.copy(img_path, img_output_path)

# 전체 실행 함수
def convert_dataset():
    # 클래스 이름과 ID 매핑
    label_map = CLASS2IND

    create_cityscapes_dirs()
    
    # Train 이미지, JSON 파일 경로 및 그룹 수집
    img_paths, json_paths, groups = collect_data()
    
    # Train 데이터를 GroupKFold로 나누고 마스크 생성
    split_and_create_masks(img_paths, json_paths, groups, label_map)
    
    # Test 데이터를 Cityscapes 구조로 복사
    copy_test_files()
    
    print("Dataset conversion complete!")

# 실행
convert_dataset()
