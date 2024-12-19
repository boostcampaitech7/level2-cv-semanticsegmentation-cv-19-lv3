import numpy as np
import pandas as pd
import os

### 아래 세 개의 경로 맞는지 확인해주세요 !!! ###
excel_file = "meta_data.xlsx"
test_dir = "data/ephemeral/home/data/test/DCM/"
train_dir = "data/ephemeral/home/data/train/DCM/"

df = pd.read_excel(excel_file)
counts = len(df)

columns_to_initialize = {
    "train": 1,         # 기본값 1
    "rotate": 0,        # 기본값 0
    "image_path": None  # 기본값 None
}

# 열 추가 또는 초기화
for i, (column, default_value) in enumerate(columns_to_initialize.items()):
    if column not in df.columns:
        # 열이 없으면 삽입
        df.insert(4 + i + 1, column, default_value)
    else:
        # 열이 있으면 초기화
        df[column] = default_value

# test 데이터 ID*** 폴더 목록 가져오기
existing_folders = [f for f in os.listdir(test_dir) if f.startswith("ID") and f[2:].isdigit()]
# ID274~ID319 + ID321
ids_to_update = list(range(274, 320)) + [321]

# 폴더 이름에서 숫자 부분 추출 및 -1 인덱스 처리
indices_to_update_train = [int(folder[2:]) - 1 for folder in existing_folders]
# -1 인덱스 처리
indices_to_update_rotate = [id_num - 1 for id_num in ids_to_update]

# train 컬럼 값을 0으로 변경
df.loc[indices_to_update_train, "train"] = 0
# rotate 컬럼 값을 1로 변경
df.loc[indices_to_update_rotate, "rotate"] = 1

def get_image_paths(row):
    folder_id = f"ID{row.name + 1:03d}"  # 인덱스 + 1을 ID*** 형식으로 변환
    base_dir = test_dir if row["train"] == 0 else train_dir  # 디렉토리 선택
    folder_path = os.path.join(base_dir, folder_id)
    try:
        images = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(".png")]
        right_image = images[0]
        left_image = images[1]
        return [left_image, right_image]
    except FileNotFoundError:  # 제외된 메타데이터일 때
        return [None, None]
    
# image_path 업데이트
df["image_path"] = df.apply(get_image_paths, axis=1)

# 변경된 데이터프레임 저장
output_path = "update_meta_data.xlsx"
df.to_excel(output_path, index=False)

print("작업 완료!")