import os
import shutil

# 기본 디렉토리와 새 폴더 정의
base_dir = 'data/test/DCM'
new_folder = os.path.join(base_dir, 'all_pngs')

# 새 폴더가 없으면 생성
if not os.path.exists(new_folder):
    os.makedirs(new_folder)

# ID*** 폴더를 순회하며 PNG 파일을 새 폴더로 복사
for subdir, dirs, files in os.walk(base_dir):
    if os.path.basename(subdir) == 'all_pngs':
        continue  # 새 폴더는 건너뛰기
    for file in files:
        if file.endswith('.png'):
            src_path = os.path.join(subdir, file)
            dst_path = os.path.join(new_folder, file)
            shutil.copy(src_path, dst_path)
            print(f'복사됨: {src_path} -> {dst_path}')

# 결과 확인
png_files = [f for f in os.listdir(new_folder) if f.endswith('.png')]
print(f'복사된 PNG 파일 수: {len(png_files)}')