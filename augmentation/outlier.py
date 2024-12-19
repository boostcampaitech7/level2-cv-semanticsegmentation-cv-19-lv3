import os
import json
import shutil

"""
ID363 오른손 약지 반지 삭제
ID487 왼손목 물체 삭제
ID058 이미지 annotation 수정
ID089 이미지 annotation 수정 or 삭제
"""

def delete_folder(folder_path):
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
        print(f"Delete '{folder_path}'")
    else:
        print(f"No Exist '{folder_path}'")

def change_label_ID058():
    """
    finger1 <-> finger3
    Trapezium <-> Hamate
    Trapezoid <-> Capitate
    Pisiform -> Scaphoid
    Scaphoid -> Triquetrum
    Triquetrum -> Lundate
    Lundate -> Pisiform
    Radius <-> Ulna
    """
    file_path = "/data/ephemeral/home/data_v1/train/outputs_json/ID058/image1661392103627.json"

    with open(file_path, 'r') as file:
        data = json.load(file)
    
    for annotation in data.get('annotations', []):
        if annotation['label'] == 'finger1':
            annotation['label'] = 'finger3_'
        if annotation['label'] == 'finger3':
            annotation['label'] = 'finger1_'
        if annotation['label'] == 'Trapezium':
            annotation['label'] = 'Hamate_'
        if annotation['label'] == 'Hamate':
            annotation['label'] = 'Trapezium_'
        if annotation['label'] == 'Trapezoid':
            annotation['label'] = 'Capitate_'
        if annotation['label'] == 'Capitate':
            annotation['label'] = 'Trapezoid_'
        if annotation['label'] == 'Pisiform':
            annotation['label'] = 'Scaphoid_'
        if annotation['label'] == 'Scaphoid':
            annotation['label'] = 'Triquetrum_'
        if annotation['label'] == 'Triquetrum':
            annotation['label'] = 'Lunate_'
        if annotation['label'] == 'Lunate':
            annotation['label'] = 'Pisiform_'
        if annotation['label'] == 'Radius':
            annotation['label'] = 'Ulna_'
        if annotation['label'] == 'Ulna':
            annotation['label'] = 'Radius_'
    
    for annotation in data.get('annotations', []):
        if annotation['label'][-1] == '_':
            annotation['label'] = annotation['label'][:-1]

    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)

    print(f"Edit '{file_path}'")

def change_label_ID089():
    """
    Trapezium <-> Trapezoid
    """
    file_path = "/data/ephemeral/home/data_v1/train/outputs_json/ID089/image1661821711879.json"
    with open(file_path, 'r') as file:
        data = json.load(file)
    
    for annotation in data.get('annotations', []):
        if annotation['label'] == 'Trapezium':
            annotation['label'] = 'Trapezoid_'
        if annotation['label'] == 'Trapezoid':
            annotation['label'] = 'Trapezium_'
    
    for annotation in data.get('annotations', []):
        if annotation['label'][-1] == '_':
            annotation['label'] = annotation['label'][:-1]

    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)

    print(f"Edit '{file_path}'")

if __name__ == "__main__":
    # ID363 오른손 약지 반지 삭제
    delete_folder("/data/ephemeral/home/data_v1/train/DCM/ID363")
    delete_folder("/data/ephemeral/home/data_v1/train/outputs_json/ID363")

    # ID487 왼손목 물체 삭제
    delete_folder("/data/ephemeral/home/data_v1/train/DCM/ID487")
    delete_folder("/data/ephemeral/home/data_v1/train/outputs_json/ID487")

    # ID058 이미지 annotation 수정
    change_label_ID058()

    # ID089 이미지 annotation 수정 or 삭제 or 그대로두기
    # change_label_ID089()
    # delete_folder("/data/ephemeral/home/data_v1/train/DCM/ID089")
    # delete_folder("/data/ephemeral/home/data_v1/train/outputs_json/ID089")
    