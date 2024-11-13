# 📋 Project Overview


![project_image](https://github.com/user-attachments/assets/15a67fec-8077-492e-a517-f4a0cee5acbf)

뼈는 우리 몸의 구조와 기능에 중요한 영향을 미치기 때문에, 정확한 뼈 분할은 의료 진단 및 치료 계획을 개발하는 데 필수적입니다.

Bone Segmentation은 인공지능 분야에서 중요한 응용 분야 중 하나로, 특히, 딥러닝 기술을 이용한 뼈 Segmentation은 많은 연구가 이루어지고 있으며, 다양한 목적으로 도움을 줄 수 있습니다.

- 질병 진단의 목적으로 뼈의 형태나 위치가 변형되거나 부러지거나 골절 등이 있을 경우, 그 부위에서 발생하는 문제를 정확하게 파악하여 적절한 치료를 시행할 수 있습니다.

- 수술 계획을 세우는데 도움이 됩니다. 의사들은 뼈 구조를 분석하여 어떤 종류의 수술이 필요한지, 어떤 종류의 재료가 사용될 수 있는지 등을 결정할 수 있습니다.

- 의료장비 제작에 필요한 정보를 제공합니다. 예를 들어, 인공 관절이나 치아 임플란트를 제작할 때 뼈 구조를 분석하여 적절한 크기와 모양을 결정할 수 있습니다.

- 의료 교육에서도 활용될 수 있습니다. 의사들은 병태 및 부상에 대한 이해를 높이고 수술 계획을 개발하는 데 필요한 기술을 연습할 수 있습니다.
<br/>


- Input :

hand bone x-ray 객체가 담긴 이미지가 모델의 인풋으로 사용됩니다. segmentation annotation은 json file로 제공됩니다.

- Output :

모델은 각 클래스(29개)에 대한 확률 맵을 갖는 멀티채널 예측을 수행하고, 이를 기반으로 각 픽셀을 해당 클래스에 할당합니다.
최종적으로 예측된 결과를 Run-Length Encoding(RLE) 형식으로 변환하여 csv 파일로 제출합니다.

<br/>
<br/>

# 🗃️ Dataset

- 이미지 크기 : (2048 x 2048), 3 channel

![image](https://github.com/user-attachments/assets/7a596f2c-e7e2-415f-872a-d812a7b47825)

-  image, target 시각화 및 pixel 별로 예측해야할 29개의 classes
  
![image](https://github.com/user-attachments/assets/3474aac7-4542-4437-ad49-514a9dd72212)

<br/>
<br/>
<br/>

# 😄 Team Member

<table align="center">
    <tr align="center">
        <td><img src="https://github.com/user-attachments/assets/337d06ce-6a68-4ff9-9638-b54b2d17e9e9" width="200" height="120"></td>
        <td><img src="https://github.com/user-attachments/assets/f962dbc4-1ac0-49c1-bc1a-b999e01fa67f" width="200" height="120"></td>
        <td><img src="https://github.com/user-attachments/assets/dcd46b40-5117-437c-a8a0-8217cffcb487" width="200" height="120"></td>
        <td><img src="https://github.com/user-attachments/assets/9b936eca-2463-48d2-b01b-3196761e738e" width="200" height="120"></td>
        <td><img src="https://github.com/user-attachments/assets/4a8f05bf-9635-47f7-b90e-39bb7c6f6824" width="200" height="120"></td>
        <td><img src="https://github.com/user-attachments/assets/78c78353-ba3b-494d-ba94-429c4f838cd1" width="200" height="120"></td>
    </tr>
    <tr align="center">
        <td><a href="https://github.com/minrongtic" target="_blank">김민영</a></td>
        <td><a href="https://github.com/june21a" target="_blank">박준일</a></td>
        <td><a href="https://github.com/sejongmin" target="_blank">오종민</a></td>
        <td><a href="https://github.com/Soy17" target="_blank">이소영</a></td>
        <td><a href="https://github.com/wonjeongjeong" target="_blank">정원정</a></td>
        <td><a href="https://github.com/Yoon0717" target="_blank">한승윤</a></td>
    </tr>
    <tr align="center">
        <td>T7173</td>
        <td>T7154</td>
        <td>T7207</td>
        <td>T7222</td>
        <td>T7272</td>
        <td>T7261</td>
    </tr>
</table>

<br/>



## 🔗 Reference

### [📎 Segmentation Notion](https://www.notion.so/Hand-Bone-Image-Segmentation-13bcb8c4237680f0baeef241f0f6856b)

<br>

## Commit Convention

1. `Feature` : **새로운 기능 추가**
2. `Fix` : **버그 수정**
3. `Docs` : **문서 수정**
4. `Style` : **코드 포맷팅 → Code Convention**
5. `Refactor` : **코드 리팩토링**
6. `Test` : **테스트 코드**
7. `Comment` : **주석 추가 및 수정**

커밋할 때 헤더에 위 내용을 작성하고 전반적인 내용을 간단하게 작성합니다.

### 예시

- `git commit -m "[#issue] Feature : message content"`

커밋할 때 상세 내용을 작성해야 한다면 아래와 같이 진행합니다.

### 예시

> `git commit`  
> 어떠한 에디터로 진입하게 된 후 아래와 같이 작성합니다.  
> `[header]: 전반적인 내용`  
> . **(한 줄 비워야 함)**  
> 상세 내용

<br/>

## Branch Naming Convention

브랜치를 새롭게 만들 때, 브랜치 이름은 항상 위 `Commit Convention`의 Header와 함께 작성되어야 합니다.

### 예시

- `Feature/~~~`
- `Refactor/~~~`
