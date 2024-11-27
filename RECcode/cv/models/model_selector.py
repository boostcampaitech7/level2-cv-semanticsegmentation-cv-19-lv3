from .base_model import UnetModel

class ModelSelector():
    """
    model을 새롭게 추가하기 위한 방법
        1. model 폴더 내부에 사용하고자하는 custom model 구현
        2. 구현한 Model Class를 model_selector.py 내부로 import
        3. self.model_classes에 아래와 같은 형식으로 추가
        4. yaml파일의 model_name을 설정한 key값으로 변경
    """
    def __init__(self) -> None:
        self.model_classes = {
            "Unet" : UnetModel,
        }

    def get_model(self, model_name, **model_parameter):
        return self.model_classes.get(model_name, None)(**model_parameter)