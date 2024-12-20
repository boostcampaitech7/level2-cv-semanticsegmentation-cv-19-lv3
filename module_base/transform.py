import albumentations as A

class AlbumentationTransform:
    def __init__(self, is_train, resize):
        common_transform = [
            A.Resize(resize, resize),
            A.RandomBrightnessContrast(brightness_limit=(0.1, 0.1), contrast_limit=(0.1, 0.1), p=1),
            A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), always_apply=False, p=1.0),
            ]
        
        if is_train:
            self.transform = A.Compose(
                [
                    A.HorizontalFlip(0.2),
                    A.Rotate(limit=(-15, 15), p=0.5),
                ]+ common_transform)
        else:
            self.transform = A.Compose(common_transform)
    
    def __call__(self, **kwargs):
        return self.transform(**kwargs)

class TransformSelector:
    """
    이미지 변환 라이브러리를 선택하기 위한 클래스.
    """
    def __init__(self, transform_type):
        if transform_type in ["albumentation"]:
            self.transform_type = transform_type
        else:
            raise ValueError("Unknown transformation library specified.")
        
    def get_transform(self, is_train, resize):
        if self.transform_type == 'albumentation':
            transform = AlbumentationTransform(is_train, resize)
        
        return transform