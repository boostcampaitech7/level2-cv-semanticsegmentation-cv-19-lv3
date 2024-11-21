import albumentations as A

class TransformSelector:
    def __init__(self, image_size):
        self.common_transform = [A.Resize(image_size, image_size)]

    def get_transform(self, is_train=True):
        if is_train:
            transform = A.Compose(
                [
                    A.HorizontalFlip(p=0.5)
                ]+ self.common_transform)
        else:
            transform = A.Compose(self.common_transform)

        return transform

