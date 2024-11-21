import albumentations as A

"""
augmentation list

A.HorizontalFlip(p=0.5)
A.Rotate(limit=(-10, 10), border_mode=cv2.BORDER_CONSTANT, value=(255, 255, 255), p=0.5)
A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), always_apply=None, p=0.5)
A.ColorJitter(brightness=(0.8, 1.2), contrast=(0.8, 1.2), saturation=(0.8, 1.2), hue=(-0.5, 0.5), p=0.5, always_apply=None)
A.GaussNoise(var_limit=None, mean=None, std_range=(0.2, 0.44), mean_range=(0.0, 0.0), per_channel=True, noise_scale_factor=1, always_apply=None, p=0.5)
A.RandomBrightnessContrast(brightness_limit=(-0.2, 0.2), contrast_limit=(-0.2, 0.2), brightness_by_max=True, ensure_safe_range=False, always_apply=None, p=0.5),
A.Morphological(scale=(2, 3), operation='dilation' or 'erosion', p=0.5, always_apply=None)
A.PixelDropout(dropout_prob=0.01, per_channel=False, drop_value=0, mask_drop_value=None, p=0.5, always_apply=None)
A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, normalization='standard', always_apply=None, p=1.0)

"""
class TransformSelector:
    def __init__(self, image_size, ):
        self.common_transform = [A.Resize(image_size, image_size)]

    def get_transform(self, is_train=True):
        if is_train:
            transform = A.Compose(
                [
                    A.HorizontalFlip(),
                    A.ColorJitter()
                ]+ self.common_transform)
        else:
            transform = A.Compose(self.common_transform)

        return transform
