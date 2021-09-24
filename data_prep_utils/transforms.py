import torchvision.transforms as T
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2


def get_classification_train_transforms():
    return T.Compose([
        T.ToTensor(),
        T.Normalize(mean=(0.45,), std=(0.225,)),
        # T.ColorJitter(),
        T.Resize(256),
        T.RandomHorizontalFlip(),
    ])


def get_classification_valid_transforms():
    return T.Compose([
        T.ToTensor(),
        T.Normalize(mean=(0.45,), std=(0.225,)),
        T.Resize(256),
    ])


def get_detection_train_transforms():
    return A.Compose(
        [
            A.Normalize(mean=(0.45,), std=(0.225,)),
            A.RandomBrightnessContrast(brightness_limit=0.2,
                                       contrast_limit=0.2, p=0.9),
            A.HorizontalFlip(p=0.5),
            # A.Rotate(),
            A.Resize(height=256, width=256, p=1),
            ToTensorV2(p=1.0),
        ],
        p=1.0,
        bbox_params=A.BboxParams(
            format='pascal_voc',
            min_area=0,
            min_visibility=0,
            label_fields=['labels']
        )
    )


def get_detection_valid_transforms():
    return A.Compose(
        [
            A.Normalize(mean=(0.45,), std=(0.225,)),
            A.Resize(height=256, width=256, p=1.0),
            ToTensorV2(p=1.0),
        ],
        p=1.0,
        bbox_params=A.BboxParams(
            format='pascal_voc',
            min_area=0,
            min_visibility=0,
            label_fields=['labels']
        )
    )


__all__ = [
    'get_classification_train_transforms',
    'get_classification_valid_transforms',
    'get_detection_train_transforms',
    'get_detection_valid_transforms',
]
