import os
import cv2
import torch
import logging
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Настройка логирования
import os
log_file = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'system.log')
logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('medical_ai_system')


def init_directories():
    """Инициализация необходимых директорий"""
    # Получаем корневую директорию проекта
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    directories = [
        os.path.join(project_root, "data/backups"),
        os.path.join(project_root, "data/users"),
        os.path.join(project_root, "data/models"),
        os.path.join(project_root, "data/segmentations"),
        os.path.join(project_root, "data/datasets"),
        os.path.join(project_root, "data/logs"),
        os.path.join(project_root, "data/temp")
    ]
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            logger.info(f"Создана директория: {directory}")


# Инициализация директорий при импорте
init_directories()

# Трансформации для обучения изображений
train_transforms = A.Compose([
    A.Resize(256, 256),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.Affine(
        translate_percent=(-0.15, 0.15),
        scale=(0.85, 1.15),
        rotate=(-25, 25),
        shear=(-15, 15),
        fill_value=0,
        p=0.7
    ),
    A.ElasticTransform(
        alpha=1,
        sigma=50,
        alpha_affine=50,
        border_mode=cv2.BORDER_CONSTANT,
        value=0,
        p=0.3
    ),
    A.GridDistortion(
        num_steps=5,
        distort_limit=0.3,
        border_mode=cv2.BORDER_CONSTANT,
        value=0,
        p=0.2
    ),
    A.OneOf([
        A.RandomBrightnessContrast(
            brightness_limit=0.3,
            contrast_limit=0.3,
            p=1.0
        ),
        A.HueSaturationValue(
            hue_shift_limit=20,
            sat_shift_limit=30,
            val_shift_limit=20,
            p=1.0
        ),
        A.RandomGamma(
            gamma_limit=(80, 120),
            p=1.0
        ),
        A.CLAHE(clip_limit=4.0, p=1.0),
    ], p=0.6),
    A.OneOf([
        A.GaussianBlur(blur_limit=5, p=1.0),
        A.MotionBlur(blur_limit=5, p=1.0),
        A.MedianBlur(blur_limit=5, p=1.0),
        A.GaussNoise(var_limit=(10.0, 50.0), p=1.0),
    ], p=0.4),
    A.OneOf([
        A.CoarseDropout(
            max_holes=12,
            max_height=24,
            max_width=24,
            min_holes=6,
            min_height=12,
            min_width=12,
            fill_value=0,
            p=1.0
        ),
        A.GridDropout(
            unit_size_min=16,
            unit_size_max=32,
            holes_number_x=3,
            holes_number_y=3,
            p=1.0
        ),
        A.Compose([
            A.RandomCrop(height=200, width=200),
            A.Resize(256, 256)
        ], p=1.0),
    ], p=0.4),
    A.RandomShadow(
        shadow_roi=(0, 0.5, 1, 1),
        num_shadows_lower=1,
        num_shadows_upper=2,
        shadow_dimension=5,
        p=0.2
    ),
    A.OpticalDistortion(
        distort_limit=0.2,
        shift_limit=0.2,
        border_mode=cv2.BORDER_CONSTANT,
        value=0,
        p=0.2
    ),
    A.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
    ToTensorV2(),
])

# Трансформации для обучения масок
mask_train_transforms = A.Compose([
    A.Resize(256, 256, interpolation=cv2.INTER_NEAREST),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.Affine(
        translate_percent=(-0.15, 0.15),
        scale=(0.85, 1.15),
        rotate=(-25, 25),
        shear=(-15, 15),
        fill_value=0,
        p=0.7,
        interpolation=cv2.INTER_NEAREST
    ),
    A.ElasticTransform(
        alpha=1,
        sigma=50,
        alpha_affine=50,
        border_mode=cv2.BORDER_CONSTANT,
        value=0,
        p=0.3,
        interpolation=cv2.INTER_NEAREST
    ),
    A.GridDistortion(
        num_steps=5,
        distort_limit=0.3,
        border_mode=cv2.BORDER_CONSTANT,
        value=0,
        p=0.2,
        interpolation=cv2.INTER_NEAREST
    ),
    A.OneOf([
        A.CoarseDropout(
            max_holes=12,
            max_height=24,
            max_width=24,
            min_holes=6,
            min_height=12,
            min_width=12,
            fill_value=0,
            p=1.0
        ),
        A.GridDropout(
            unit_size_min=16,
            unit_size_max=32,
            holes_number_x=3,
            holes_number_y=3,
            p=1.0
        ),
        A.Compose([
            A.RandomCrop(height=200, width=200),
            A.Resize(256, 256, interpolation=cv2.INTER_NEAREST)
        ], p=1.0),
    ], p=0.4),
    A.OpticalDistortion(
        distort_limit=0.2,
        shift_limit=0.2,
        border_mode=cv2.BORDER_CONSTANT,
        value=0,
        p=0.2,
        interpolation=cv2.INTER_NEAREST
    ),
    ToTensorV2(),
])

# Трансформации для валидации изображений
val_transforms = A.Compose([
    A.Resize(256, 256),
    A.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
    ToTensorV2(),
])

# Трансформации для валидации масок
mask_val_transforms = A.Compose([
    A.Resize(256, 256, interpolation=cv2.INTER_NEAREST),
    ToTensorV2(),
])

# Устройство для вычислений (GPU или CPU)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
