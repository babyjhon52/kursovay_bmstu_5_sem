import os
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset
try:
    import segmentation_models_pytorch as smp
except ImportError:
    import sys
    print("Ошибка: Не установлен segmentation_models_pytorch")
    print("Пожалуйста, установите его с помощью команды:")
    print("pip install segmentation-models-pytorch")
    sys.exit(1)


class UNet(nn.Module):
    """Модель UNet++ для сегментации (обертка над smp.UnetPlusPlus)"""

    def __init__(
        self,
        encoder_name='efficientnet-b4',
        encoder_weights='imagenet',
        in_channels=3,
        out_channels=1,
        logit_clamp=20.0
    ):
        super().__init__()
        self.unet = smp.UnetPlusPlus(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=out_channels,
            activation=None
        )
        self.logit_clamp = logit_clamp

    def forward(self, image):
        logits = self.unet(image)
        if self.logit_clamp is not None:
            logits = logits.clamp(min=-self.logit_clamp, max=self.logit_clamp)
        return logits


class MultimodalMedicalDataset(Dataset):
    """Датасет для медицинских изображений"""

    def __init__(self, root_dir, transform=None, mask_transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.mask_transform = mask_transform
        self.patient_dirs = [d for d in os.listdir(
            root_dir) if os.path.isdir(os.path.join(root_dir, d))]

    def __len__(self):
        return len(self.patient_dirs)

    def __getitem__(self, idx):
        patient_dir = os.path.join(self.root_dir, self.patient_dirs[idx])
        image_files = [f for f in os.listdir(patient_dir) if not f.endswith('_mask.tif') and not f.endswith('_mask.tiff') and (
            f.endswith('.tif') or f.endswith('.tiff') or f.endswith('.png') or f.endswith('.jpg') or f.endswith('.jpeg'))]
        if not image_files:
            raise ValueError(f"No image files found in {patient_dir}")
        image_file = image_files[0]  # Берем первый файл изображения
        image_path = os.path.join(patient_dir, image_file)
        # Пытаемся найти соответствующую маску
        mask_file = None
        possible_mask_names = [
            f"{os.path.splitext(image_file)[0]}_mask.tif",
            f"{os.path.splitext(image_file)[0]}_mask.tiff",
            f"{os.path.splitext(image_file)[0]}_mask.png",
            f"{os.path.splitext(image_file)[0]}_mask.jpg",
            f"{os.path.splitext(image_file)[0]}_mask.jpeg"
        ]
        for name in possible_mask_names:
            if os.path.exists(os.path.join(patient_dir, name)):
                mask_file = name
                break
        if mask_file is None:
            raise ValueError(
                f"No mask file found for {image_file} in {patient_dir}")
        mask_path = os.path.join(patient_dir, mask_file)
        # Загружаем изображение и маску
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")
        # Если изображение в градациях серого, конвертируем в RGB
        if len(image.shape) == 2 or image.shape[2] == 1:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"Failed to load mask: {mask_path}")
        # Применяем трансформации
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        if self.mask_transform:
            augmented = self.mask_transform(image=mask)
            mask = augmented['image']
        # Для маски берем только один канал
        # Убедимся, что mask имеет форму [H, W] перед возвратом
        # Проверяем, является ли mask тензором или numpy массивом
        if isinstance(mask, torch.Tensor):
            if mask.dim() == 3 and mask.shape[0] == 1:
                # Убираем размерность канала, если она есть и равна 1
                mask = mask.squeeze(0)
        elif hasattr(mask, 'shape'):
            # Для numpy массивов используем len(shape)
            if len(mask.shape) == 3 and mask.shape[0] == 1:
                mask = mask.squeeze(0)
        return image, mask
