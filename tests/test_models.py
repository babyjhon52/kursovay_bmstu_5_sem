"""Тесты для модуля models.py"""
import pytest
import os
import tempfile
import shutil
import numpy as np
import cv2
import torch
from src.core.models import UNet, MultimodalMedicalDataset


class TestUNet:
    """Тесты для модели UNet"""

    def test_unet_initialization(self):
        """Тест инициализации UNet"""
        model = UNet(
            encoder_name='efficientnet-b0',
            encoder_weights='imagenet',
            in_channels=3,
            out_channels=1
        )
        assert model is not None
        assert hasattr(model, 'unet')
        assert hasattr(model, 'logit_clamp')

    def test_unet_forward(self):
        """Тест forward pass UNet"""
        model = UNet(in_channels=3, out_channels=1)
        model.eval()

        # Создаем тестовый тензор
        test_input = torch.randn(1, 3, 256, 256)

        with torch.no_grad():
            output = model(test_input)

        assert output is not None
        assert output.shape[0] == 1  # batch size
        assert output.shape[1] == 1  # channels
        assert output.shape[2] == 256  # height
        assert output.shape[3] == 256  # width


class TestMultimodalMedicalDataset:
    """Тесты для MultimodalMedicalDataset"""

    def setup_method(self):
        """Настройка перед каждым тестом"""
        self.temp_dir = tempfile.mkdtemp()
        # Создаем структуру датасета
        self.patient_dir = os.path.join(self.temp_dir, "patient_001")
        os.makedirs(self.patient_dir)

        # Создаем тестовое изображение
        test_image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        cv2.imwrite(os.path.join(self.patient_dir, "image.tif"), test_image)

        # Создаем тестовую маску
        test_mask = np.random.randint(0, 255, (256, 256), dtype=np.uint8)
        cv2.imwrite(os.path.join(self.patient_dir,
                    "image_mask.tif"), test_mask)

    def teardown_method(self):
        """Очистка после каждого теста"""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_dataset_initialization(self):
        """Тест инициализации датасета"""
        dataset = MultimodalMedicalDataset(root_dir=self.temp_dir)
        assert dataset is not None
        assert len(dataset) == 1

    def test_dataset_getitem(self):
        """Тест получения элемента датасета"""
        dataset = MultimodalMedicalDataset(root_dir=self.temp_dir)
        image, mask = dataset[0]

        assert image is not None
        assert mask is not None
        # Проверяем, что это тензоры после трансформаций или numpy массивы
        assert hasattr(image, 'shape') or isinstance(image, torch.Tensor)
        assert hasattr(mask, 'shape') or isinstance(mask, torch.Tensor)

    def test_dataset_len(self):
        """Тест длины датасета"""
        dataset = MultimodalMedicalDataset(root_dir=self.temp_dir)
        assert len(dataset) == 1
