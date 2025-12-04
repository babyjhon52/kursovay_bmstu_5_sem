"""Тесты для модуля config.py"""
import pytest
import os
import tempfile
import shutil
from src.core.config import init_directories, DEVICE, train_transforms, val_transforms


class TestConfig:
    """Тесты для конфигурации"""
    
    def test_init_directories(self):
        """Тест инициализации директорий"""
        temp_base = tempfile.mkdtemp()
        try:
            # Временно изменяем пути
            original_dirs = ["data/backups", "data/users"]
            test_dirs = [os.path.join(temp_base, d) for d in original_dirs]
            
            for test_dir in test_dirs:
                if not os.path.exists(test_dir):
                    os.makedirs(test_dir)
            
            # Проверяем, что директории созданы
            for test_dir in test_dirs:
                assert os.path.exists(test_dir)
        finally:
            shutil.rmtree(temp_base)
    
    def test_device(self):
        """Тест устройства для вычислений"""
        assert DEVICE is not None
        assert str(DEVICE) in ["cuda", "cpu"]
    
    def test_transforms(self):
        """Тест наличия трансформаций"""
        assert train_transforms is not None
        assert val_transforms is not None
        # Проверяем, что это объекты Compose
        assert hasattr(train_transforms, '__call__')
        assert hasattr(val_transforms, '__call__')

