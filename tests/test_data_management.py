"""Тесты для модуля data_management.py"""
import pytest
import os
import tempfile
import shutil
import datetime
from src.core.data_management import SegmentationHistory, BackupManager
from src.core.security import EncryptionManager


class TestSegmentationHistory:
    """Тесты для SegmentationHistory"""

    def setup_method(self):
        """Настройка перед каждым тестом"""
        self.temp_dir = tempfile.mkdtemp()
        self.history_dir = os.path.join(self.temp_dir, "segmentations")
        self.history = SegmentationHistory(history_dir=self.history_dir)
        self.test_user_id = "test_user_123"
        self.history.set_current_user(
            {"user_id": self.test_user_id, "role": "doctor"})

    def teardown_method(self):
        """Очистка после каждого теста"""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_save_segmentation(self):
        """Тест сохранения сегментации"""
        # Создаем тестовые файлы
        test_image = os.path.join(self.temp_dir, "test_image.tif")
        test_mask = os.path.join(self.temp_dir, "test_mask.tif")
        with open(test_image, 'w') as f:
            f.write("test")
        with open(test_mask, 'w') as f:
            f.write("test")

        model_info = {"name": "test_model", "path": "test/path"}
        metrics = {"dice": 0.9, "iou": 0.8}

        seg_id = self.history.save_segmentation(
            self.test_user_id, test_image, test_mask, model_info, metrics, confirmed=False
        )

        assert seg_id is not None
        assert os.path.exists(os.path.join(self.history_dir, seg_id))
        assert os.path.exists(os.path.join(
            self.history_dir, seg_id, "info.enc"))

    def test_get_user_history(self):
        """Тест получения истории пользователя"""
        # Создаем тестовые файлы
        test_image = os.path.join(self.temp_dir, "test_image.tif")
        test_mask = os.path.join(self.temp_dir, "test_mask.tif")
        with open(test_image, 'w') as f:
            f.write("test")
        with open(test_mask, 'w') as f:
            f.write("test")

        model_info = {"name": "test_model"}
        self.history.save_segmentation(
            self.test_user_id, test_image, test_mask, model_info
        )

        history = self.history.get_user_history(self.test_user_id)
        assert len(history) > 0
        assert history[0]["user_id"] == self.test_user_id

    def test_get_segmentation_by_id(self):
        """Тест получения сегментации по ID"""
        test_image = os.path.join(self.temp_dir, "test_image.tif")
        test_mask = os.path.join(self.temp_dir, "test_mask.tif")
        with open(test_image, 'w') as f:
            f.write("test")
        with open(test_mask, 'w') as f:
            f.write("test")

        model_info = {"name": "test_model"}
        seg_id = self.history.save_segmentation(
            self.test_user_id, test_image, test_mask, model_info
        )

        seg_info = self.history.get_segmentation_by_id(seg_id)
        assert seg_info is not None
        assert seg_info["segmentation_id"] == seg_id


class TestBackupManager:
    """Тесты для BackupManager"""

    def setup_method(self):
        """Настройка перед каждым тестом"""
        self.temp_dir = tempfile.mkdtemp()
        self.backup_dir = os.path.join(self.temp_dir, "backups")
        self.backup_manager = BackupManager(backup_dir=self.backup_dir)
        self.backup_manager.set_current_user(
            {"username": "test_admin", "role": "admin"})

    def teardown_method(self):
        """Очистка после каждого теста"""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_create_backup(self):
        """Тест создания резервной копии"""
        test_data = {"users": {"test": "data"}, "models": []}
        backup_path = self.backup_manager.create_backup(test_data)

        assert backup_path is not None
        assert os.path.exists(backup_path)
        assert backup_path.endswith(".enc")

    def test_list_backups(self):
        """Тест получения списка резервных копий"""
        test_data = {"test": "data"}
        self.backup_manager.create_backup(test_data)

        backups = self.backup_manager.list_backups()
        assert len(backups) > 0
        assert all(b.endswith(".enc") for b in backups)

    def test_load_latest_backup(self):
        """Тест загрузки последней резервной копии"""
        test_data = {"test": "data", "number": 123}
        self.backup_manager.create_backup(test_data)

        loaded_data = self.backup_manager.load_latest_backup()
        assert loaded_data is not None
        assert loaded_data == test_data
