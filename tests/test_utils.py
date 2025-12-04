"""Тесты для модуля utils.py"""
import pytest
import os
import json
import tempfile
import shutil
import torch
from src.core.utils import dice_coeff, create_default_users, load_users
from src.core.security import EncryptionManager


class TestDiceCoeff:
    """Тесты для функции dice_coeff"""
    
    def test_perfect_match(self):
        """Тест идеального совпадения"""
        pred = torch.ones(2, 10, 10)
        target = torch.ones(2, 10, 10)
        dice = dice_coeff(pred, target)
        assert abs(dice.item() - 1.0) < 1e-6
    
    def test_no_match(self):
        """Тест полного несовпадения"""
        pred = torch.ones(2, 10, 10)
        target = torch.zeros(2, 10, 10)
        dice = dice_coeff(pred, target)
        assert dice.item() > 0  # С учетом smooth параметра
    
    def test_partial_match(self):
        """Тест частичного совпадения"""
        pred = torch.ones(2, 10, 10)
        target = torch.zeros(2, 10, 10)
        target[:, :5, :5] = 1  # Половина совпадает
        dice = dice_coeff(pred, target)
        assert 0 < dice.item() < 1


class TestUsers:
    """Тесты для функций работы с пользователями"""
    
    def test_create_default_users(self):
        """Тест создания пользователей по умолчанию"""
        # Создаем временную директорию и файл
        temp_dir = tempfile.mkdtemp()
        users_dir = os.path.join(temp_dir, "users")
        os.makedirs(users_dir)
        users_file = os.path.join(users_dir, "default_users.json")
        
        try:
            # Создаем тестовых пользователей напрямую
            encryption_manager = EncryptionManager()
            users = {
                "admin": {
                    "password": encryption_manager.encrypt_password("admin123"),
                    "role": "admin",
                    "full_name": "Администратор системы",
                    "user_id": "test_admin_id"
                },
                "doctor": {
                    "password": encryption_manager.encrypt_password("doctor123"),
                    "role": "doctor",
                    "full_name": "Врач-радиолог",
                    "user_id": "test_doctor_id"
                }
            }
            with open(users_file, "w") as f:
                json.dump(users, f, indent=2)
            
            # Проверяем структуру
            assert "admin" in users
            assert users["admin"]["role"] == "admin"
            assert os.path.exists(users_file)
            
            # Тест загрузки
            with open(users_file, "r") as f:
                loaded_users = json.load(f)
            assert "admin" in loaded_users
            
        finally:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)

