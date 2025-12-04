import os
import json
import uuid
import logging
import torch
from src.core.security import EncryptionManager

logger = logging.getLogger('medical_ai_system')


def dice_coeff(pred, target, smooth=1):
    """Вычисление коэффициента Dice для оценки качества сегментации"""
    pred = pred.contiguous()
    target = target.contiguous()
    intersection = (pred * target).sum(dim=(1, 2))
    dice = (2. * intersection + smooth) / \
           (pred.sum(dim=(1, 2)) + target.sum(dim=(1, 2)) + smooth)
    return dice.mean()


def create_default_users():
    """Создание пользователей по умолчанию"""
    encryption_manager = EncryptionManager()
    users = {
        "admin": {
            "password": encryption_manager.encrypt_password("admin123"),
            "role": "admin",
            "full_name": "Администратор системы",
            "user_id": str(uuid.uuid4())
        },
        "doctor": {
            "password": encryption_manager.encrypt_password("doctor123"),
            "role": "doctor",
            "full_name": "Врач-радиолог",
            "user_id": str(uuid.uuid4())
        },
        "researcher": {
            "password": encryption_manager.encrypt_password("research123"),
            "role": "researcher",
            "full_name": "Медицинский исследователь",
            "user_id": str(uuid.uuid4())
        }
    }
    # Сохраняем пользователей в отдельную директорию
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    users_file = os.path.join(project_root, "data/users/default_users.json")
    with open(users_file, "w") as f:
        json.dump(users, f, indent=2)
    logger.info("Созданы пользователи по умолчанию")
    return users


def load_users():
    """Загрузка пользователей из файла"""
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    users_file = os.path.join(project_root, "data/users/default_users.json")
    if os.path.exists(users_file):
        with open(users_file, "r") as f:
            return json.load(f)
    else:
        return create_default_users()
