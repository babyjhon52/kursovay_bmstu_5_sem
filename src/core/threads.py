import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import datetime
import logging
from torch.utils.data import DataLoader, random_split
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtWidgets import QMessageBox

try:
    import segmentation_models_pytorch as smp
except ImportError:
    import sys
    print("Ошибка: Не установлен segmentation_models_pytorch")
    print("Пожалуйста, установите его с помощью команды:")
    print("pip install segmentation-models-pytorch")
    sys.exit(1)

from src.core.config import DEVICE, val_transforms, train_transforms, mask_train_transforms, mask_val_transforms
from src.core.models import MultimodalMedicalDataset
from src.core.utils import dice_coeff

logger = logging.getLogger('medical_ai_system')


class SegmentationThread(QThread):
    """Поток для выполнения сегментации"""
    finished = pyqtSignal(object, object)
    progress = pyqtSignal(int)

    def __init__(self, model_path, image_path):
        super().__init__()
        self.model_path = model_path
        self.image_path = image_path
        self.model = None
        # Загрузка модели с корректной обработкой весов
        try:
            self.model = smp.UnetPlusPlus(
                encoder_name="efficientnet-b4",
                encoder_weights="imagenet",
                in_channels=3,
                classes=1,
                activation=None
            )
            # Загружаем веса из чекпоинта
            if os.path.exists(model_path):
                checkpoint = torch.load(model_path, map_location=DEVICE)
                # Проверяем формат чекпоинта
                if 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                else:
                    state_dict = checkpoint
                # Загружаем веса, игнорируя несоответствия (например, оптимизатор)
                self.model.load_state_dict(state_dict, strict=False)
                self.model.eval()
                logger.info(
                    f"Модель успешно загружена из чекпоинта: {model_path}")
            else:
                logger.error(f"Файл модели не существует: {model_path}")
                QMessageBox.warning(
                    None, "Ошибка", f"Файл модели не найден: {model_path}")
        except Exception as e:
            logger.error(f"Ошибка при загрузке модели для инференса: {str(e)}")
            QMessageBox.warning(None, "Ошибка загрузки модели",
                                f"Не удалось загрузить модель для сегментации: {str(e)}")

    def run(self):
        try:
            self.progress.emit(10)
            image = cv2.imread(self.image_path)
            if image is None:
                logger.error(
                    f"Ошибка: Не удалось загрузить изображение {self.image_path}")
                self.finished.emit(None, None)
                return
            # Если изображение в градациях серого, конвертируем в RGB
            if len(image.shape) == 2 or image.shape[2] == 1:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            else:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            orig_shape = image.shape
            image_resized = cv2.resize(image, (256, 256))  # [H, W, C]
            # Применяем трансформации для инференса
            augmented = val_transforms(image=image_resized)
            image_tensor = augmented['image']  # [C, H, W]
            # Добавляем размерность батча
            image_tensor = image_tensor.unsqueeze(0)  # [1, C, H, W]
            image_tensor = image_tensor.to(DEVICE)
            self.progress.emit(30)
            with torch.no_grad():
                mask_pred = self.model(image_tensor)
            self.progress.emit(70)
            mask_pred = mask_pred.squeeze().cpu().numpy()  # [H, W]
            mask_pred = (mask_pred > 0.5).astype(np.uint8) * 255
            mask_resized = cv2.resize(
                # [W, H] -> [orig_W, orig_H]
                mask_pred, (orig_shape[1], orig_shape[0]))
            self.progress.emit(90)
            # Возвращаем оригинальное изображение
            original_image_for_display = cv2.imread(self.image_path)
            if original_image_for_display is None:
                original_image_for_display = cv2.cvtColor(
                    image_resized, cv2.COLOR_RGB2BGR)
            self.finished.emit(original_image_for_display, mask_resized)
        except Exception as e:
            logger.error(f"Ошибка при сегментации: {str(e)}")
            self.finished.emit(None, None)


class TrainingThread(QThread):
    """Поток для обучения модели"""
    epoch_finished = pyqtSignal(int, float, float)
    finished = pyqtSignal(str, dict)
    progress = pyqtSignal(int)

    def __init__(self, checkpoint_path, dataset_path, epochs=10, batch_size=4, lr=0.001, model_name=None):
        super().__init__()
        self.checkpoint_path = checkpoint_path
        self.dataset_path = dataset_path
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.model_name = model_name

        # Инициализируем модель
        self.model = smp.UnetPlusPlus(
            encoder_name="efficientnet-b4",
            encoder_weights="imagenet",
            in_channels=3,
            classes=1,
            activation=None
        )
        # Если указан путь к чекпоинту, загружаем веса
        if checkpoint_path and os.path.exists(checkpoint_path):
            try:
                checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
                # Проверяем формат чекпоинта
                if 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                else:
                    state_dict = checkpoint
                # Загружаем веса
                self.model.load_state_dict(state_dict, strict=True)
                logger.info(
                    f"Модель успешно загружена из чекпоинта: {checkpoint_path}")
            except Exception as e:
                logger.error(f"Ошибка при загрузке чекпоинта: {str(e)}")
                QMessageBox.warning(None, "Ошибка загрузки модели",
                                    f"Не удалось загрузить веса модели из файла: {checkpoint_path}\nОшибка: {str(e)}")
        self.optimizer = optim.Adam(
            self.model.parameters(), lr=self.lr)
        self.criterion = nn.BCEWithLogitsLoss()

    def run(self):
        try:
            # Создаем датасет
            full_dataset = MultimodalMedicalDataset(
                root_dir=self.dataset_path,
                transform=None,
                mask_transform=None
            )
            if len(full_dataset) == 0:
                logger.error(
                    "Датасет пустой. Проверьте путь и структуру данных.")
                self.finished.emit(
                    "", {"error": "Датасет пустой. Проверьте путь и структуру данных."})
                return

            # Разделяем на обучающую и валидационную выборки
            train_size = int(0.85 * len(full_dataset))
            val_size = len(full_dataset) - train_size
            if train_size == 0 or val_size == 0:
                logger.error(
                    "Размер обучающей или валидационной выборки равен 0. Проверьте размер датасета.")
                self.finished.emit(
                    "", {"error": "Размер обучающей или валидационной выборки равен 0. Проверьте размер датасета."})
                return

            train_dataset, val_dataset = random_split(
                full_dataset, [train_size, val_size])
            # Применяем трансформации
            train_dataset.dataset.transform = train_transforms
            train_dataset.dataset.mask_transform = mask_train_transforms
            val_dataset.dataset.transform = val_transforms
            val_dataset.dataset.mask_transform = mask_val_transforms

            # Создаем DataLoader'ы
            train_loader = DataLoader(
                train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=2,
                pin_memory=True
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=2,
                pin_memory=True
            )
            self.progress.emit(20)

            best_val_dice = 0
            metrics = {
                "train_loss": [],
                "train_dice": [],
                "val_loss": [],
                "val_dice": []
            }

            for epoch in range(self.epochs):
                # Обучение
                self.model.train()
                train_loss = 0
                train_dice = 0
                for batch_idx, (images, masks) in enumerate(train_loader):
                    images = images.to(DEVICE)
                    # masks.shape = [B, H, W] dtype=torch.float32
                    masks = masks.to(DEVICE) / 255.0
                    self.optimizer.zero_grad()
                    # outputs.shape = [B, 1, H, W]
                    outputs = self.model(images)
                    # squeeze outputs to [B, H, W] for BCE
                    outputs_squeezed = outputs.squeeze(1)  # [B, H, W]
                    # [B, H, W], [B, H, W]
                    loss = self.criterion(outputs_squeezed, masks)
                    loss.backward()
                    self.optimizer.step()

                    # Вычисляем Dice коэффициент для бинарной сегментации
                    with torch.no_grad():
                        preds = torch.sigmoid(
                            outputs_squeezed) > 0.5  # [B, H, W]
                        # передаем [B, H, W], [B, H, W]
                        dice = dice_coeff(preds, masks)
                    train_loss += loss.item()
                    train_dice += dice.item()

                    progress = 20 + int((epoch * len(train_loader) + batch_idx) /
                                        (self.epochs * len(train_loader)) * 40)
                    self.progress.emit(progress)

                avg_train_loss = train_loss / len(train_loader)
                avg_train_dice = train_dice / len(train_loader)
                metrics["train_loss"].append(avg_train_loss)
                metrics["train_dice"].append(avg_train_dice)

                # Валидация
                self.model.eval()
                val_loss = 0
                val_dice = 0
                with torch.no_grad():
                    for images, masks in val_loader:
                        images = images.to(DEVICE)
                        # masks.shape = [B, H, W] dtype=torch.float32
                        masks = masks.to(DEVICE) / 255.0
                        # outputs.shape = [B, 1, H, W]
                        outputs = self.model(images)
                        outputs_squeezed = outputs.squeeze(1)  # [B, H, W]
                        # [B, H, W], [B, H, W]
                        loss = self.criterion(outputs_squeezed, masks)
                        preds = torch.sigmoid(
                            outputs_squeezed) > 0.5  # [B, H, W]
                        # передаем [B, H, W], [B, H, W]
                        dice = dice_coeff(preds, masks)
                        val_loss += loss.item()
                        val_dice += dice.item()

                avg_val_loss = val_loss / len(val_loader)
                avg_val_dice = val_dice / len(val_loader)
                metrics["val_loss"].append(avg_val_loss)
                metrics["val_dice"].append(avg_val_dice)

                self.epoch_finished.emit(epoch + 1, avg_val_loss, avg_val_dice)

                # Сохраняем лучшую модель
                if avg_val_dice > best_val_dice:
                    best_val_dice = avg_val_dice
                progress = 60 + int(epoch / self.epochs * 30)
                self.progress.emit(progress)

            self.progress.emit(90)

            # Сохраняем финальную модель
            # Используем переданное имя модели или генерируем по умолчанию
            project_root = os.path.dirname(
                os.path.dirname(os.path.dirname(__file__)))
            models_dir = os.path.join(project_root, "data/models")
            if self.model_name:
                model_filename = f"{self.model_name}.pth"
            else:
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                model_filename = f"model_{timestamp}.pth"
            model_path = os.path.join(models_dir, model_filename)
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'epoch': self.epochs,
                'best_val_dice': best_val_dice,
                'metrics': metrics
            }, model_path)
            metrics["best_dice"] = best_val_dice
            self.progress.emit(100)
            logger.info(
                f"Обучение модели завершено. Лучший Val Dice: {best_val_dice:.4f}, модель сохранена как {model_path}")
            self.finished.emit(model_path, metrics)
        except Exception as e:
            logger.error(f"Ошибка при обучении модели: {str(e)}")
            self.finished.emit("", {"error": str(e)})
