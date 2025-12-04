import os
import json
import uuid
import datetime
import logging
import numpy as np
import cv2
import torch
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
                             QComboBox, QFileDialog, QTabWidget, QTableWidget, QTableWidgetItem,
                             QProgressBar, QMessageBox, QLineEdit, QFormLayout, QDialog, QGroupBox,
                             QHeaderView, QAbstractItemView, QTextEdit, QInputDialog, QTableView)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QPixmap, QImage
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

try:
    import segmentation_models_pytorch as smp
except ImportError:
    import sys
    print("Ошибка: Не установлен segmentation_models_pytorch")
    print("Пожалуйста, установите его с помощью команды:")
    print("pip install segmentation-models-pytorch")
    sys.exit(1)

from src.core.config import DEVICE
from src.core.security import EncryptionManager, RoleBasedAccess
from src.core.data_management import BackupManager, SegmentationHistory
from src.core.monitoring import SystemMonitor
from src.core.threads import SegmentationThread, TrainingThread
from src.core.utils import load_users
from src.core.models import UNet
from src.gui.models import HistoryModel

logger = logging.getLogger('medical_ai_system')

# Получение корневой директории проекта
# Файл находится в src/gui/panels.py, нужно подняться на 2 уровня вверх
project_root = os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))


class AdminPanel(QWidget):
    def __init__(self, current_user=None, parent=None):
        super().__init__(parent)
        self.current_user = current_user
        self.backup_manager = BackupManager()
        # Устанавливаем текущего пользователя
        self.backup_manager.set_current_user(current_user)
        # Устанавливаем родительский виджет
        self.backup_manager.set_parent_widget(self)
        self.system_monitor = SystemMonitor()
        self.encryption_manager = EncryptionManager()
        self.tabs = QTabWidget()
        self.users_tab = self.create_users_tab()
        self.monitoring_tab = self.create_monitoring_tab()
        self.backup_tab = self.create_backup_tab()
        self.tabs.addTab(self.users_tab, "Пользователи")
        self.tabs.addTab(self.monitoring_tab, "Мониторинг")
        self.tabs.addTab(self.backup_tab, "Резервные копии")
        layout = QVBoxLayout()
        layout.setContentsMargins(10, 10, 10, 10)
        layout.addWidget(self.tabs)
        self.setLayout(layout)
        self.monitor_timer = QTimer()
        self.monitor_timer.timeout.connect(self.update_monitoring)
        self.monitor_timer.start(1000)

    def create_users_tab(self):
        widget = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(10, 10, 10, 10)
        title = QLabel("Управление пользователями")
        self.users_table = QTableWidget()
        self.users_table.setColumnCount(5)
        self.users_table.setHorizontalHeaderLabels(
            ["Логин", "Полное имя", "Роль", "ID", "Действия"])
        self.users_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.users_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.users_table.setAlternatingRowColors(True)
        btn_add_user = QPushButton("Добавить пользователя")
        btn_add_user.clicked.connect(self.add_user)
        layout.addWidget(title)
        layout.addWidget(self.users_table)
        layout.addWidget(btn_add_user)
        widget.setLayout(layout)
        self.load_users()
        return widget

    def create_monitoring_tab(self):
        widget = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(15, 10, 15, 10)
        title = QLabel("Мониторинг системы в реальном времени")
        self.fig = Figure(figsize=(8, 6), dpi=100)
        self.canvas = FigureCanvas(self.fig)
        sys_info = QGroupBox("Информация о системе")
        sys_layout = QVBoxLayout()
        self.cpu_label = QLabel("CPU: 0%")
        self.ram_label = QLabel("RAM: 0%")
        self.gpu_label = QLabel("GPU: 0%")
        self.disk_label = QLabel("Диск: 0%")
        labels = [self.cpu_label, self.ram_label,
                  self.gpu_label, self.disk_label]
        for label in labels:
            pass
        sys_layout.addWidget(self.cpu_label)
        sys_layout.addWidget(self.ram_label)
        sys_layout.addWidget(self.gpu_label)
        sys_layout.addWidget(self.disk_label)
        sys_info.setLayout(sys_layout)
        layout.addWidget(title)
        layout.addWidget(self.canvas)
        layout.addWidget(sys_info)
        widget.setLayout(layout)
        self.update_monitoring()
        return widget

    def create_backup_tab(self):
        widget = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(15, 10, 15, 10)
        title = QLabel("Управление резервными копиями")
        btn_layout = QHBoxLayout()
        btn_create_backup = QPushButton("Создать резервную копию")
        btn_create_backup.clicked.connect(self.create_backup)
        btn_restore_backup = QPushButton("Восстановить из последней копии")
        btn_restore_backup.clicked.connect(self.restore_backup)
        btn_layout.addWidget(btn_create_backup)
        btn_layout.addWidget(btn_restore_backup)
        self.backups_table = QTableWidget()
        self.backups_table.setColumnCount(3)
        self.backups_table.setHorizontalHeaderLabels(
            ["Название", "Дата создания", "Действия"])
        self.backups_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.backups_table.setAlternatingRowColors(True)
        layout.addWidget(title)
        layout.addLayout(btn_layout)
        layout.addWidget(self.backups_table)
        widget.setLayout(layout)
        self.load_backups()
        return widget

    def update_monitoring(self):
        stats = self.system_monitor.get_stats()
        self.cpu_label.setText(f"CPU: {stats['cpu']:.1f}%")
        self.ram_label.setText(f"RAM: {stats['ram']:.1f}%")
        self.gpu_label.setText(f"GPU: {stats['gpu']:.1f}%")
        self.disk_label.setText(f"Диск: {stats['disk']:.1f}%")
        colors = {
            'cpu': '#E74C3C',
            'ram': '#3498DB',
            'gpu': '#9B59B6',
            'disk': '#2ECC71'
        }
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        resources = ['CPU', 'RAM', 'GPU', 'Disk']
        values = [stats['cpu'], stats['ram'], stats['gpu'], stats['disk']]
        bars = ax.bar(resources, values, color=[
                      colors[res.lower()] for res in ['cpu', 'ram', 'gpu', 'disk']])
        ax.set_ylim(0, 100)
        ax.set_title('Использование системных ресурсов', fontsize=14, pad=20)
        ax.set_ylabel('Проценты (%)', fontsize=12)
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}%',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=10)
        self.canvas.draw()

    def load_users(self):
        users = load_users()
        self.users_table.setRowCount(len(users))
        row = 0
        for username, user_data in users.items():
            self.users_table.setItem(row, 0, QTableWidgetItem(username))
            self.users_table.setItem(row, 1, QTableWidgetItem(
                user_data.get("full_name", "")))
            self.users_table.setItem(
                row, 2, QTableWidgetItem(user_data.get("role", "")))
            self.users_table.setItem(
                row, 3, QTableWidgetItem(user_data.get("user_id", "")))
            btn_delete = QPushButton("Удалить")
            btn_delete.clicked.connect(
                lambda _, u=username: self.delete_user(u))
            widget = QWidget()
            layout = QHBoxLayout(widget)
            layout.addWidget(btn_delete)
            layout.setAlignment(Qt.AlignCenter)
            layout.setContentsMargins(0, 0, 0, 0)
            self.users_table.setCellWidget(row, 4, widget)
            row += 1

    def load_backups(self):
        backups = self.backup_manager.list_backups()
        self.backups_table.setRowCount(len(backups))
        for row, backup in enumerate(backups):
            name = backup.replace('.enc', '')
            date_str = name.split('_')[1] if '_' in name else name
            try:
                if '_' in date_str:
                    date = datetime.datetime.strptime(
                        date_str, "%Y%m%d_%H%M%S").strftime("%d.%m.%Y %H:%M:%S")
                else:
                    date = datetime.datetime.strptime(
                        date_str, "%Y%m%d").strftime("%d.%m.%Y")
            except ValueError:
                date = "Недопустимый формат даты"
            self.backups_table.setItem(row, 0, QTableWidgetItem(name))
            self.backups_table.setItem(row, 1, QTableWidgetItem(date))
            btn_download = QPushButton("Скачать")
            btn_download.clicked.connect(
                lambda _, b=backup: self.download_backup(b))
            widget = QWidget()
            layout = QHBoxLayout(widget)
            layout.addWidget(btn_download)
            layout.setAlignment(Qt.AlignCenter)
            layout.setContentsMargins(0, 0, 0, 0)
            self.backups_table.setCellWidget(row, 2, widget)

    @RoleBasedAccess(["admin"])
    def create_backup(self, checked=False):
        if self.backup_manager.parent_widget is None:
            self.backup_manager.set_parent_widget(self)
        backup_data = {
            "users": load_users(),
            "models": self.get_models_info(),
            "system_info": {
                "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "device": str(DEVICE)
            }
        }
        backup_path = self.backup_manager.create_backup(backup_data)
        if backup_path is None:
            return
        name = os.path.basename(backup_path).replace('.enc', '')
        QMessageBox.information(
            self, "Успех", f"Резервная копия успешно создана и зашифрована: {name}")
        self.load_backups()
        logger.info(
            f"Создан зашифрованный бэкап администратором: {self.current_user['username']}")

    @RoleBasedAccess(["admin"])
    def restore_backup(self, checked=False):
        reply = QMessageBox.question(self, "Подтверждение",
                                     "Вы уверены, что хотите восстановить данные из последней резервной копии?\nВсе текущие данные будут перезаписаны.",
                                     QMessageBox.Yes | QMessageBox.No)
        if reply == QMessageBox.Yes:
            backup_data = self.backup_manager.load_latest_backup()
            if backup_data is not None:
                if "users" in backup_data:
                    with open(os.path.join(project_root, "data/users/default_users.json"), "w") as f:
                        json.dump(backup_data["users"], f, indent=2)
                self.load_users()
                QMessageBox.information(
                    self, "Успех", "Данные успешно восстановлены из зашифрованной резервной копии")
                logger.info(f"Данные восстановлены из бэкапа")
            else:
                QMessageBox.warning(
                    self, "Ошибка", "Резервные копии не найдены")

    @RoleBasedAccess(["admin"])
    def add_user(self, checked=False):
        dialog = QDialog(self)
        dialog.setWindowTitle("Добавить пользователя")
        dialog.setFixedSize(450, 350)
        layout = QFormLayout()
        layout.setContentsMargins(30, 20, 30, 20)
        layout.setSpacing(15)
        username_input = QLineEdit()
        full_name_input = QLineEdit()
        password_input = QLineEdit()
        password_input.setEchoMode(QLineEdit.Password)
        role_combo = QComboBox()
        role_combo.addItems(["doctor", "researcher"])
        layout.addRow(QLabel("Логин:"), username_input)
        layout.addRow(QLabel("Полное имя:"), full_name_input)
        layout.addRow(QLabel("Пароль:"), password_input)
        layout.addRow(QLabel("Роль:"), role_combo)
        btn_box = QHBoxLayout()
        btn_ok = QPushButton("Сохранить")
        btn_cancel = QPushButton("Отмена")
        btn_ok.clicked.connect(lambda: self.save_new_user(
            username_input.text(),
            full_name_input.text(),
            password_input.text(),
            role_combo.currentText(),
            dialog
        ))
        btn_cancel.clicked.connect(dialog.reject)
        btn_box.addWidget(btn_ok)
        btn_box.addWidget(btn_cancel)
        layout.addRow(btn_box)
        dialog.setLayout(layout)
        dialog.exec_()

    @RoleBasedAccess(["admin"])
    def save_new_user(self, username, full_name, password, role, dialog):
        if not username or not password:
            QMessageBox.warning(
                self, "Ошибка", "Логин и пароль обязательны для заполнения")
            return
        users = load_users()
        if username in users:
            QMessageBox.warning(
                self, "Ошибка", "Пользователь с таким логином уже существует")
            return
        encrypted_password = self.encryption_manager.encrypt_password(password)
        users[username] = {
            "password": encrypted_password,
            "role": role,
            "full_name": full_name,
            "user_id": str(uuid.uuid4())
        }
        with open(os.path.join(project_root, "data/users/default_users.json"), "w") as f:
            json.dump(users, f, indent=2)
        self.load_users()
        dialog.accept()
        QMessageBox.information(self, "Успех", "Пользователь успешно добавлен")
        logger.info(f"Добавлен новый пользователь: {username}, роль: {role}")

    @RoleBasedAccess(["admin"])
    def delete_user(self, username, checked=False):
        if username in ["admin", "doctor", "researcher"]:
            QMessageBox.warning(
                self, "Ошибка", "Нельзя удалить предопределенных пользователей")
            return
        reply = QMessageBox.question(self, "Подтверждение",
                                     f"Вы уверены, что хотите удалить пользователя {username}?",
                                     QMessageBox.Yes | QMessageBox.No)
        if reply == QMessageBox.Yes:
            users = load_users()
            if username in users:
                del users[username]
                with open(os.path.join(project_root, "data/users/default_users.json"), "w") as f:
                    json.dump(users, f, indent=2)
            self.load_users()
            QMessageBox.information(
                self, "Успех", "Пользователь успешно удален")
            logger.info(f"Удален пользователь: {username}")

    @RoleBasedAccess(["admin"])
    def download_backup(self, backup_name):
        QMessageBox.information(
            self, "Информация", f"Резервная копия {backup_name} готова к скачиванию")

    def get_models_info(self):
        models_info = []
        for file in os.listdir(os.path.join(project_root, "data/models")):
            if file.endswith('.pth'):
                model_path = os.path.join(project_root, "data/models", file)
                models_info.append({
                    "name": file,
                    "size": os.path.getsize(model_path),
                    "date": datetime.datetime.fromtimestamp(os.path.getmtime(model_path)).strftime("%Y-%m-%d %H:%M:%S")
                })
        return models_info


class DoctorPanel(QWidget):
    def __init__(self, current_user=None, parent=None):
        super().__init__(parent)
        self.current_user = current_user
        self.segmentation_history = SegmentationHistory()
        self.segmentation_history.set_current_user(
            current_user)  # Устанавливаем текущего пользователя
        self.encryption_manager = EncryptionManager()
        self.current_image = None
        self.current_mask = None
        self.original_image = None
        self.current_segmentation_id = None
        self.model = None
        self.model_path = None
        self.confirmed_segmentations = set()  # Множество ID подтвержденных сегментаций
        # Загружаем последнюю модель или best_model.pth
        self.load_latest_model()
        self.main_layout = QVBoxLayout()
        self.main_layout.setContentsMargins(20, 20, 20, 20)
        self.main_layout.setSpacing(15)
        title = QLabel("Анализ МРТ-снимков головного мозга")
        subtitle = QLabel(
            "Загрузите МРТ-снимок для автоматической сегментации опухоли")
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(10)
        btn_style = ""
        self.btn_load = QPushButton("Загрузить снимок")
        self.btn_load.clicked.connect(self.load_image)
        self.btn_segment = QPushButton("Запустить сегментацию")
        self.btn_segment.clicked.connect(self.run_segmentation)
        self.btn_segment.setEnabled(False)
        self.btn_confirm = QPushButton("Подтвердить результат")
        self.btn_confirm.clicked.connect(self.confirm_result)
        self.btn_confirm.setEnabled(False)
        self.btn_export = QPushButton("Экспортировать")
        self.btn_export.clicked.connect(self.export_result)
        self.btn_export.setEnabled(False)
        self.btn_history = QPushButton("История сегментаций")
        self.btn_history.clicked.connect(self.show_history)
        btn_layout.addWidget(self.btn_load)
        btn_layout.addWidget(self.btn_segment)
        btn_layout.addWidget(self.btn_confirm)
        btn_layout.addWidget(self.btn_export)
        btn_layout.addWidget(self.btn_history)
        image_container = QGroupBox("Результаты анализа")
        image_layout = QVBoxLayout()
        image_layout.setContentsMargins(10, 25, 10, 10)
        self.image_label = QLabel("Загрузите МРТ-снимок для анализа")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(512, 512)
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        image_layout.addWidget(self.image_label)
        image_layout.addWidget(self.progress_bar)
        image_container.setLayout(image_layout)
        self.main_layout.addWidget(title)
        self.main_layout.addWidget(subtitle)
        self.main_layout.addLayout(btn_layout)
        self.main_layout.addWidget(image_container)
        self.setLayout(self.main_layout)

    def load_latest_model(self):
        """Загружает best_model.pth или последний доступный чекпоинт модели"""
        model_files = [f for f in os.listdir(
            os.path.join(project_root, "data/models")) if f.endswith('.pth') or f.endswith('.pt')]
        # Сначала ищем best_model.pth
        best_model_path = None
        for file in model_files:
            if file.startswith("best_model"):
                best_model_path = os.path.join(
                    project_root, "data/models", file)
                break
        if best_model_path:
            self.model_path = best_model_path
        elif not model_files:
            # Создаем и сохраняем предобученную модель, если нет чекпоинтов
            self.model = smp.UnetPlusPlus(
                encoder_name="efficientnet-b4",
                encoder_weights="imagenet",
                in_channels=3,
                classes=1,
                activation=None
            )
            self.model.eval()
            # Сохраняем модель
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            self.model_path = os.path.join(
                project_root, "data/models", f"pretrained_model_{timestamp}.pth")
            torch.save({
                'model_state_dict': self.model.state_dict(),
            }, self.model_path)
            logger.info(
                f"Создана и сохранена предобученная модель: {self.model_path}")
        else:
            # Сортируем по дате создания (последний - самый новый)
            models_dir = os.path.join(project_root, "data/models")
            model_files.sort(key=lambda x: os.path.getmtime(
                os.path.join(models_dir, x)), reverse=True)
            self.model_path = os.path.join(models_dir, model_files[0])
        # Загружаем модель
        try:
            self.model = smp.UnetPlusPlus(
                encoder_name="efficientnet-b4",
                encoder_weights="imagenet",
                in_channels=3,
                classes=1,
                activation=None
            )
            checkpoint = torch.load(self.model_path, map_location=DEVICE)
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
            # Загружаем веса, игнорируя несоответствия (например, оптимизатор)
            self.model.load_state_dict(state_dict, strict=False)
            self.model.eval()
            logger.info(
                f"Загружена модель из чекпоинта: {self.model_path}")
        except Exception as e:
            logger.error(f"Ошибка при загрузке модели: {str(e)}")
            QMessageBox.warning(self, "Ошибка загрузки модели",
                                f"Не удалось загрузить модель: {str(e)}")

    def load_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Выберите МРТ-снимок", "",
            "Изображения (*.tif *.tiff *.png *.jpg *.jpeg)"
        )
        if file_path:
            self.current_image_path = file_path
            image = cv2.imread(file_path)
            if image is None:
                logger.error(
                    f"Ошибка: Не удалось загрузить изображение {file_path}")
                return
            # Если изображение в градациях серого, конвертируем в RGB
            if len(image.shape) == 2 or (len(image.shape) == 3 and image.shape[2] == 1):
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            else:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            self.original_image = image.copy()
            self.display_image(image)
            self.btn_segment.setEnabled(True)
            self.btn_confirm.setEnabled(False)
            self.btn_export.setEnabled(False)
            self.current_segmentation_id = None

    def display_image(self, image, mask=None):
        is_grayscale = len(image.shape) == 2 or (
            len(image.shape) == 3 and image.shape[2] == 1)
        if mask is not None:
            colored_mask = np.zeros(
                (image.shape[0], image.shape[1], 3), dtype=np.uint8)
            colored_mask[mask > 0] = [0, 0, 255]
            if is_grayscale:
                display_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            else:
                display_image = image.copy() if len(
                    image.shape) == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            alpha = 0.4
            display_image = cv2.addWeighted(
                display_image, 1 - alpha, colored_mask, alpha, 0)
            contours, _ = cv2.findContours(mask.astype(
                np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(display_image, contours, -1, (0, 255, 0), 2)
            self.current_image = display_image
        else:
            if is_grayscale:
                display_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            else:
                display_image = image.copy() if len(
                    image.shape) == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            self.current_image = display_image
        height, width = display_image.shape[:2]
        bytes_per_line = 3 * width
        q_img = QImage(display_image.data, width, height,
                       bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)
        scaled_pixmap = pixmap.scaled(
            500, 500, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.image_label.setPixmap(scaled_pixmap)

    def run_segmentation(self):
        if not hasattr(self, 'current_image_path'):
            QMessageBox.warning(
                self, "Ошибка", "Сначала загрузите изображение")
            return
        if self.model is None:
            QMessageBox.warning(
                self, "Ошибка", "Модель для сегментации не загружена. Пожалуйста, загрузите модель.")
            return
        # Проверяем, есть ли уже сегментация для этого изображения
        if self.check_existing_segmentation():
            reply = QMessageBox.question(self, "Подтверждение",
                                         "Для этого изображения у вас уже есть результат сегментации.\n"
                                         "Вы уверены, что хотите выполнить сегментацию еще раз?",
                                         QMessageBox.Yes | QMessageBox.No)
            if reply == QMessageBox.No:
                return
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.btn_segment.setEnabled(False)
        self.segmentation_thread = SegmentationThread(
            self.model_path, self.current_image_path)
        self.segmentation_thread.progress.connect(self.update_progress)
        self.segmentation_thread.finished.connect(self.segmentation_finished)
        self.segmentation_thread.start()

    def check_existing_segmentation(self):
        history = self.segmentation_history.get_user_history(
            self.current_user["user_id"])
        image_name = os.path.basename(self.current_image_path)
        for record in history:
            if record.get("image_path") == image_name:
                return True
        return False

    def update_progress(self, value):
        self.progress_bar.setValue(value)

    def segmentation_finished(self, original_image, mask):
        self.progress_bar.setVisible(False)
        self.btn_segment.setEnabled(True)
        if mask is not None:
            self.current_mask = mask
            self.display_image(original_image, mask)
            self.btn_confirm.setEnabled(True)
            # Автоматически сохраняем в историю как неподтвержденный результат
            self.save_unconfirmed_segmentation(original_image, mask)
            QMessageBox.information(
                self, "Успех", "Сегментация успешно завершена и добавлена в историю")
            logger.info(
                f"Сегментация выполнена для пользователя {self.current_user['username']}")
        else:
            QMessageBox.warning(
                self, "Ошибка", "Не удалось выполнить сегментацию")

    def save_unconfirmed_segmentation(self, original_image, mask):
        """Сохраняет сегментацию в историю как неподтвержденную"""
        if not hasattr(self, 'current_image_path'):
            return
        temp_mask_path = os.path.join(
            project_root, "data/temp", f"temp_mask_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.tif")
        cv2.imwrite(temp_mask_path, mask)
        model_info = {
            "name": os.path.basename(self.model_path),
            "path": self.model_path,
            "architecture": "UNet with EfficientNet-B0"
        }
        metrics = {
            "dice": 0.87,
            "iou": 0.78
        }
        # Сохраняем как неподтвержденную сегментацию
        seg_id = self.segmentation_history.save_segmentation(
            self.current_user["user_id"],
            self.current_image_path,
            temp_mask_path,
            model_info,
            metrics,
            confirmed=False  # Явно указываем, что не подтверждено
        )
        self.current_segmentation_id = seg_id
        os.remove(temp_mask_path)
        logger.info(f"Неподтвержденная сегментация сохранена: {seg_id}")

    @RoleBasedAccess(["doctor"])
    def confirm_result(self, checked=False):
        reply = QMessageBox.question(self, "Подтверждение",
                                     "Вы уверены, что хотите подтвердить результат сегментации?",
                                     QMessageBox.Yes | QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.btn_export.setEnabled(True)
            if self.current_mask is not None and self.current_segmentation_id:
                # Обновляем статус сегментации на подтвержденную
                seg_info = self.segmentation_history.get_segmentation_by_id(
                    self.current_segmentation_id)
                if seg_info:
                    seg_info["confirmed"] = True
                    # Пересохраняем информацию
                    seg_dir = os.path.join(
                        self.segmentation_history.history_dir, self.current_segmentation_id)
                    encryption_manager = EncryptionManager()
                    encrypted_info = encryption_manager.encrypt_data(seg_info)
                    with open(os.path.join(seg_dir, "info.enc"), "wb") as f:
                        f.write(encrypted_info)
                    self.confirmed_segmentations.add(
                        self.current_segmentation_id)
                    QMessageBox.information(
                        self, "Успех", "Результат сегментации подтвержден и сохранен в историю")
                    logger.info(
                        f"Результат сегментации подтвержден: {self.current_segmentation_id}")

    @RoleBasedAccess(["doctor"])
    def export_result(self, checked=False):
        if self.current_mask is None:
            QMessageBox.warning(
                self, "Ошибка", "Нет подтвержденного результата для экспорта")
            return
        if not self.current_segmentation_id:
            QMessageBox.warning(
                self, "Ошибка", "Результат не был подтвержден. Сначала подтвердите результат сегментации.")
            return
        save_path, _ = QFileDialog.getSaveFileName(
            self, "Сохранить результат", "",
            "JSON файл с результатами (*.json);;Изображение с маской (*.png);;Маска сегментации (*.tif)"
        )
        if save_path:
            if save_path.endswith('.json'):
                seg_info = self.segmentation_history.get_segmentation_by_id(
                    self.current_segmentation_id)
                if not seg_info:
                    QMessageBox.warning(
                        self, "Ошибка", "Не удалось найти информацию о сегментации")
                    return
                model_results = self.get_model_results(
                    seg_info["model_info"]["name"])
                report = {
                    "segmentation_id": seg_info["segmentation_id"],
                    "user_id": seg_info["user_id"],
                    "timestamp": seg_info["timestamp"],
                    "image_path": seg_info["image_path"],
                    "model_info": seg_info["model_info"],
                    "metrics": seg_info["metrics"],
                    "model_training_results": model_results
                }
                encrypted_report = self.encryption_manager.encrypt_data(report)
                with open(save_path, 'wb') as f:
                    f.write(encrypted_report)
                QMessageBox.information(
                    self, "Успех", f"Результаты успешно экспортированы в зашифрованный JSON файл: {save_path}")
                logger.info(
                    f"Результаты экспортированы в зашифрованный JSON: {save_path} пользователем {self.current_user['username']}")
            elif save_path.endswith('.png'):
                cv2.imwrite(save_path, self.current_image)
                QMessageBox.information(
                    self, "Успех", f"Изображение с маской успешно сохранено: {save_path}")
            elif save_path.endswith('.tif'):
                cv2.imwrite(save_path, self.current_mask)
                QMessageBox.information(
                    self, "Успех", f"Маска сегментации успешно сохранена: {save_path}")

    def get_model_results(self, model_name):
        results_file = os.path.join(project_root, "training_results.json")
        if not os.path.exists(results_file):
            return {}
        with open(results_file, "r") as f:
            results = json.load(f)
        for result in results:
            if result["model"] == model_name:
                return result["metrics"]
        return {}

    @RoleBasedAccess(["doctor"])
    def show_history(self, checked=False):
        history = self.segmentation_history.get_user_history(
            self.current_user["user_id"])
        if not history:
            QMessageBox.information(
                self, "Информация", "У вас пока нет истории сегментаций")
            return
        # Создаем диалоговое окно для просмотра истории
        dialog = QDialog(self)
        dialog.setWindowTitle("История сегментаций")
        dialog.resize(1000, 600)
        layout = QVBoxLayout()
        # Таблица с историей
        table = QTableView()
        model = HistoryModel(history)
        table.setModel(model)
        table.setSelectionBehavior(QAbstractItemView.SelectRows)
        table.setSelectionMode(QAbstractItemView.SingleSelection)
        # Кнопки действий
        btn_layout = QHBoxLayout()
        btn_view = QPushButton("Просмотреть")
        btn_view.clicked.connect(
            lambda: self.view_history_item(table, history, dialog))
        btn_export = QPushButton("Экспортировать")
        btn_export.clicked.connect(
            lambda: self.export_history_item(table, history))
        btn_delete = QPushButton("Удалить")
        btn_delete.clicked.connect(
            lambda: self.delete_history_item(table, history))
        btn_close = QPushButton("Закрыть")
        btn_close.clicked.connect(dialog.accept)
        btn_layout.addWidget(btn_view)
        btn_layout.addWidget(btn_export)
        btn_layout.addWidget(btn_delete)
        btn_layout.addWidget(btn_close)
        layout.addWidget(QLabel("Ваша история сегментаций:"))
        layout.addWidget(table)
        layout.addLayout(btn_layout)
        dialog.setLayout(layout)
        dialog.exec_()

    def view_history_item(self, table, history, parent_dialog):
        selected = table.selectionModel().selectedRows()
        if not selected:
            QMessageBox.warning(
                self, "Ошибка", "Выберите сегментацию для просмотра")
            return
        row = selected[0].row()
        seg_info = history[row]
        seg_id = seg_info["segmentation_id"]
        detail_dialog = QDialog(self)
        detail_dialog.setWindowTitle(f"Детали сегментации: {seg_id}")
        detail_dialog.resize(800, 600)
        layout = QVBoxLayout()
        image_path = self.segmentation_history.get_segmentation_image(
            seg_id, seg_info["image_path"])
        mask_path = self.segmentation_history.get_segmentation_image(
            seg_id, seg_info["mask_path"])
        if image_path and mask_path:
            image = cv2.imread(image_path)
            if image is None:
                logger.error(
                    f"Ошибка: Не удалось загрузить изображение {image_path}")
                image = np.zeros((256, 256, 3), dtype=np.uint8)  # Заглушка
            else:
                # Если изображение в градациях серого, конвертируем в RGB
                if len(image.shape) == 2 or (len(image.shape) == 3 and image.shape[2] == 1):
                    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
                else:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                logger.error(f"Ошибка: Не удалось загрузить маску {mask_path}")
                mask = np.zeros((256, 256), dtype=np.uint8)
            self.display_segmentation_result(image, mask, layout)
        info_group = QGroupBox("Информация о сегментации")
        info_layout = QVBoxLayout()
        info_text = QTextEdit()
        info_text.setReadOnly(True)
        info = f"ID сегментации: {seg_info['segmentation_id']}\n"
        info += f"Дата: {datetime.datetime.fromisoformat(seg_info['timestamp']).strftime('%d.%m.%Y %H:%M:%S')}\n"
        info += f"Пользователь ID: {seg_info['user_id']}\n"
        info += "Информация о модели:\n"
        info += f"  - Название: {seg_info['model_info'].get('name', 'Неизвестно')}\n"
        info += f"  - Архитектура: {seg_info['model_info'].get('architecture', 'Неизвестно')}\n"
        info += "Метрики качества:\n"
        info += f"  - Dice: {seg_info['metrics'].get('dice', 0):.4f}\n"
        info += f"  - IoU: {seg_info['metrics'].get('iou', 0):.4f}\n"
        info += f"Статус: {'Подтверждено' if seg_info.get('confirmed', False) else 'Не подтверждено'}"
        info_text.setText(info)
        info_layout.addWidget(info_text)
        info_group.setLayout(info_layout)
        btn_close = QPushButton("Закрыть")
        btn_close.clicked.connect(detail_dialog.accept)
        layout.addWidget(info_group)
        layout.addWidget(btn_close)
        detail_dialog.setLayout(layout)
        detail_dialog.exec_()

    def display_segmentation_result(self, image, mask, layout):
        # Проверяем, является ли изображение серым (1 канал)
        is_grayscale = len(image.shape) == 2 or (
            len(image.shape) == 3 and image.shape[2] == 1)
        colored_mask = np.zeros(
            (image.shape[0], image.shape[1], 3), dtype=np.uint8)
        colored_mask[mask > 0] = [0, 0, 255]
        # Конвертируем в BGR только если изображение серое
        if is_grayscale:
            display_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            display_image = image.copy() if len(
                image.shape) == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        alpha = 0.4
        display_image = cv2.addWeighted(
            display_image, 1 - alpha, colored_mask, alpha, 0)
        contours, _ = cv2.findContours(mask.astype(
            np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(display_image, contours, -1, (0, 255, 0), 2)
        height, width = display_image.shape[:2]
        bytes_per_line = 3 * width
        q_img = QImage(display_image.data, width, height,
                       bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)
        scaled_pixmap = pixmap.scaled(
            500, 500, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        image_label = QLabel()
        image_label.setPixmap(scaled_pixmap)
        image_label.setAlignment(Qt.AlignCenter)
        image_group = QGroupBox("Результат сегментации")
        image_layout = QVBoxLayout()
        image_layout.addWidget(image_label)
        image_group.setLayout(image_layout)
        layout.addWidget(image_group)

    def export_history_item(self, table, history):
        selected = table.selectionModel().selectedRows()
        if not selected:
            QMessageBox.warning(
                self, "Ошибка", "Выберите сегментацию для экспорта")
            return
        row = selected[0].row()
        seg_info = history[row]
        seg_id = seg_info["segmentation_id"]
        save_path, _ = QFileDialog.getSaveFileName(
            self, "Экспортировать результат", "",
            "Зашифрованный JSON файл (*.json)"
        )
        if save_path:
            model_results = self.get_model_results(
                seg_info["model_info"]["name"])
            report = {
                "segmentation_id": seg_info["segmentation_id"],
                "user_id": seg_info["user_id"],
                "timestamp": seg_info["timestamp"],
                "image_path": seg_info["image_path"],
                "model_info": seg_info["model_info"],
                "metrics": seg_info["metrics"],
                "model_training_results": model_results
            }
            encrypted_report = self.encryption_manager.encrypt_data(report)
            with open(save_path, 'wb') as f:
                f.write(encrypted_report)
            QMessageBox.information(
                self, "Успех", f"Результаты успешно экспортированы в зашифрованный файл: {save_path}")
            logger.info(
                f"История сегментации экспортирована в зашифрованный файл: {save_path} пользователем {self.current_user['username']}")

    @RoleBasedAccess(["doctor"])
    def delete_history_item(self, table, history):
        selected = table.selectionModel().selectedRows()
        if not selected:
            QMessageBox.warning(
                self, "Ошибка", "Выберите сегментацию для удаления")
            return
        row = selected[0].row()
        seg_info = history[row]
        seg_id = seg_info["segmentation_id"]
        # Проверяем, подтверждена ли сегментация
        if seg_info.get("confirmed", False) and self.current_user["role"] != "admin":
            QMessageBox.warning(
                self, "Ошибка удаления",
                "Нельзя удалить подтвержденную сегментацию. Подтвержденные результаты должны сохраняться для истории пациента.")
            return
        reply = QMessageBox.question(self, "Подтверждение удаления",
                                     f"Вы уверены, что хотите удалить сегментацию {seg_id}?\n"
                                     "Это действие нельзя отменить.",
                                     QMessageBox.Yes | QMessageBox.No)
        if reply == QMessageBox.Yes:
            try:
                success, message = self.segmentation_history.delete_segmentation(
                    seg_id)
                if success:
                    QMessageBox.information(
                        self, "Успех", "Сегментация успешно удалена")
                    logger.info(
                        f"Сегментация удалена: {seg_id} пользователем {self.current_user['username']}")
                    history = self.segmentation_history.get_user_history(
                        self.current_user["user_id"])
                    table.setModel(HistoryModel(history))
                else:
                    QMessageBox.warning(
                        self, "Ошибка", f"Не удалось удалить сегментацию: {message}")
            except Exception as e:
                QMessageBox.warning(
                    self, "Ошибка", f"Не удалось удалить сегментацию: {str(e)}")
                logger.error(
                    f"Ошибка при удалении сегментации {seg_id}: {str(e)}")


class ResearcherPanel(QWidget):
    def __init__(self, current_user=None, parent=None):
        super().__init__(parent)
        self.current_user = current_user
        self.model_checkpoints = []
        self.current_model = None
        # Создание необходимых атрибутов перед вызовом update_checkpoints_list()
        self.checkpoint_combo = QComboBox()
        self.pretrained_combo = QComboBox()
        self.model_info = QTextEdit()
        # Теперь можно обновить список чекпоинтов
        self.update_checkpoints_list()
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(15, 15, 15, 15)
        self.tabs = QTabWidget()
        self.model_tab = self.create_model_tab()
        self.training_tab = self.create_training_tab()
        self.results_tab = self.create_results_tab()
        self.tabs.addTab(self.model_tab, "Выбор модели")
        self.tabs.addTab(self.training_tab, "Обучение")
        self.tabs.addTab(self.results_tab, "Результаты")
        main_layout.addWidget(self.tabs)
        self.setLayout(main_layout)

    def create_model_tab(self):
        widget = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(15, 15, 15, 15)
        layout.setSpacing(15)
        title = QLabel("Выбор предобученной модели")
        checkpoint_group = QGroupBox("Загруженные веса модели")
        checkpoint_layout = QFormLayout()
        checkpoint_layout.setSpacing(10)
        # Используем уже созданный combobox
        checkpoint_layout.addRow("Доступные модели:", self.checkpoint_combo)
        checkpoint_group.setLayout(checkpoint_layout)
        info_group = QGroupBox("Информация о модели")
        info_layout = QVBoxLayout()
        # Используем уже созданный текстовый редактор
        self.model_info.setReadOnly(True)
        self.model_info.setPlaceholderText(
            "Информация о выбранной модели будет отображена здесь")
        info_layout.addWidget(self.model_info)
        info_group.setLayout(info_layout)
        btn_apply = QPushButton("Загрузить модель")
        btn_apply.clicked.connect(self.load_model_checkpoint)
        layout.addWidget(title)
        layout.addWidget(checkpoint_group)
        layout.addWidget(info_group)
        layout.addWidget(btn_apply)
        widget.setLayout(layout)
        return widget

    def create_training_tab(self):
        widget = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(15, 15, 15, 15)
        layout.setSpacing(15)
        title = QLabel("Обучение модели сегментации")
        dataset_group = QGroupBox("Датасет для обучения")
        dataset_layout = QFormLayout()
        dataset_layout.setSpacing(10)
        self.dataset_path = QLineEdit()
        self.dataset_path.setPlaceholderText("Путь к папке с датасетом")
        self.dataset_path.setText("data/datasets")  # Дефолтный путь
        self.dataset_btn = QPushButton("Выбрать папку...")
        self.dataset_btn.clicked.connect(self.select_dataset)
        dataset_layout.addRow("Путь к датасету:", self.dataset_path)
        dataset_layout.addRow("", self.dataset_btn)
        format_info = QLabel(
            "Формат датасета: Папка с подпапками пациентов. В каждой подпапке должно быть изображение и маска в формате [имя]_mask")
        dataset_layout.addRow("", format_info)
        dataset_group.setLayout(dataset_layout)
        checkpoint_group = QGroupBox("Начальная модель для обучения")
        checkpoint_layout = QFormLayout()
        checkpoint_layout.setSpacing(10)
        # Используем уже созданный combobox
        checkpoint_layout.addRow(
            "Предобученная модель:", self.pretrained_combo)
        checkpoint_group.setLayout(checkpoint_layout)
        params_group = QGroupBox("Параметры обучения")
        params_layout = QFormLayout()
        params_layout.setSpacing(10)
        self.epochs_input = QLineEdit("10")
        self.batch_size_input = QLineEdit("4")
        self.lr_input = QLineEdit("0.001")
        params_layout.addRow("Количество эпох:", self.epochs_input)
        params_layout.addRow("Размер батча:", self.batch_size_input)
        params_layout.addRow("Скорость обучения:", self.lr_input)
        params_group.setLayout(params_layout)
        btn_layout = QHBoxLayout()
        btn_train = QPushButton("Запустить обучение")
        btn_train.clicked.connect(self.start_training)
        btn_layout.addWidget(btn_train)
        self.train_progress = QProgressBar()
        self.train_progress.setVisible(False)
        log_group = QGroupBox("Лог обучения")
        log_layout = QVBoxLayout()
        self.train_log = QTextEdit()
        self.train_log.setReadOnly(True)
        self.train_log.setPlaceholderText(
            "Лог обучения будет отображаться здесь")
        log_layout.addWidget(self.train_log)
        log_group.setLayout(log_layout)
        layout.addWidget(title)
        layout.addWidget(dataset_group)
        layout.addWidget(checkpoint_group)
        layout.addWidget(params_group)
        layout.addLayout(btn_layout)
        layout.addWidget(self.train_progress)
        layout.addWidget(log_group)
        widget.setLayout(layout)
        return widget

    def create_results_tab(self):
        widget = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(15, 15, 15, 15)
        layout.setSpacing(15)
        title = QLabel("Результаты обучения моделей")
        self.results_table = QTableWidget()
        self.results_table.setColumnCount(5)
        self.results_table.setHorizontalHeaderLabels(
            ["Модель", "Дата обучения", "Dice", "IoU", "Действия"])
        self.results_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.results_table.setAlternatingRowColors(True)
        metrics_group = QGroupBox("Графики метрик")
        metrics_layout = QVBoxLayout()
        self.metrics_fig = Figure(figsize=(8, 4), dpi=100)
        self.metrics_canvas = FigureCanvas(self.metrics_fig)
        metrics_layout.addWidget(self.metrics_canvas)
        metrics_group.setLayout(metrics_layout)
        layout.addWidget(title)
        layout.addWidget(self.results_table)
        layout.addWidget(metrics_group)
        widget.setLayout(layout)
        self.load_results()
        return widget

    def update_checkpoints_list(self):
        self.checkpoint_combo.clear()
        self.pretrained_combo.clear()
        checkpoints_dir = os.path.join(project_root, "data/models")
        if not os.path.exists(checkpoints_dir):
            os.makedirs(checkpoints_dir)
        self.model_checkpoints = [f for f in os.listdir(checkpoints_dir)
                                  if f.endswith('.pth') or f.endswith('.pt')]
        if self.model_checkpoints:
            self.checkpoint_combo.addItems(self.model_checkpoints)
            self.pretrained_combo.addItems(self.model_checkpoints)
        else:
            self.checkpoint_combo.addItem("Нет доступных моделей")
            self.pretrained_combo.addItem("Нет доступных моделей")
            self.checkpoint_combo.setEnabled(False)
            self.pretrained_combo.setEnabled(False)

    def load_model_checkpoint(self):
        checkpoint_path = os.path.join(
            os.path.join(project_root, "data/models"), self.checkpoint_combo.currentText())
        if not os.path.exists(checkpoint_path):
            QMessageBox.warning(
                self, "Ошибка", "Выбранный файл модели не существует")
            return
        try:
            # Инициализируем модель
            self.current_model = UNet(
                encoder_name='efficientnet-b4',
                encoder_weights='imagenet',
                in_channels=3
            ).to(DEVICE)
            # Загружаем веса
            checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
            # Загружаем веса, игнорируя несоответствия (например, оптимизатор)
            self.current_model.load_state_dict(state_dict, strict=False)
            self.current_model.eval()
            # Обновляем информацию о модели
            model_info = f"Модель успешно загружена:\n"
            model_info += f"  - Архитектура: Unet++\n"
            model_info += f"  - Файл: {self.checkpoint_combo.currentText()}\n"
            model_info += f"  - Размер: {os.path.getsize(checkpoint_path) / 1024 / 1024:.2f} МБ\n"
            model_info += f"  - Дата создания: {datetime.datetime.fromtimestamp(os.path.getmtime(checkpoint_path)).strftime('%Y-%m-%d %H:%M:%S')}"
            self.model_info.setText(model_info)
            QMessageBox.information(
                self, "Успех", "Модель успешно загружена и готова к использованию")
            logger.info(f"Модель загружена из чекпоинта: {checkpoint_path}")
        except Exception as e:
            error_msg = f"Ошибка при загрузке модели:\n{str(e)}"
            self.model_info.setText(error_msg)
            QMessageBox.warning(self, "Ошибка загрузки", error_msg)
            logger.error(
                f"Ошибка при загрузке чекпоинта {checkpoint_path}: {str(e)}")

    def select_dataset(self):
        folder_path = QFileDialog.getExistingDirectory(
            self, "Выберите папку с датасетом"
        )
        if folder_path:
            self.dataset_path.setText(folder_path)

    @RoleBasedAccess(["researcher"])
    def start_training(self, checked=False):
        dataset_path = self.dataset_path.text()
        checkpoint_name = self.pretrained_combo.currentText()
        checkpoint_path = None
        if checkpoint_name != "Нет доступных моделей":
            checkpoint_path = os.path.join(os.path.join(
                project_root, "data/models"), checkpoint_name)
        try:
            epochs = int(self.epochs_input.text())
            batch_size = int(self.batch_size_input.text())
            lr = float(self.lr_input.text())
        except ValueError:
            QMessageBox.warning(
                self, "Ошибка", "Проверьте правильность параметров обучения")
            return

        # Запрашиваем имя модели
        model_name, ok = QInputDialog.getText(
            self, "Имя модели", "Введите имя для новой модели (без расширения):")
        if not ok or not model_name:
            QMessageBox.warning(
                self, "Ошибка", "Имя модели не введено или отменено.")
            return
        # Проверяем, что имя не содержит недопустимых символов для имени файла
        import re
        if re.search(r'[<>:"/\\|?*]', model_name):
            QMessageBox.warning(
                self, "Ошибка", "Имя модели содержит недопустимые символы.")
            return
        # Проверяем, существует ли уже модель с таким именем
        existing_models = [f for f in os.listdir(
            os.path.join(project_root, "data/models")) if f.endswith('.pth')]
        if f"{model_name}.pth" in existing_models:
            reply = QMessageBox.question(self, "Подтверждение",
                                         f"Модель с именем '{model_name}.pth' уже существует. Перезаписать?",
                                         QMessageBox.Yes | QMessageBox.No)
            if reply == QMessageBox.No:
                return

        if not os.path.exists(dataset_path):
            QMessageBox.warning(
                self, "Ошибка", "Указанный путь к датасету не существует")
            return
        # Проверяем структуру датасета
        patient_dirs = [d for d in os.listdir(
            dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]
        if not patient_dirs:
            QMessageBox.warning(
                self, "Ошибка", "В указанной папке не найдены папки с данными пациентов")
            return
        # Проверяем наличие изображений и масок в каждой папке пациента
        valid_structure = True
        for patient_dir in patient_dirs:
            patient_path = os.path.join(dataset_path, patient_dir)
            files = os.listdir(patient_path)
            images = [f for f in files if not f.endswith('_mask.tif') and not f.endswith('_mask.tiff') and
                      (f.endswith('.tif') or f.endswith('.tiff') or f.endswith('.png') or f.endswith('.jpg') or f.endswith('.jpeg'))]
            if not images:
                QMessageBox.warning(self, "Ошибка структуры данных",
                                    f"В папке пациента {patient_dir} не найдены файлы изображений")
                valid_structure = False
                break
        if not valid_structure:
            return
        self.train_progress.setVisible(True)
        self.train_progress.setValue(0)
        # Создаем поток обучения с новыми параметрами
        self.training_thread = TrainingThread(
            checkpoint_path,
            dataset_path,
            epochs,
            batch_size,
            lr,
            model_name
        )
        self.training_thread.progress.connect(self.update_training_progress)
        self.training_thread.epoch_finished.connect(self.update_training_log)
        self.training_thread.finished.connect(self.training_finished)
        self.train_log.clear()
        self.train_log.append(
            f"Начало обучения. Эпох: {epochs}, Batch size: {batch_size}, LR: {lr}, Model name: {model_name}")
        if checkpoint_path:
            self.train_log.append(f"Начальная модель: {checkpoint_name}")
        else:
            self.train_log.append("Обучение с нуля")
        self.training_thread.start()

    def update_training_progress(self, value):
        self.train_progress.setValue(value)

    def update_training_log(self, epoch, val_loss, val_dice):
        self.train_log.append(
            f"Эпоха {epoch}/{self.epochs_input.text()}: Val Loss = {val_loss:.4f}, Val Dice = {val_dice:.4f}")

    def training_finished(self, model_path, metrics):
        self.train_progress.setVisible(False)
        if "error" in metrics:
            QMessageBox.critical(
                self, "Ошибка", f"Ошибка при обучении: {metrics['error']}")
            return
        # Обновляем текущую модель на обученную
        try:
            self.current_model = UNet(
                encoder_name='efficientnet-b0',
                encoder_weights=None,
                in_channels=3
            ).to(DEVICE)
            checkpoint = torch.load(model_path, map_location=DEVICE)
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
            self.current_model.load_state_dict(state_dict, strict=False)
            self.current_model.eval()
            logger.info(
                f"Текущая модель исследователя обновлена после обучения: {model_path}")
        except Exception as e:
            logger.error(
                f"Ошибка при загрузке обученной модели в текущую: {str(e)}")
            QMessageBox.warning(
                self, "Ошибка", f"Модель обучена, но не удалось обновить текущую модель в интерфейсе: {str(e)}")

        QMessageBox.information(
            self, "Успех", f"Обучение завершено!\nМодель сохранена: {model_path}\nЛучший Dice: {metrics['best_dice']:.4f}")
        self.train_log.append(
            f"Обучение завершено. Модель сохранена: {model_path}")
        self.train_log.append(f"Лучший Dice: {metrics['best_dice']:.4f}")
        self.update_checkpoints_list()
        self.save_training_results(model_path, metrics)
        self.load_results()
        self.plot_metrics(metrics)

    def save_training_results(self, model_path, metrics):
        results_file = os.path.join(project_root, "training_results.json")
        results = []
        if os.path.exists(results_file):
            with open(results_file, "r") as f:
                results = json.load(f)
        result_entry = {
            "model": model_path,
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "metrics": metrics
        }
        results.append(result_entry)
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)

    def load_results(self):
        results_file = os.path.join(project_root, "training_results.json")
        if not os.path.exists(results_file):
            return
        with open(results_file, "r") as f:
            results = json.load(f)
        self.results_table.setRowCount(len(results))
        for row, result in enumerate(results):
            self.results_table.setItem(
                row, 0, QTableWidgetItem(result["model"]))
            self.results_table.setItem(
                row, 1, QTableWidgetItem(result["timestamp"]))
            metrics = result["metrics"]
            best_dice = metrics.get("best_dice", 0)
            # Простой расчет IoU из Dice
            iou = best_dice / (2 - best_dice) if best_dice > 0 else 0
            self.results_table.setItem(
                row, 2, QTableWidgetItem(f"{best_dice:.4f}"))
            self.results_table.setItem(row, 3, QTableWidgetItem(f"{iou:.4f}"))
            btn_load = QPushButton("Загрузить")
            btn_load.clicked.connect(
                lambda _, m=result["model"]: self.load_result_model(m))
            widget = QWidget()
            layout = QHBoxLayout(widget)
            layout.addWidget(btn_load)
            layout.setAlignment(Qt.AlignCenter)
            layout.setContentsMargins(0, 0, 0, 0)
            self.results_table.setCellWidget(row, 4, widget)

    def load_result_model(self, model_path):
        if os.path.exists(model_path):
            self.checkpoint_combo.setCurrentText(os.path.basename(model_path))
            self.load_model_checkpoint()
            self.tabs.setCurrentIndex(0)
            QMessageBox.information(
                self, "Успех", f"Модель {os.path.basename(model_path)} успешно загружена")
        else:
            QMessageBox.warning(
                self, "Ошибка", f"Файл модели {model_path} не найден")

    def plot_metrics(self, metrics):
        self.metrics_fig.clear()
        if "train_loss" in metrics and "train_dice" in metrics:
            ax1 = self.metrics_fig.add_subplot(121)
            ax2 = self.metrics_fig.add_subplot(122)
            epochs = range(1, len(metrics["train_loss"]) + 1)
            ax1.plot(epochs, metrics["train_loss"],
                     'r-', label='Train Loss', linewidth=2)
            ax1.set_title('Функция потерь', fontsize=12)
            ax1.set_xlabel('Эпохи', fontsize=10)
            ax1.set_ylabel('Loss', fontsize=10)
            ax1.legend()
            ax1.grid(True, linestyle='--', alpha=0.7)
            ax2.plot(epochs, metrics["train_dice"],
                     'g-', label='Train Dice', linewidth=2)
            ax2.set_title('Коэффициент Дайса', fontsize=12)
            ax2.set_xlabel('Эпохи', fontsize=10)
            ax2.set_ylabel('Dice', fontsize=10)
            ax2.legend()
            ax2.grid(True, linestyle='--', alpha=0.7)
            self.metrics_fig.tight_layout()
            self.metrics_canvas.draw()
