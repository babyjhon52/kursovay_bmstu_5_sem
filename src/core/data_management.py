import os
import shutil
import datetime
import logging
from src.core.security import EncryptionManager, RoleBasedAccess

logger = logging.getLogger('medical_ai_system')


class SegmentationHistory:
    """Управление историей сегментаций"""
    def __init__(self, history_dir=None):
        if history_dir is None:
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            history_dir = os.path.join(project_root, "data/segmentations")
        self.history_dir = history_dir
        if not os.path.exists(history_dir):
            os.makedirs(history_dir)
        self.current_user = None

    def set_current_user(self, user):
        """Установка текущего пользователя для проверки прав доступа"""
        self.current_user = user

    def save_segmentation(self, user_id, image_path, mask_path, model_info, metrics=None, confirmed=False):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        seg_id = f"{user_id}_{timestamp}"
        seg_dir = os.path.join(self.history_dir, seg_id)
        if not os.path.exists(seg_dir):
            os.makedirs(seg_dir)
        img_filename = os.path.basename(image_path)
        mask_filename = os.path.basename(mask_path)
        shutil.copy2(image_path, os.path.join(seg_dir, img_filename))
        shutil.copy2(mask_path, os.path.join(seg_dir, mask_filename))
        seg_info = {
            "segmentation_id": seg_id,
            "user_id": user_id,
            "timestamp": datetime.datetime.now().isoformat(),
            "image_path": img_filename,
            "mask_path": mask_filename,
            "model_info": model_info,
            "metrics": metrics or {},
            "confirmed": confirmed
        }
        # Шифруем информацию о сегментации
        encryption_manager = EncryptionManager()
        encrypted_info = encryption_manager.encrypt_data(seg_info)
        with open(os.path.join(seg_dir, "info.enc"), "wb") as f:
            f.write(encrypted_info)
        logger.info(
            f"Сохранена сегментация {seg_id} для пользователя {user_id}, подтверждено: {confirmed}")
        return seg_id

    def get_user_history(self, user_id):
        history = []
        if not os.path.exists(self.history_dir):
            return history
        for seg_id in os.listdir(self.history_dir):
            seg_dir = os.path.join(self.history_dir, seg_id)
            info_file = os.path.join(seg_dir, "info.enc")
            if os.path.exists(info_file):
                with open(info_file, "rb") as f:
                    try:
                        encrypted_data = f.read()
                        # Расшифровываем информацию
                        encryption_manager = EncryptionManager()
                        info = encryption_manager.decrypt_data(encrypted_data)
                        if info.get("user_id") == user_id:
                            history.append(info)
                    except:
                        continue
        history.sort(key=lambda x: x["timestamp"], reverse=True)
        return history

    def get_segmentation_by_id(self, seg_id):
        seg_dir = os.path.join(self.history_dir, seg_id)
        info_file = os.path.join(seg_dir, "info.enc")
        if os.path.exists(info_file):
            with open(info_file, "rb") as f:
                try:
                    encrypted_data = f.read()
                    encryption_manager = EncryptionManager()
                    return encryption_manager.decrypt_data(encrypted_data)
                except:
                    return None
        return None

    def get_segmentation_image(self, seg_id, image_name):
        seg_dir = os.path.join(self.history_dir, seg_id)
        image_path = os.path.join(seg_dir, image_name)
        if os.path.exists(image_path):
            return image_path
        return None

    @RoleBasedAccess(["admin", "doctor"])
    def delete_segmentation(self, seg_id):
        """Удаление сегментации с проверкой прав доступа и статуса подтверждения"""
        seg_info = self.get_segmentation_by_id(seg_id)
        if seg_info and seg_info.get("confirmed", False) and self.current_user["role"] != "admin":
            logger.warning(
                f"Попытка удаления подтвержденной сегментации {seg_id} пользователем {self.current_user['username']}")
            return False, "Нельзя удалить подтвержденную сегментацию"
        seg_dir = os.path.join(self.history_dir, seg_id)
        if os.path.exists(seg_dir):
            try:
                shutil.rmtree(seg_dir)
                logger.info(
                    f"Сегментация {seg_id} удалена пользователем {self.current_user['username']}")
                return True, "Сегментация успешно удалена"
            except Exception as e:
                logger.error(
                    f"Ошибка при удалении сегментации {seg_id}: {str(e)}")
                return False, f"Ошибка при удалении: {str(e)}"
        return False, "Сегментация не найдена"


class BackupManager:
    """Управление резервными копиями"""
    def __init__(self, backup_dir=None):
        if backup_dir is None:
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            backup_dir = os.path.join(project_root, "data/backups")
        self.backup_dir = backup_dir
        self.encryption_manager = EncryptionManager()
        if not os.path.exists(backup_dir):
            os.makedirs(backup_dir)
        self.current_user = None
        self.parent_widget = None  # Для хранения родительского виджета

    def set_current_user(self, user):
        """Установка текущего пользователя для проверки прав доступа"""
        self.current_user = user

    def set_parent_widget(self, widget):
        """Установка родительского виджета для показа сообщений"""
        self.parent_widget = widget

    @RoleBasedAccess(["admin"])
    def create_backup(self, data, name=None):
        try:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            if name is None:
                name = f"backup_{timestamp}"
            backup_path = os.path.join(self.backup_dir, f"{name}.enc")
            backup_data = {
                "timestamp": timestamp,
                "data": data,
                "created_by": self.current_user["username"]
            }
            encrypted_data = self.encryption_manager.encrypt_data(backup_data)
            with open(backup_path, "wb") as f:
                f.write(encrypted_data)
            logger.info(f"Создан зашифрованный бэкап: {name}")
            return backup_path
        except Exception as e:
            logger.error(f"Ошибка при создании бэкапа: {str(e)}")
            if self.parent_widget is not None:
                from PyQt5.QtWidgets import QMessageBox
                QMessageBox.warning(self.parent_widget, "Ошибка создания бэкапа",
                                    f"Не удалось создать резервную копию: {str(e)}")
            return None

    def load_latest_backup(self):
        backup_files = [f for f in os.listdir(
            self.backup_dir) if f.endswith(".enc")]
        if not backup_files:
            logger.warning("Бэкапы не найдены")
            return None
        backup_files.sort(reverse=True)
        latest_backup = os.path.join(self.backup_dir, backup_files[0])
        try:
            with open(latest_backup, "rb") as f:
                encrypted_data = f.read()
            backup_data = self.encryption_manager.decrypt_data(encrypted_data)
            logger.info(f"Загружен бэкап: {backup_files[0]}")
            return backup_data["data"]
        except Exception as e:
            logger.error(f"Ошибка при загрузке бэкапа: {str(e)}")
            return None

    def list_backups(self):
        backup_files = [f for f in os.listdir(
            self.backup_dir) if f.endswith(".enc")]
        backup_files.sort(reverse=True)
        return backup_files

    @RoleBasedAccess(["admin"])
    def restore_backup(self, backup_path):
        try:
            with open(backup_path, "rb") as f:
                encrypted_data = f.read()
            backup_data = self.encryption_manager.decrypt_data(encrypted_data)
            logger.info("Бэкап восстановлен")
            return backup_data["data"]
        except Exception as e:
            logger.error(f"Ошибка при восстановлении бэкапа: {str(e)}")
            raise

