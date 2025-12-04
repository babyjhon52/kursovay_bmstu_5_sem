import os
import json
import base64
import logging
from functools import wraps
from cryptography.fernet import Fernet, InvalidToken
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from PyQt5.QtWidgets import QApplication, QWidget, QMessageBox

logger = logging.getLogger('medical_ai_system')


class RoleBasedAccess:
    """Декоратор для контроля доступа на основе ролей"""

    def __init__(self, allowed_roles):
        self.allowed_roles = allowed_roles

    def __call__(self, func):
        @wraps(func)
        def wrapper(instance, *args, **kwargs):
            # Проверяем наличие current_user
            if not hasattr(instance, 'current_user') or instance.current_user is None:
                logger.warning("Попытка доступа без авторизации")
                # Находим подходящий родительский виджет
                parent_widget = None
                if isinstance(instance, QWidget):
                    parent_widget = instance
                elif hasattr(instance, 'parent') and isinstance(instance.parent, QWidget):
                    parent_widget = instance.parent
                elif hasattr(instance, 'parent_widget') and isinstance(instance.parent_widget, QWidget):
                    parent_widget = instance.parent_widget
                else:
                    parent_widget = QApplication.activeWindow()
                if parent_widget is not None:
                    QMessageBox.warning(parent_widget, "Ошибка доступа",
                                        "Вы не авторизованы в системе")
                return None
            user_role = instance.current_user.get('role', '')
            if user_role not in self.allowed_roles:
                logger.warning(
                    f"Попытка несанкционированного доступа. Пользователь: {instance.current_user.get('username', 'unknown')}, Роль: {user_role}, Запрошенные роли: {self.allowed_roles}")
                # Находим подходящий родительский виджет
                parent_widget = None
                if isinstance(instance, QWidget):
                    parent_widget = instance
                elif hasattr(instance, 'parent') and isinstance(instance.parent, QWidget):
                    parent_widget = instance.parent
                elif hasattr(instance, 'parent_widget') and isinstance(instance.parent_widget, QWidget):
                    parent_widget = instance.parent_widget
                else:
                    parent_widget = QApplication.activeWindow()
                if parent_widget is not None:
                    QMessageBox.warning(parent_widget, "Доступ запрещен",
                                        "У вас нет прав для выполнения данной операции")
                return None
            return func(instance, *args, **kwargs)
        return wrapper


class EncryptionManager:
    """Менеджер для шифрования и дешифрования данных"""

    def __init__(self, key_file=None):
        if key_file is None:
            project_root = os.path.dirname(
                os.path.dirname(os.path.dirname(__file__)))
            key_file = os.path.join(project_root, "data/encryption_key.key")
        self.key_file = key_file
        self.key = self.load_or_generate_key()
        self.cipher = Fernet(self.key)

    def load_or_generate_key(self):
        if os.path.exists(self.key_file):
            with open(self.key_file, "rb") as key_file:
                key = key_file.read()
                logger.info("Ключ шифрования загружен из файла")
                return key
        else:
            master_password = "MedicalSystemSecurePass2025!".encode()
            salt = b"medical_ai_salt_2025"
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt,
                iterations=100000,
            )
            key = base64.urlsafe_b64encode(kdf.derive(master_password))
            # Сохраняем ключ в файл
            with open(self.key_file, "wb") as key_file:
                key_file.write(key)
                logger.info("Сгенерирован и сохранен новый ключ шифрования")
            return key

    def encrypt_data(self, data):
        if isinstance(data, dict):
            data = json.dumps(data)
        elif not isinstance(data, str):
            data = str(data)
        encrypted_data = self.cipher.encrypt(data.encode('utf-8'))
        return encrypted_data

    def decrypt_data(self, encrypted_data):
        try:
            decrypted_data = self.cipher.decrypt(encrypted_data)
            decoded = decrypted_data.decode('utf-8')
            # Пытаемся распарсить как JSON, если не получается - возвращаем строку
            try:
                return json.loads(decoded)
            except json.JSONDecodeError:
                # Если это не JSON, возвращаем как строку
                return decoded
        except Exception as e:
            logger.error(f"Ошибка при дешифровании данных: {str(e)}")
            raise

    def encrypt_password(self, password):
        return self.cipher.encrypt(password.encode('utf-8')).decode('utf-8')

    def decrypt_password(self, encrypted_password):
        try:
            return self.cipher.decrypt(encrypted_password.encode('utf-8')).decode('utf-8')
        except InvalidToken:
            logger.warning(
                "Попытка расшифровки с недействительным токеном. Генерация нового ключа.")
            new_manager = EncryptionManager()
            try:
                return new_manager.cipher.decrypt(encrypted_password.encode('utf-8')).decode('utf-8')
            except InvalidToken:
                logger.error(
                    "Не удалось расшифровать пароль даже с новым ключом")
                raise
        except Exception as e:
            logger.error(f"Ошибка при расшифровке пароля: {str(e)}")
            raise
