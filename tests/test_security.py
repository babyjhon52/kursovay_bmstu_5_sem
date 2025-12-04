"""Тесты для модуля security.py"""
import pytest
import os
import json
import tempfile
import shutil
from src.core.security import EncryptionManager, RoleBasedAccess


class TestEncryptionManager:
    """Тесты для EncryptionManager"""
    
    def setup_method(self):
        """Настройка перед каждым тестом"""
        self.temp_dir = tempfile.mkdtemp()
        self.key_file = os.path.join(self.temp_dir, "test_key.key")
        self.encryption_manager = EncryptionManager(key_file=self.key_file)
    
    def teardown_method(self):
        """Очистка после каждого теста"""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_encrypt_decrypt_data(self):
        """Тест шифрования и дешифрования данных"""
        test_data = {"test": "data", "number": 123}
        encrypted = self.encryption_manager.encrypt_data(test_data)
        assert encrypted is not None
        assert isinstance(encrypted, bytes)
        
        decrypted = self.encryption_manager.decrypt_data(encrypted)
        assert decrypted == test_data
    
    def test_encrypt_decrypt_string(self):
        """Тест шифрования и дешифрования строки"""
        test_string = "test string"
        encrypted = self.encryption_manager.encrypt_data(test_string)
        assert encrypted is not None
        
        decrypted = self.encryption_manager.decrypt_data(encrypted)
        assert decrypted == json.loads(json.dumps(test_string))
    
    def test_encrypt_decrypt_password(self):
        """Тест шифрования и дешифрования пароля"""
        password = "test_password_123"
        encrypted = self.encryption_manager.encrypt_password(password)
        assert encrypted is not None
        assert isinstance(encrypted, str)
        
        decrypted = self.encryption_manager.decrypt_password(encrypted)
        assert decrypted == password
    
    def test_key_generation(self):
        """Тест генерации ключа"""
        assert os.path.exists(self.key_file)
        assert os.path.getsize(self.key_file) > 0
    
    def test_key_loading(self):
        """Тест загрузки существующего ключа"""
        # Создаем новый менеджер с тем же файлом ключа
        new_manager = EncryptionManager(key_file=self.key_file)
        assert new_manager.key == self.encryption_manager.key


class TestRoleBasedAccess:
    """Тесты для RoleBasedAccess декоратора"""
    
    def test_allowed_role(self):
        """Тест доступа с разрешенной ролью"""
        @RoleBasedAccess(["admin", "doctor"])
        def test_method(self):
            return "success"
        
        class TestClass:
            def __init__(self):
                self.current_user = {"role": "admin"}
        
        obj = TestClass()
        result = test_method(obj)
        assert result == "success"
    
    def test_denied_role(self):
        """Тест доступа с запрещенной ролью"""
        @RoleBasedAccess(["admin"])
        def test_method(self):
            return "success"
        
        class TestClass:
            def __init__(self):
                self.current_user = {"role": "researcher"}
        
        obj = TestClass()
        result = test_method(obj)
        assert result is None
    
    def test_no_user(self):
        """Тест доступа без пользователя"""
        @RoleBasedAccess(["admin"])
        def test_method(self):
            return "success"
        
        class TestClass:
            def __init__(self):
                self.current_user = None
        
        obj = TestClass()
        result = test_method(obj)
        assert result is None

