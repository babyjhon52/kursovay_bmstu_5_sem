import json
import uuid
import logging
from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QLabel, QPushButton,
                             QFormLayout, QLineEdit, QMessageBox)
from PyQt5.QtCore import Qt
from cryptography.fernet import InvalidToken

from src.core.security import EncryptionManager
from src.core.utils import load_users, create_default_users

logger = logging.getLogger('medical_ai_system')


class RoleSelectionDialog(QDialog):
    """Диалог выбора роли пользователя"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Выбор роли")
        self.setFixedSize(400, 250)
        layout = QVBoxLayout()
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(10)
        label = QLabel("Выберите вашу роль в системе:")
        layout.addWidget(label)
        btn_admin = QPushButton("Администратор")
        btn_admin.clicked.connect(lambda: self.accept_role("admin"))
        btn_doctor = QPushButton("Врач-клиницист")
        btn_doctor.clicked.connect(lambda: self.accept_role("doctor"))
        btn_researcher = QPushButton("Исследователь")
        btn_researcher.clicked.connect(lambda: self.accept_role("researcher"))
        layout.addWidget(btn_admin)
        layout.addWidget(btn_doctor)
        layout.addWidget(btn_researcher)
        self.selected_role = None
        self.setLayout(layout)

    def accept_role(self, role):
        self.selected_role = role
        self.accept()


class LoginWindow(QDialog):
    """Окно входа в систему"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Вход в систему")
        self.setFixedSize(450, 400)
        layout = QVBoxLayout()
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)
        title = QLabel(
            "Система анализа МРТ-снимков\nдля выявления опухолей головного мозга")
        title.setAlignment(Qt.AlignCenter)
        form_layout = QFormLayout()
        form_layout.setSpacing(15)
        self.username_input = QLineEdit()
        self.username_input.setPlaceholderText("Введите логин")
        self.password_input = QLineEdit()
        self.password_input.setEchoMode(QLineEdit.Password)
        self.password_input.setPlaceholderText("Введите пароль")
        form_layout.addRow("Логин:", self.username_input)
        form_layout.addRow("Пароль:", self.password_input)
        self.role_btn = QPushButton("Выбрать роль")
        self.role_btn.clicked.connect(self.show_role_selection)
        self.selected_role = None
        self.login_btn = QPushButton("Войти в систему")
        self.login_btn.clicked.connect(self.login)
        layout.addWidget(title)
        layout.addLayout(form_layout)
        layout.addWidget(self.role_btn)
        layout.addWidget(self.login_btn)
        self.setLayout(layout)

    def show_role_selection(self):
        role_dialog = RoleSelectionDialog(self)
        if role_dialog.exec_():
            self.selected_role = role_dialog.selected_role
            self.role_btn.setText(f"Роль выбрана: {self.selected_role}")

    def login(self):
        username = self.username_input.text()
        password = self.password_input.text()
        role = self.selected_role
        users = load_users()
        if username in users:
            try:
                encryption_manager = EncryptionManager()
                stored_password = encryption_manager.decrypt_password(
                    users[username]["password"])
                if stored_password == password:
                    if role is None:
                        QMessageBox.warning(
                            self, "Ошибка", "Выберите роль пользователя")
                        return
                    if users[username]["role"] != role:
                        reply = QMessageBox.question(self, "Подтверждение роли",
                                                     f"Пользователь {username} имеет роль '{users[username]['role']}'.\n"
                                                     f"Вы выбрали роль '{role}'.\n"
                                                     "Продолжить вход с выбранной ролью?",
                                                     QMessageBox.Yes | QMessageBox.No)
                        if reply == QMessageBox.No:
                            return
                    current_user = users[username]
                    current_user["username"] = username
                    if "user_id" not in current_user:
                        current_user["user_id"] = str(uuid.uuid4())
                        users[username] = current_user
                        with open("data/users/default_users.json", "w") as f:
                            json.dump(users, f, indent=2)
                    self.accept()
                    return
            except InvalidToken:
                logger.warning(
                    "Обнаружены недействительные токены. Пересоздание пользователей.")
                create_default_users()
                users = load_users()
                encryption_manager = EncryptionManager()
                stored_password = encryption_manager.decrypt_password(
                    users[username]["password"])
                if stored_password == password:
                    current_user = users[username]
                    current_user["username"] = username
                    if "user_id" not in current_user:
                        current_user["user_id"] = str(uuid.uuid4())
                        users[username] = current_user
                        with open("data/users/default_users.json", "w") as f:
                            json.dump(users, f, indent=2)
                    self.accept()
                    return
        QMessageBox.warning(self, "Ошибка", "Неверный логин или пароль")
