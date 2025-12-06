import datetime
import logging
from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QLabel, QPushButton, QFrame, QStackedWidget, QMessageBox, QStatusBar)
from PyQt5.QtCore import Qt

from src.core.data_management import BackupManager
from src.gui.panels import AdminPanel, DoctorPanel, ResearcherPanel

logger = logging.getLogger('medical_ai_system')


class MainWindow(QMainWindow):
    """Главное окно приложения"""

    def __init__(self, current_user=None):
        super().__init__()
        self.current_user = current_user
        self.setWindowTitle("Интеллектуальная система анализа МРТ-снимков")
        self.setGeometry(100, 100, 1200, 800)
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        header = QFrame()
        header.setFixedHeight(70)
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(20, 0, 20, 0)
        title = QLabel("Система анализа МРТ-снимков головного мозга")
        user_info = QLabel(
            f"Пользователь: {current_user['full_name']} | Роль: {current_user['role']}")
        header_layout.addWidget(title)
        header_layout.addStretch()
        header_layout.addWidget(user_info)
        nav_bar = QFrame()
        nav_layout = QHBoxLayout(nav_bar)
        nav_layout.setContentsMargins(10, 5, 10, 5)
        self.btn_admin = QPushButton("Администратор")
        self.btn_doctor = QPushButton("Врач")
        self.btn_researcher = QPushButton("Исследователь")
        self.btn_admin.clicked.connect(lambda: self.show_panel("admin"))
        self.btn_doctor.clicked.connect(lambda: self.show_panel("doctor"))
        self.btn_researcher.clicked.connect(
            lambda: self.show_panel("researcher"))
        nav_layout.addWidget(self.btn_admin)
        nav_layout.addWidget(self.btn_doctor)
        nav_layout.addWidget(self.btn_researcher)
        nav_layout.addStretch()
        self.panel_container = QStackedWidget()
        if current_user["role"] == "admin":
            self.admin_panel = AdminPanel(current_user=current_user)
            self.panel_container.addWidget(self.admin_panel)
        elif current_user["role"] == "doctor":
            self.doctor_panel = DoctorPanel(current_user=current_user)
            self.panel_container.addWidget(self.doctor_panel)
        elif current_user["role"] == "researcher":
            self.researcher_panel = ResearcherPanel(current_user=current_user)
            self.panel_container.addWidget(self.researcher_panel)
        main_layout.addWidget(header)
        main_layout.addWidget(nav_bar)
        main_layout.addWidget(self.panel_container)
        central_widget.setLayout(main_layout)
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage(
            f"Добро пожаловать, {current_user['full_name']}!")
        self.apply_role_restrictions()
        self.load_from_backup()

    def apply_role_restrictions(self):
        """Применение ограничений доступа в зависимости от роли"""
        if self.current_user["role"] == "admin":
            self.btn_admin.setEnabled(True)
            self.btn_doctor.setEnabled(False)
            self.btn_researcher.setEnabled(False)
        elif self.current_user["role"] == "doctor":
            self.btn_admin.setEnabled(False)
            self.btn_doctor.setEnabled(True)
            self.btn_researcher.setEnabled(False)
        elif self.current_user["role"] == "researcher":
            self.btn_admin.setEnabled(False)
            self.btn_doctor.setEnabled(False)
            self.btn_researcher.setEnabled(True)

    def show_panel(self, panel_name):
        """Переключение между панелями"""
        if panel_name == "admin" and self.current_user["role"] not in ["admin"]:
            QMessageBox.warning(
                self, "Доступ запрещен", "У вас нет прав для доступа к панели администратора")
            return
        elif panel_name == "doctor" and self.current_user["role"] not in ["admin", "doctor"]:
            QMessageBox.warning(self, "Доступ запрещен",
                                "У вас нет прав для доступа к панели врача")
            return
        elif panel_name == "researcher" and self.current_user["role"] not in ["admin", "researcher"]:
            QMessageBox.warning(
                self, "Доступ запрещен", "У вас нет прав для доступа к панели исследователя")
            return
        if panel_name == "admin":
            if not hasattr(self, 'admin_panel'):
                self.admin_panel = AdminPanel(current_user=self.current_user)
                self.panel_container.addWidget(self.admin_panel)
            self.panel_container.setCurrentWidget(self.admin_panel)
        elif panel_name == "doctor":
            if not hasattr(self, 'doctor_panel'):
                self.doctor_panel = DoctorPanel(current_user=self.current_user)
                self.panel_container.addWidget(self.doctor_panel)
            self.panel_container.setCurrentWidget(self.doctor_panel)
        elif panel_name == "researcher":
            if not hasattr(self, 'researcher_panel'):
                self.researcher_panel = ResearcherPanel(
                    current_user=self.current_user)
                self.panel_container.addWidget(self.researcher_panel)
            self.panel_container.setCurrentWidget(self.researcher_panel)

    def load_from_backup(self):
        """Загрузка данных из резервной копии"""
        backup_manager = BackupManager()
        backup_manager.set_current_user(self.current_user)
        backup_data = backup_manager.load_latest_backup()
        if backup_data is not None:
            self.status_bar.showMessage(
                f"Данные загружены из зашифрованного бэкапа: {datetime.datetime.now().strftime('%H:%M:%S')}")
