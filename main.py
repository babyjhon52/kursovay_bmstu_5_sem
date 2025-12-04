"""
Точка входа в приложение системы анализа МРТ-снимков
"""
from src.core.config import init_directories
from src.core.utils import load_users
from src.gui.main_window import MainWindow
from src.gui.dialogs import LoginWindow
from PyQt5.QtGui import QPalette, QColor
from PyQt5.QtWidgets import QApplication, QDialog
import sys
import os

# Добавляем путь к корню проекта в PYTHONPATH
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)


if __name__ == "__main__":
    # Инициализация директорий
    init_directories()

    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(245, 247, 250))
    palette.setColor(QPalette.WindowText, QColor(44, 62, 80))
    palette.setColor(QPalette.Base, QColor(255, 255, 255))
    palette.setColor(QPalette.AlternateBase, QColor(245, 247, 250))
    palette.setColor(QPalette.ToolTipBase, QColor(255, 255, 255))
    palette.setColor(QPalette.ToolTipText, QColor(44, 62, 80))
    palette.setColor(QPalette.Text, QColor(44, 62, 80))
    palette.setColor(QPalette.Button, QColor(236, 240, 241))
    palette.setColor(QPalette.ButtonText, QColor(44, 62, 80))
    palette.setColor(QPalette.BrightText, QColor(231, 76, 60))
    palette.setColor(QPalette.Highlight, QColor(52, 152, 219))
    palette.setColor(QPalette.HighlightedText, QColor(255, 255, 255))
    app.setPalette(palette)

    login_window = LoginWindow()
    if login_window.exec_() == QDialog.Accepted:
        current_user = {
            "username": login_window.username_input.text(),
            **load_users()[login_window.username_input.text()]
        }
        main_window = MainWindow(current_user=current_user)
        main_window.show()
        sys.exit(app.exec_())
    else:
        sys.exit(0)
