import datetime
from PyQt5.QtCore import Qt, QAbstractTableModel
from PyQt5.QtGui import QColor


class HistoryModel(QAbstractTableModel):
    """Модель для отображения истории сегментаций в таблице"""
    def __init__(self, history_data=None, parent=None):
        super().__init__(parent)
        self.history_data = history_data or []
        # Обновленные заголовки с добавлением статуса
        self.headers = ["ID сегментации", "Дата", "Модель", "Dice", "Статус"]

    def rowCount(self, parent=None):
        return len(self.history_data)

    def columnCount(self, parent=None):
        return len(self.headers)

    def data(self, index, role=Qt.DisplayRole):
        if role == Qt.DisplayRole:
            seg = self.history_data[index.row()]
            if index.column() == 0:
                return seg.get("segmentation_id", "")
            elif index.column() == 1:
                return datetime.datetime.fromisoformat(seg["timestamp"]).strftime("%d.%m.%Y %H:%M")
            elif index.column() == 2:
                return seg.get("model_info", {}).get("name", "Неизвестно")
            elif index.column() == 3:
                return f"{seg.get('metrics', {}).get('dice', 0):.4f}"
            elif index.column() == 4:  # Статус подтверждения
                return "Подтверждено" if seg.get("confirmed", False) else "Не подтверждено"
        elif role == Qt.ForegroundRole:
            # Цвет текста для статуса
            if index.column() == 4:
                seg = self.history_data[index.row()]
                if seg.get("confirmed", False):
                    return QColor(39, 174, 96)  # Зеленый для подтвержденных
                else:
                    return QColor(231, 76, 60)  # Красный для неподтвержденных
        return None

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if role == Qt.DisplayRole and orientation == Qt.Horizontal:
            return self.headers[section]
        return None

