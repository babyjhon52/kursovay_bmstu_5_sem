import time
import psutil
import torch
import logging

logger = logging.getLogger('medical_ai_system')


class SystemMonitor:
    """Мониторинг системных ресурсов"""

    def __init__(self):
        self.last_check = time.time()
        self.cpu_percent = 0
        self.ram_percent = 0
        self.gpu_percent = 0
        self.disk_percent = 0
        self.update_stats()

    def update_stats(self):
        current_time = time.time()
        if current_time - self.last_check > 1:
            self.cpu_percent = psutil.cpu_percent()
            self.ram_percent = psutil.virtual_memory().percent
            self.gpu_percent = torch.cuda.utilization() if torch.cuda.is_available() else 0
            disk = psutil.disk_usage('/')
            self.disk_percent = disk.percent
            self.last_check = current_time

    def get_stats(self):
        self.update_stats()
        return {
            "cpu": self.cpu_percent,
            "ram": self.ram_percent,
            "gpu": self.gpu_percent,
            "disk": self.disk_percent
        }
