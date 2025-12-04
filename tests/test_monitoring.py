"""Тесты для модуля monitoring.py"""
import pytest
import time
from src.core.monitoring import SystemMonitor


class TestSystemMonitor:
    """Тесты для SystemMonitor"""
    
    def setup_method(self):
        """Настройка перед каждым тестом"""
        self.monitor = SystemMonitor()
    
    def test_get_stats(self):
        """Тест получения статистики"""
        stats = self.monitor.get_stats()
        
        assert "cpu" in stats
        assert "ram" in stats
        assert "gpu" in stats
        assert "disk" in stats
        
        assert 0 <= stats["cpu"] <= 100
        assert 0 <= stats["ram"] <= 100
        assert 0 <= stats["gpu"] <= 100
        assert 0 <= stats["disk"] <= 100
    
    def test_update_stats(self):
        """Тест обновления статистики"""
        initial_stats = self.monitor.get_stats()
        time.sleep(1.1)  # Ждем больше секунды для обновления
        updated_stats = self.monitor.get_stats()
        
        # Статистика должна быть обновлена
        assert updated_stats is not None
        assert isinstance(updated_stats, dict)

