# Тесты для системы анализа МРТ-снимков

## Установка зависимостей для тестирования

```bash
pip install pytest pytest-cov
```

## Запуск тестов

### Запуск всех тестов

```bash
pytest tests/ -v
```

### Запуск конкретного файла тестов

```bash
pytest tests/test_security.py -v
pytest tests/test_utils.py -v
pytest tests/test_data_management.py -v
pytest tests/test_monitoring.py -v
pytest tests/test_models.py -v
pytest tests/test_config.py -v
```

### Запуск с покрытием кода

```bash
pytest tests/ --cov=. --cov-report=html
```

### Запуск конкретного теста

```bash
pytest tests/test_security.py::TestEncryptionManager::test_encrypt_decrypt_data -v
```
