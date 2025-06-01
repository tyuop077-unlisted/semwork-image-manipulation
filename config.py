import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# Просто бд в папке проекта
DATABASE_URL = f"sqlite:///{BASE_DIR / 'experiments.db'}"

DATA_STORAGE_PATH = BASE_DIR / "data_storage"
TEMP_STORAGE_PATH = BASE_DIR / "temp_storage"

IMAGE_ACCESS_URL_PREFIX = "/images"

# создаём если нету
DATA_STORAGE_PATH.mkdir(parents=True, exist_ok=True)
TEMP_STORAGE_PATH.mkdir(parents=True, exist_ok=True)
