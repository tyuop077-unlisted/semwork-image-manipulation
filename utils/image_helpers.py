import cv2
import numpy as np
from fastapi import UploadFile
import shutil
import uuid
from pathlib import Path
from typing import Tuple


def save_image_to_path(image_array: np.ndarray, dir_path: Path, filename_prefix: str = "img_") -> Tuple[str, Path]:
    filename = f"{filename_prefix}{uuid.uuid4().hex}.png"
    full_path = dir_path / filename
    cv2.imwrite(str(full_path), image_array)
    return filename, full_path

def read_image_from_uploadfile(file: UploadFile) -> np.ndarray:
    contents = file.file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    file.file.seek(0)
    return img

def read_image_from_path(image_path: Path) -> np.ndarray:
    return cv2.imread(str(image_path))

def save_upload_file_to_storage(upload_file: UploadFile, dir_path: Path) -> Tuple[str, Path]:
    original_ext = Path(upload_file.filename).suffix if upload_file.filename else ".png"
    filename = f"upload_{uuid.uuid4().hex}{original_ext}"
    full_path = dir_path / filename
    try:
        with open(full_path, "wb") as buffer:
            shutil.copyfileobj(upload_file.file, buffer)
    finally:
        upload_file.file.close()
    return filename, full_path
