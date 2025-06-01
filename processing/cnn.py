import cv2
import numpy as np
import torch
import torchvision.transforms.v2 as T
from typing import Dict, Any, Optional
from pathlib import Path

from config import BASE_DIR

_loaded_models_cache: Dict[str, torch.nn.Module] = {}
_model_device: Optional[torch.device] = None

def get_model_device() -> torch.device:
    global _model_device
    if _model_device is None:
        if torch.cuda.is_available():
            _model_device = torch.device("cuda")
            print("CNN: CUDA")
        else:
            _model_device = torch.device("cpu")
            print("CNN: CPU")
    return _model_device

def load_pytorch_model(model_filename: str) -> Optional[torch.nn.Module]:
    global _loaded_models_cache
    if model_filename in _loaded_models_cache:
        return _loaded_models_cache[model_filename]

    model_path = BASE_DIR / "models" / model_filename
    if not model_path.exists():
        print(f"Error: Model file not found at {model_path}")
        return None

    device = get_model_device()
    try:
        # model.load_state_dict(torch.load(model_path, map_location=device))
        model = torch.load(model_path, map_location=device)

        model.to(device)
        model.eval()
        _loaded_models_cache[model_filename] = model
        print(f"CNN: Загрузил '{model_filename}' в {device}.")
        return model
    except Exception as e:
        print(f"Error не загрузилась модель '{model_filename}': {e}")
        return None

def preprocess_image_for_cnn(image: np.ndarray, target_size: tuple = (256, 256)) -> Optional[torch.Tensor]:
    if image is None or image.size == 0:
        return None

    try:
        if image.shape[2] == 1:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

        transform = T.Compose([
            T.ToImage(),
            T.ToDtype(torch.float32, scale=True), # [0.0, 1.0]
            T.Resize(target_size, antialias=True),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        input_tensor = transform(image) # (C, H, W)
        return input_tensor.unsqueeze(0) # (1, C, H, W)
    except Exception as e:
        print(f"Error | CNN image preprocessing: {e}")
        return None


def postprocess_segmentation_mask(
        mask_tensor: torch.Tensor,
        original_image_shape: tuple,
        threshold: float = 0.5
) -> Optional[np.ndarray]:
    try:
        mask = mask_tensor.squeeze().cpu().detach()

        if mask.ndim == 2: # (H, W)
            mask_probs = torch.sigmoid(mask)
        elif mask.ndim == 3 and mask.shape[0] == 1: # (1, H, W)
            mask_probs = torch.sigmoid(mask.squeeze(0))
        else: # (num_classes, H, W)
            print(f"[!] Mask {mask.shape}")
            if mask.ndim == 3:
                mask_probs = torch.sigmoid(mask[0])
            else:
                mask_probs = mask


        binary_mask_np = (mask_probs > threshold).numpy().astype(np.uint8) * 255

        return binary_mask_np

    except Exception as e:
        print(f"Error | CNN mask postprocessing: {e}")
        return None


def count_cells_cnn(image: np.ndarray, params: Dict[str, Any]) -> int:
    if image is None or image.size == 0: return 0

    model_filename = params.get("model_filename", "model.pth") # по-умолчанию
    target_size_h = params.get("target_input_height", 256)
    target_size_w = params.get("target_input_width", 256)
    confidence_threshold = params.get("confidence_threshold", 0.5)
    min_area = params.get("min_cell_area_pixels", 20)

    if not model_filename:
        print("CNN Error: model_filename parameter is missing.")
        return -1

    # загружаем
    model = load_pytorch_model(model_filename)
    if model is None:
        return -2

    # preprocess
    original_shape = image.shape
    input_tensor = preprocess_image_for_cnn(image, target_size=(target_size_h, target_size_w))
    if input_tensor is None:
        return -3

    device = get_model_device()
    input_tensor = input_tensor.to(device)

    # inference
    try:
        with torch.no_grad():
            output_mask_tensor = model(input_tensor) # model output
    except Exception as e:
        print(f"Error | CNN model inference: {e}")
        return -4

    # postprocess Mask
    binary_mask_np = postprocess_segmentation_mask(output_mask_tensor, original_shape, confidence_threshold)
    if binary_mask_np is None:
        return -5

    # cv2.imwrite(str(BASE_DIR / "temp_storage" / f"cnn_mask_{uuid.uuid4().hex}.png"), binary_mask_np)

    # тут считаем
    num_labels, labels_im, stats, centroids = cv2.connectedComponentsWithStats(binary_mask_np, connectivity=8)

    cell_count = 0
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            cell_count += 1

    print(f"CNN: Нашлось {cell_count} клеток используя модель '{model_filename}'. Изначально: {num_labels-1}")
    return cell_count
