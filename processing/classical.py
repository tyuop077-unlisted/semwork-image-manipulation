import cv2
import numpy as np
from typing import Dict, Any

def count_cells_classical(image: np.ndarray, params: Dict[str, Any]) -> int:
    if image is None or image.size == 0:
        return 0
    if len(image.shape) == 2: # grayscale
        gray = image
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # -- 1
    blur_ksize = params.get("blur_ksize", 5)
    if blur_ksize % 2 == 0: blur_ksize +=1 # odd
    blurred = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)

    # -- 2
    threshold_type = params.get("threshold_type", "otsu")
    if threshold_type == "otsu":
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    elif threshold_type == "adaptive":
        block_size = params.get("adaptive_block_size", 11)
        if block_size % 2 == 0: block_size +=1
        c_val = params.get("adaptive_C", 2)
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, block_size, c_val)
    else:
        thresh_val = params.get("simple_thresh_val", 127)
        _, thresh = cv2.threshold(blurred, thresh_val, 255, cv2.THRESH_BINARY_INV)

    # -- 3
    open_k_size = params.get("morph_open_kernel_size", 3)
    kernel = np.ones((open_k_size, open_k_size), np.uint8)
    opening_iter = params.get("morph_open_iter", 2)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=opening_iter)

    dilate_iter = params.get("morph_dilate_iter", 3)
    sure_bg = cv2.dilate(opening, kernel, iterations=dilate_iter)

    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    dist_thresh_ratio = params.get("dist_transform_thresh_ratio", 0.5)
    _, sure_fg = cv2.threshold(dist_transform, dist_thresh_ratio * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)

    unknown = cv2.subtract(sure_bg, sure_fg)

    # -- 4
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    # -- 5
    img_for_watershed = image.copy()
    if len(image.shape) == 2 or image.shape[2] == 1:
        img_for_watershed = cv2.cvtColor(gray if len(image.shape) == 2 else image, cv2.COLOR_GRAY2BGR)

    try:
        markers = cv2.watershed(img_for_watershed, markers)
    except cv2.error as e:
        print(f"Watershed failed: {e}. Falling back to connected components on sure_fg.")
        num_labels, _, _, _ = cv2.connectedComponentsWithStats(sure_fg, connectivity=8)
        return max(0, num_labels - 1)

    num_cells = len(np.unique(markers)) - 2

    return max(0, num_cells)
