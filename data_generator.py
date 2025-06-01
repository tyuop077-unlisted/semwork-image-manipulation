import numpy as np
import cv2
from typing import Tuple
from schemas import GeneratorParams

def generate_synthetic_cells_image(params: GeneratorParams) -> Tuple[np.ndarray, int]:
    img = np.zeros((params.image_height, params.image_width, 3), dtype=np.uint8)
    actual_cells_count = 0

    cells_drawn = []

    for _ in range(params.num_cells):
        attempts = 0
        while attempts < 50:
            radius = np.random.randint(params.min_radius, params.max_radius + 1)
            x = np.random.randint(radius, params.image_width - radius)
            y = np.random.randint(radius, params.image_height - radius)

            can_draw = True
            if not params.overlap:
                for cx, cy, cr in cells_drawn:
                    distance_sq = (x - cx)**2 + (y - cy)**2
                    if distance_sq < (radius + cr)**2: # найден overlap
                        can_draw = False
                        break

            if can_draw:
                color_intensity = np.random.randint(100, 255) # grayscale intensity
                color = (color_intensity, color_intensity, color_intensity) # simple gray cells
                # color = (np.random.randint(100, 200), np.random.randint(150, 250), np.random.randint(100, 200))
                cv2.circle(img, (x, y), radius, color, -1)
                cv2.circle(img, (x, y), radius, (color_intensity // 2, color_intensity // 2, color_intensity // 2), 1) # Add a border

                cells_drawn.append((x, y, radius))
                actual_cells_count += 1
                break
            attempts += 1

    if params.noise_level > 0:
        # noise
        noise = np.random.normal(0, params.noise_level * 255, img.shape)
        img_noisy = img.astype(np.float32) + noise
        img = np.clip(img_noisy, 0, 255).astype(np.uint8)

    return img, actual_cells_count
