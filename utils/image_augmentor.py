import cv2
import numpy as np
from numpy import random
from typing import Optional

def noise(image: np.ndarray, percent: int) -> np.ndarray:
    if percent == 0:
        return image.copy()
    # сделаем копию, так как изображение immutable
    img_copy = image.copy()
    noise_mask = random.rand(*img_copy.shape[:2]) < (percent / 100.0)

    if len(img_copy.shape) == 3: # цвет
        noise_values = random.randint(0, 256, (img_copy.shape[0], img_copy.shape[1], img_copy.shape[2]), dtype=np.uint8)
    else: # grayscale
        noise_values = random.randint(0, 256, (img_copy.shape[0], img_copy.shape[1]), dtype=np.uint8)

    img_copy[noise_mask] = noise_values[noise_mask]
    return img_copy

def noise_removal(image: np.ndarray, power: int) -> np.ndarray:
    if power <= 0:
        return image.copy()
    if power % 2 == 0:
        power += 1 # odd
    median = cv2.medianBlur(image, power)
    return median

def equalization(image: np.ndarray) -> np.ndarray:
    if len(image.shape) == 2 or image.shape[2] == 1: # grayscale
        equalized_image = cv2.equalizeHist(image)
        return equalized_image
    elif len(image.shape) == 3 and image.shape[2] == 3: # цвет
        img_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
        img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
        equalized_image = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
        return equalized_image
    return image.copy()

def color_correction(image: np.ndarray) -> np.ndarray:
    if len(image.shape) < 3 or image.shape[2] != 3: # цветной
        return image.copy()
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l_clahe = clahe.apply(l)
    lab_clahe = cv2.merge((l_clahe, a, b))
    output = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)
    return output

def scaling(image: np.ndarray, width: int, height: Optional[int] = None) -> np.ndarray:

    if height is None:
        height = width

    if width <= 0 or height <=0:
        return image.copy()

    resized_image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    return resized_image

def rotation(image: np.ndarray, angle: int) -> np.ndarray:
    if angle == 0:
        return image.copy()
    height, width = image.shape[:2]
    center = (width // 2, height // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_image = cv2.warpAffine(image, M, (width, height))
    return rotated_image

def glass_effect(image: np.ndarray, power: int) -> np.ndarray:
    if power <= 0:
        return image.copy()
    img_copy = image.copy()
    height, width = img_copy.shape[:2]
    dst = np.zeros_like(img_copy)

    rand_x = random.randint(-power, power + 1, size=(height, width))
    rand_y = random.randint(-power, power + 1, size=(height, width))

    for i in range(height):
        for j in range(width):
            offset_x = rand_x[i, j]
            offset_y = rand_y[i, j]

            new_x = np.clip(i + offset_x, 0, height - 1)
            new_y = np.clip(j + offset_y, 0, width - 1)

            dst[i, j] = img_copy[new_x, new_y]
    return dst

def motion_blur(image: np.ndarray, degree: int, angle: Optional[int]=None) -> np.ndarray:
    if degree <= 0:
        return image.copy()

    kernel_size = degree
    if angle is None:
        angle_val = 0
    else:
        angle_val = angle

    kernel_motion_blur = np.zeros((kernel_size, kernel_size))

    center = (kernel_size - 1) // 2

    if kernel_size == 1:
        return image.copy()

    M = cv2.getRotationMatrix2D((center, center), angle_val, 1)
    motion_blur_kernel_diag = np.diag(np.ones(kernel_size)) # диагональная линия
    rotated_kernel = cv2.warpAffine(motion_blur_kernel_diag, M, (kernel_size, kernel_size))

    # нормализируем
    kernel_sum = rotated_kernel.sum()
    if kernel_sum == 0:
        return image.copy()
    motion_blur_kernel = rotated_kernel / kernel_sum

    # и применяем
    blurred = cv2.filter2D(image, -1, motion_blur_kernel)
    return blurred.astype(np.uint8)


def apply_augmentation(image: np.ndarray, operation: str, parameter: Optional[int]=None) -> np.ndarray:
    processed_image = image.copy()

    # тут стандартные значения если не указали
    if operation == 'noise':
        if parameter is None: parameter = 10
        processed_image = noise(processed_image, parameter)
    elif operation == 'noise_removal':
        if parameter is None: parameter = 3
        processed_image = noise_removal(processed_image, parameter)
    elif operation == 'equalization':
        processed_image = equalization(processed_image)
    elif operation == 'color_correction':
        processed_image = color_correction(processed_image)
    elif operation == 'scaling':
        if parameter is None: parameter = 128
        processed_image = scaling(processed_image, width=parameter, height=parameter)
    elif operation == 'rotation':
        if parameter is None: parameter = 0
        processed_image = rotation(processed_image, parameter)
    elif operation == 'glass_effect':
        if parameter is None: parameter = 5
        processed_image = glass_effect(processed_image, parameter)
    elif operation == 'motion_blur':
        if parameter is None: parameter = 5
        processed_image = motion_blur(processed_image, degree=parameter, angle=parameter)
    else:
        pass

    return processed_image
