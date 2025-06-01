from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import cv2
import numpy as np
from numpy import random
from typing import List
from io import BytesIO

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def read_image(file: UploadFile) -> np.ndarray:
    image = np.frombuffer(file.file.read(), np.uint8)
    return cv2.imdecode(image, cv2.IMREAD_COLOR)

def to_bytes(image: np.ndarray) -> bytes:
    success, encoded_image = cv2.imencode('.png', image)
    if not success:
        raise ValueError("Could not encode image")
    return encoded_image.tobytes()

def noise(image, percent):
    mask = random.rand(*image.shape[:2]) < (percent / 100)
    values = random.randint(0, 256, (image.shape[0], image.shape[1], 3), dtype=np.uint8)
    image[mask] = values[mask]
    return image

def noise_removal(image, power):
    if power % 2 == 0:
        power += 1
    return cv2.medianBlur(image, power)

def equalization(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.equalizeHist(image)
    return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

def color_correction(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

def scaling(image, width, height):
    return cv2.resize(image, (width, height))

def rotation(image, angle):
    height, width = image.shape[:2]
    center = (width // 2, height // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1)
    return cv2.warpAffine(image, M, (width, height))

def glass_effect(image, power):
    height, width = image.shape[:2]
    dst = np.zeros_like(image)
    offset_x = np.random.randint(0, power, (height, width))
    offset_y = np.random.randint(0, power, (height, width))

    for i in range(height):
        for j in range(width):
            new_x = min(height - 1, i + offset_x[i, j])
            new_y = min(width - 1, j + offset_y[i, j])
            dst[i, j] = image[new_x, new_y]
    return dst

def motion_blur(image, degree, angle):
    image = np.array(image)
    M = cv2.getRotationMatrix2D((degree / 2, degree / 2), angle, 1)
    motion_blur_kernel = np.diag(np.ones(degree))
    motion_blur_kernel = cv2.warpAffine(motion_blur_kernel, M, (degree, degree))
    motion_blur_kernel = motion_blur_kernel / degree
    blurred = cv2.filter2D(image, -1, motion_blur_kernel)
    cv2.normalize(blurred, blurred, 0, 255, cv2.NORM_MINMAX)
    return np.array(blurred, dtype=np.uint8)

@app.post("/process")
async def process_image(operation: str, files: List[UploadFile] = File(...), parameter: int = 0):
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")

    processed_images = []
    for file in files:
        image = read_image(file)

        if operation == 'noise':
            processed_image = noise(image, parameter)
        elif operation == 'noise_removal':
            processed_image = noise_removal(image, parameter)
        elif operation == 'equalization':
            processed_image = equalization(image)
        elif operation == 'color_correction':
            processed_image = color_correction(image)
        elif operation == 'scaling':
            if not parameter:
                raise HTTPException(status_code=400, detail="parameter missing")
            width, height = parameter, parameter
            processed_image = scaling(image, width, height)
        elif operation == 'rotation':
            processed_image = rotation(image, parameter)
        elif operation == 'glass_effect':
            processed_image = glass_effect(image, parameter)
        elif operation == 'motion_blur':
            if not parameter:
                raise HTTPException(status_code=400, detail="parameter expected")
            degree, angle = parameter, parameter
            processed_image = motion_blur(image, degree, angle)
        else:
            raise HTTPException(status_code=400, detail="Unknown operation")

        processed_images.append((file.filename, processed_image))

    if len(processed_images) == 1:
        filename, image = processed_images[0]
        return StreamingResponse(
            BytesIO(to_bytes(image)),
            media_type="image/png",
            headers={'Content-Disposition': f'attachment; filename={filename}'}
        )
    else:
        return [
            StreamingResponse(
                BytesIO(to_bytes(image)),
                media_type="image/png",
                headers={'Content-Disposition': f'attachment; filename={filename}'}
            )
            for filename, image in processed_images
        ]

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
