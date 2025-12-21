import cv2
import numpy as np

def to_binary(gray):
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    _, th = cv2.threshold(
        blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # fondo blanco / tinta negra
    if np.mean(th) < 127:
        th = 255 - th

    return th

def crop_tight(binary):
    ys, xs = np.where(binary < 128)
    if len(xs) == 0:
        return binary
    return binary[ys.min():ys.max()+1, xs.min():xs.max()+1]

def pad_square(img):
    h, w = img.shape
    size = max(h, w)
    out = np.ones((size, size), dtype=np.uint8) * 255
    y = (size - h) // 2
    x = (size - w) // 2
    out[y:y+h, x:x+w] = img
    return out

def normalize_28x28(binary):
    cropped = crop_tight(binary)
    squared = pad_square(cropped)
    resized = cv2.resize(squared, (28, 28), interpolation=cv2.INTER_AREA)

    # tinta = 1, fondo = 0
    return 1.0 - (resized.astype(np.float32) / 255.0)
