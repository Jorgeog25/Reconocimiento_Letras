import cv2
import numpy as np

def to_binary(img):
    """
    Convierte a binario:
    - fondo blanco (255)
    - tinta negra (0)
    """
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    _, th = cv2.threshold(
        img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    th = 255 - th
    return th

def normalize_28x28(binary, size=28, margin=4):
    """
    Normaliza un carácter a 28x28:
    - recorte de tinta
    - mantiene proporción
    - centrado
    - salida float [0,1] (tinta=1)
    """
    ys, xs = np.where(binary < 255)
    if len(xs) == 0:
        return np.zeros((size, size), dtype=np.float32)

    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    roi = binary[y0:y1+1, x0:x1+1]

    h, w = roi.shape
    scale = (size - 2 * margin) / max(h, w)
    nh = max(1, int(h * scale))
    nw = max(1, int(w * scale))

    roi = cv2.resize(roi, (nw, nh), interpolation=cv2.INTER_AREA)

    canvas = np.ones((size, size), dtype=np.uint8) * 255
    yoff = (size - nh) // 2
    xoff = (size - nw) // 2
    canvas[yoff:yoff+nh, xoff:xoff+nw] = roi

    canvas = 1.0 - (canvas / 255.0)
    return canvas.astype(np.float32)
