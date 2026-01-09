import cv2
import numpy as np

def normalize_28x28(roi, size=28):
    """
    Normalización OCR robusta:
    - Mantiene proporciones
    - Más margen vertical (ñ, i)
    """
    roi = roi.astype(np.uint8)

    ys, xs = np.where(roi < 255)
    if len(xs) == 0:
        return np.zeros((size, size), dtype=np.float32)

    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    roi = roi[y0:y1+1, x0:x1+1]

    h, w = roi.shape
    scale = (size - 6) / max(h, w)

    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    roi = cv2.resize(roi, (nw, nh), interpolation=cv2.INTER_AREA)

    canvas = np.ones((size, size), dtype=np.uint8) * 255
    xoff = (size - nw) // 2
    yoff = (size - nh) // 2

    canvas[yoff:yoff+nh, xoff:xoff+nw] = roi
    return 1.0 - (canvas / 255.0)
