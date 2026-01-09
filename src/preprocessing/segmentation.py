import cv2
import numpy as np

def segment_characters(binary):
    """
    Segmentación OCR robusta:
    - Une punto de la i
    - Une tilde de la ñ
    - Evita fragmentación
    - Devuelve cajas ordenadas
    """

    ink = (binary < 128).astype(np.uint8)

    # Dilatación vertical (clave para i y ñ)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 5))
    ink = cv2.dilate(ink, kernel, iterations=1)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(
        ink, connectivity=8
    )

    raw_boxes = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < 25:
            continue
        if h < 10 or w < 3:
            continue
        raw_boxes.append((x, y, w, h))

    # --- Unión por proximidad vertical (i, ñ) ---
    raw_boxes = sorted(raw_boxes, key=lambda b: b[0])

    merged = []
    for box in raw_boxes:
        if not merged:
            merged.append(box)
            continue

        x, y, w, h = box
        x2, y2, w2, h2 = merged[-1]

        close_x = abs(x - x2) < 8
        vertical_stack = y < y2 and (y2 - (y + h)) < 15

        if close_x and vertical_stack:
            nx = min(x, x2)
            ny = min(y, y2)
            nx2 = max(x + w, x2 + w2)
            ny2 = max(y + h, y2 + h2)
            merged[-1] = (nx, ny, nx2 - nx, ny2 - ny)
        else:
            merged.append(box)

    return merged
