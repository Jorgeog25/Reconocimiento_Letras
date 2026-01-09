import cv2

def segment_characters(binary):
    """
    Segmentación OCR genérica con fusión de componentes.
    Funciona para:
    - i, j
    - ñ
    - signos compuestos
    - manuscrito fragmentado
    """

    inv = 255 - binary
    contours, _ = cv2.findContours(
        inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    boxes = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)

        if w * h < 50:
            continue
        if h < 8 or w < 3:
            continue

        boxes.append((x, y, w, h))

    # ordenar por x
    boxes.sort(key=lambda b: b[0])

    merged = []

    for box in boxes:
        if not merged:
            merged.append(box)
            continue

        x, y, w, h = box
        px, py, pw, ph = merged[-1]

        # solapamiento horizontal
        overlap_x = min(x + w, px + pw) - max(x, px)
        overlap_ratio = overlap_x / min(w, pw)

        # relación vertical
        vertical_gap = py - (y + h)

        is_above = y < py
        close_vertically = 0 < vertical_gap < ph * 0.7
        small_top = h < ph * 0.7

        if (
            overlap_ratio > 0.3
            and is_above
            and close_vertically
            and small_top
        ):
            # fusionar
            nx = min(x, px)
            ny = min(y, py)
            nx2 = max(x + w, px + pw)
            ny2 = max(y + h, py + ph)
            merged[-1] = (nx, ny, nx2 - nx, ny2 - ny)
        else:
            merged.append(box)

    return merged
