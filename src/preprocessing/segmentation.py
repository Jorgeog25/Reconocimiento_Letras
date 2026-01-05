import cv2

def segment_characters(binary):
    """
    Devuelve bounding boxes ordenadas de izquierda a derecha
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
        boxes.append((x, y, w, h))

    boxes = sorted(boxes, key=lambda b: b[0])
    return boxes
