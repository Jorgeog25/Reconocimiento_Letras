## python OCR_Final_simple.py pruebas\hola.png

import json
import cv2
import numpy as np
import tensorflow as tf
from utils_preprocess import to_binary

# =============================
# CARGA DE MODELO Y ETIQUETAS
# =============================

MODEL_PATH = "models/ocr_cnn.keras"
LABEL_PATH = "models/label_map.json"

model = tf.keras.models.load_model(MODEL_PATH)

with open(LABEL_PATH, "r", encoding="utf-8") as f:
    label_map = {int(k): v for k, v in json.load(f).items()}

# =============================
# NORMALIZACIÓN CORRECTA DE CARACTERES
# =============================

def normalize_char(roi, size=28, margin=4):
    """
    Normaliza un carácter para que sea compatible con el entrenamiento:
    - elimina fondo sobrante
    - mantiene proporción
    - centra el carácter
    - fondo blanco, tinta negra
    """
    # eliminar zonas blancas
    ys, xs = np.where(roi < 255)
    if len(xs) == 0 or len(ys) == 0:
        return np.ones((size, size), dtype=np.float32)

    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    roi = roi[y0:y1+1, x0:x1+1]

    h, w = roi.shape
    scale = (size - 2 * margin) / max(h, w)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))

    roi_resized = cv2.resize(roi, (new_w, new_h), interpolation=cv2.INTER_AREA)

    canvas = np.ones((size, size), dtype=np.float32) * 255
    y_off = (size - new_h) // 2
    x_off = (size - new_w) // 2
    canvas[y_off:y_off + new_h, x_off:x_off + new_w] = roi_resized

    canvas = 1.0 - (canvas / 255.0)  # tinta=1, fondo=0
    return canvas

# =============================
# SEGMENTACIÓN DE CARACTERES
# =============================

def find_character_boxes(binary):
    inv = 255 - binary
    contours, _ = cv2.findContours(
        inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    boxes = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        area = w * h

        if area < 50:
            continue
        if h < 10 or w < 3:
            continue

        boxes.append((x, y, w, h))

    return boxes

def sort_left_to_right(boxes):
    return sorted(boxes, key=lambda b: b[0])

# =============================
# OCR PRINCIPAL
# =============================

def ocr_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"No se pudo leer la imagen: {image_path}")

    binary = to_binary(img)

    boxes = find_character_boxes(binary)
    boxes = sort_left_to_right(boxes)

    result = ""

    for x, y, w, h in boxes:
        roi = binary[y:y + h, x:x + w]
        x28 = normalize_char(roi)

        inp = x28[None, ..., None]  # (1, 28, 28, 1)
        probs = model.predict(inp, verbose=0)
        idx = int(np.argmax(probs))

        result += label_map[idx]

    return result

# =============================
# MAIN
# =============================

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Uso: python OCR_Final_simple.py imagen.png")
        raise SystemExit(1)

    texto = ocr_image(sys.argv[1])
    print("\n--- TEXTO RECONOCIDO ---")
    print(texto)
