import os
import json
import cv2
import numpy as np
import tensorflow as tf

MODEL_PATH = "models/ocr_cnn.keras"
LABEL_PATH = "models/label_map.json"

DEBUG_SAVE = True
DEBUG_DIR = "debug_out"

model = tf.keras.models.load_model(MODEL_PATH)
with open(LABEL_PATH, "r", encoding="utf-8") as f:
    label_map = {int(k): v for k, v in json.load(f).items()}

def to_binary_robust(gray: np.ndarray) -> np.ndarray:
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # asegurar fondo blanco
    if np.sum(th == 0) > np.sum(th == 255):
        th = 255 - th
    return th  # fondo 255, tinta 0

def find_character_boxes(binary: np.ndarray):
    inv = 255 - binary
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    inv = cv2.morphologyEx(inv, cv2.MORPH_OPEN, kernel, iterations=1)
    contours, _ = cv2.findContours(inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w*h < 80:  # ruido
            continue
        if h < 12 or w < 4:
            continue
        boxes.append((x, y, w, h))

    boxes.sort(key=lambda b: b[0])
    return boxes

def normalize_char_for_cnn(roi_binary: np.ndarray, size=28, margin=3) -> np.ndarray:
    # roi_binary: fondo 255, tinta 0
    ys, xs = np.where(roi_binary < 255)
    if len(xs) == 0 or len(ys) == 0:
        return np.zeros((size, size), dtype=np.float32)

    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    roi = roi_binary[y0:y1+1, x0:x1+1]

    h, w = roi.shape
    target = size - 2 * margin
    scale = target / max(h, w)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    # resize (impreso genera grises)
    roi_resized = cv2.resize(roi, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    # RE-BINARIZAR tras resize (CRUCIAL)
    _, roi_resized = cv2.threshold(roi_resized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Engrosar un poco para acercarlo a manuscrito/EMNIST
    ink = 255 - roi_resized
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    ink = cv2.dilate(ink, k, iterations=1)
    roi_resized = 255 - ink

    canvas = np.ones((size, size), dtype=np.uint8) * 255
    y_off = (size - new_h) // 2
    x_off = (size - new_w) // 2
    canvas[y_off:y_off+new_h, x_off:x_off+new_w] = roi_resized

    # float tinta=1 fondo=0
    x28 = 1.0 - (canvas.astype(np.float32) / 255.0)
    return x28

def ocr_image(image_path: str) -> str:
    gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise ValueError(f"No se pudo leer la imagen: {image_path}")

    binary = to_binary_robust(gray)
    boxes = find_character_boxes(binary)

    if DEBUG_SAVE:
        os.makedirs(DEBUG_DIR, exist_ok=True)
        dbg = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        for (x, y, w, h) in boxes:
            cv2.rectangle(dbg, (x, y), (x+w, y+h), (0, 0, 255), 1)
        cv2.imwrite(os.path.join(DEBUG_DIR, "boxes.png"), dbg)

    out = []
    for i, (x, y, w, h) in enumerate(boxes):
        roi = binary[y:y+h, x:x+w]
        x28 = normalize_char_for_cnn(roi)

        if DEBUG_SAVE:
            # Guardar EXACTAMENTE lo que entra al modelo (en uint8 correcto)
            img28 = (1.0 - x28) * 255.0
            img28 = img28.astype(np.uint8)
            cv2.imwrite(os.path.join(DEBUG_DIR, f"char_{i}.png"), img28)

        inp = x28[None, ..., None]
        probs = model.predict(inp, verbose=0)[0]
        out.append(label_map[int(np.argmax(probs))])

    return "".join(out)

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Uso: python OCR_Final_simple.py imagen.png")
        raise SystemExit(1)

    print("\n--- TEXTO RECONOCIDO ---")
    print(ocr_image(sys.argv[1]))
