import cv2
import json
import numpy as np
import tensorflow as tf

from src.preprocessing.preprocess import to_binary, normalize_28x28
from src.preprocessing.segmentation import segment_characters


def has_ink_above(roi):
    """
    Detecta tinta significativa en la parte superior del carácter.
    Sirve para distinguir n vs ñ cuando la tilde está fusionada.
    """
    h, w = roi.shape

    # Tomamos el 30% superior
    top = roi[0:int(0.3 * h), :]

    # Proporción de píxeles de tinta (negro)
    ink_ratio = np.sum(top < 255) / top.size

    # Umbral empírico (ajustable si hiciera falta)
    return ink_ratio > 0.03


def normalize_output(char):
    """
    Convierte caracteres problemáticos a ASCII seguro
    """
    if char == "ñ":
        return "enie"
    if char == "Ñ":
        return "ENIE"
    return char


class OCRInferencer:
    def __init__(self, model_path, label_path):
        self.model = tf.keras.models.load_model(model_path)

        with open(label_path, "r", encoding="utf-8") as f:
            label_map = json.load(f)

        self.idx_to_char = {int(k): v for k, v in label_map.items()}

    def predict_image(self, image_path, debug=False):
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError("❌ No se pudo cargar la imagen")

        binary = to_binary(img)
        boxes = segment_characters(binary)

        chars = []

        if debug:
            dbg = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        for (x, y, w, h) in boxes:
            roi = binary[y:y + h, x:x + w]
            x28 = normalize_28x28(roi)

            if x28 is None:
                continue

            x_in = x28[None, ..., None]
            probs = self.model.predict(x_in, verbose=0)[0]
            pred_idx = int(np.argmax(probs))
            pred_char = self.idx_to_char[pred_idx]

            # 🔥 Corrección lógica n → ñ
            if pred_char in ("n", "N"):
                if has_ink_above(roi):
                    pred_char = "ñ" if pred_char == "n" else "Ñ"

            # 🔁 Normalización ASCII (ñ → enie)
            out_char = normalize_output(pred_char)
            chars.append(out_char)

            if debug:
                cv2.rectangle(dbg, (x, y), (x + w, y + h), (0, 255, 0), 1)
                cv2.putText(
                    dbg,
                    out_char,
                    (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2
                )

        if debug:
            cv2.imshow("OCR Debug", dbg)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return "".join(chars)
