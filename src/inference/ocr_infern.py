import cv2
import json
import numpy as np
import tensorflow as tf

from src.preprocessing.preprocess import to_binary, normalize_28x28
from src.preprocessing.segmentation import segment_characters


# ============================================================
# FEATURES VISUALES ROBUSTAS (ADAPTATIVAS)
# ============================================================

def ink_ratio(img):
    return np.sum(img < 255) / img.size


def aspect_ratio(roi):
    h, w = roi.shape
    return h / max(w, 1)


def has_two_loops(roi):
    inv = 255 - roi
    contours, hierarchy = cv2.findContours(
        inv, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
    )
    if hierarchy is None:
        return False
    holes = sum(1 for h in hierarchy[0] if h[3] != -1)
    return holes >= 2


def has_tilde_like_mark(roi):
    """
    Detección adaptativa de ñ manuscrita.
    No usa umbrales absolutos peligrosos.
    """
    h, w = roi.shape

    if h < w * 0.8:
        return False

    top = roi[:int(0.25 * h), :]
    mid = roi[int(0.35 * h):int(0.65 * h), :]
    bottom = roi[int(0.65 * h):, :]

    ink_top = ink_ratio(top)
    ink_mid = ink_ratio(mid)
    ink_bottom = ink_ratio(bottom)

    # Comparaciones RELATIVAS (clave de robustez)
    cond_top = ink_top > ink_mid * 0.6
    cond_body = ink_mid > ink_bottom * 0.8

    return cond_top and cond_body


def has_upper_dot(roi):
    """
    Punto de i / j manuscrito
    """
    h, _ = roi.shape
    top = roi[:int(0.25 * h), :]
    mid = roi[int(0.4 * h):int(0.7 * h), :]

    return ink_ratio(top) > ink_ratio(mid) * 1.4


# ============================================================
# RESOLUCIÓN DE CANDIDATOS (TOP-K + VISIÓN)
# ============================================================

def resolve_candidates(candidates, roi):
    """
    candidates: [(char, prob), ...]
    """
    chars = [c for c, _ in candidates]
    ar = aspect_ratio(roi)

    # O vs 0
    if "o" in chars and "0" in chars:
        return "0" if ar > 1.2 else "o"

    # S vs 5
    if "S" in chars and "5" in chars:
        return "5" if ar > 1.3 else "S"

    # z vs 7
    if "z" in chars and "7" in chars:
        return "7" if ar > 1.2 else "z"

    # g vs B
    if "g" in chars and "B" in chars:
        return "B" if has_two_loops(roi) else "g"

    # n vs ñ
    if "n" in chars and "ñ" in chars:
        return "ñ" if has_tilde_like_mark(roi) else "n"

    # i vs j
    if "i" in chars and "j" in chars:
        return "j" if has_upper_dot(roi) else "i"

    # fallback: mayor probabilidad
    return max(candidates, key=lambda x: x[1])[0]


# ============================================================
# OCR INFERENCER
# ============================================================

class OCRInferencer:
    def __init__(self, model_path, label_path, top_k=3):
        self.model = tf.keras.models.load_model(model_path)
        self.top_k = top_k

        with open(label_path, "r", encoding="utf-8") as f:
            label_map = json.load(f)

        self.idx_to_char = {int(k): v for k, v in label_map.items()}

    def predict_image(self, image_path, debug=False):
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError("No se pudo cargar la imagen")

        binary = to_binary(img)
        boxes = segment_characters(binary)

        result = []

        if debug:
            dbg = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        for (x, y, w, h) in boxes:
            roi = binary[y:y + h, x:x + w]
            x28 = normalize_28x28(roi)

            probs = self.model.predict(x28[None, ..., None], verbose=0)[0]
            top_idx = np.argsort(probs)[-self.top_k:][::-1]

            candidates = [
                (self.idx_to_char[i], probs[i])
                for i in top_idx
            ]

            final_char = resolve_candidates(candidates, roi)
            result.append(final_char)

            if debug:
                cv2.rectangle(dbg, (x, y), (x + w, y + h), (0, 255, 0), 1)
                cv2.putText(
                    dbg, final_char,
                    (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 0, 255), 2
                )

        if debug:
            cv2.imshow("OCR Debug", dbg)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return "".join(result)
