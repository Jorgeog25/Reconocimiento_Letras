import json
import cv2
import numpy as np
import tensorflow as tf

from src.preprocessing.segmentation import segment_characters
from src.preprocessing.preprocess import to_binary, normalize_28x28


class OCRInferencer:
    def __init__(
        self,
        model_path="src/models/ocr_cnn.keras",
        label_path="src/models/label_map.json"
    ):
        self.model = tf.keras.models.load_model(model_path)
        with open(label_path, "r", encoding="utf-8") as f:
            self.idx_to_char = {int(k): v for k, v in json.load(f).items()}

    def predict_image(self, image_path, debug=False):
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"No se pudo leer la imagen: {image_path}")

        binary = to_binary(img)
        boxes = segment_characters(binary)

        result = []

        for i, (x, y, w, h) in enumerate(boxes):
            pad = 6
            H, W = binary.shape
            roi = binary[
                max(0, y-pad):min(H, y+h+pad),
                max(0, x-pad):min(W, x+w+pad)
            ]

            norm = normalize_28x28(roi)
            inp = norm[None, ..., None]

            probs = self.model.predict(inp, verbose=0)[0]
            top = np.argsort(probs)[-2:][::-1]

            c1 = self.idx_to_char[top[0]]
            c2 = self.idx_to_char[top[1]]

            # Heurística geométrica o / 0
            ar = w / max(1, h)
            if {c1, c2} == {"o", "0"}:
                char = "o" if ar < 0.85 else "0"
            elif {c1, c2} == {"1", "l"}:
                char = "l"
            else:
                char = c1

            result.append(char)

            if debug:
                dbg = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)
                cv2.rectangle(
                    dbg,
                    (0, 0),
                    (dbg.shape[1]-1, dbg.shape[0]-1),
                    (0, 255, 0),
                    1
                )
                cv2.imwrite(f"src/debug_out/roi_{i}.png", dbg)
                cv2.imwrite(
                    f"src/debug_out/char_{i}.png",
                    (norm * 255).astype("uint8")
                )

        text = "".join(result)
        return self._fix_context(text)

    def _fix_context(self, text):
        chars = list(text)
        for i in range(len(chars)):
            if chars[i] == "0":
                if (i > 0 and chars[i-1].isalpha()) or (i+1 < len(chars) and chars[i+1].isalpha()):
                    chars[i] = "o"
        return "".join(chars)
