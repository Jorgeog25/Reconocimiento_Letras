import json
import cv2
import numpy as np
import tensorflow as tf

from src.preprocessing.preprocess import to_binary, normalize_28x28
from src.preprocessing.segmentation import segment_characters


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

        # DEBUG: imagen completa
        debug_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        result = []

        for i, (x, y, w, h) in enumerate(boxes):
            pad = 4
            H, W = img.shape

            roi = binary[
                max(0, y-pad):min(H, y+h+pad),
                max(0, x-pad):min(W, x+w+pad)
            ]

            x28 = normalize_28x28(roi)
            inp = x28[None, ..., None]

            probs = self.model.predict(inp, verbose=0)[0]
            idx = int(np.argmax(probs))
            char = self.idx_to_char[idx]
            result.append(char)

            # DEBUG visual
            if debug:
                cv2.rectangle(
                    debug_img,
                    (x, y),
                    (x+w, y+h),
                    (0, 255, 0),
                    1
                )
                cv2.putText(
                    debug_img,
                    char,
                    (x, y-5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    1
                )

                cv2.imwrite(
                    f"src/debug_out/char_{i}.png",
                    (x28 * 255).astype("uint8")
                )

        if debug:
            cv2.imwrite("src/debug_out/debug_word.png", debug_img)

        return "".join(result)
