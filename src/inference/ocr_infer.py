import os
import json
import cv2
import numpy as np
import tensorflow as tf

from src.preprocessing.preprocess import to_binary, normalize_28x28
from src.preprocessing.segmentation import segment_characters


class OCRInferencer:
    def __init__(
        self,
        model_path: str = "src/models/ocr_cnn.keras",
        label_map_path: str = "src/models/label_map.json",
    ):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"No existe el modelo en: {model_path}")

        if not os.path.exists(label_map_path):
            raise FileNotFoundError(f"No existe label_map en: {label_map_path}")

        self.model = tf.keras.models.load_model(model_path)

        with open(label_map_path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        # Por si guardaste el json como {0:"A"} o {"0":"A"}
        self.idx_to_char = {int(k): v for k, v in raw.items()}

    def predict_image(self, image_path: str, debug: bool = False, debug_dir: str = "src/debug_out") -> str:
        """
        Ejecuta OCR sobre una imagen.
        - Segmenta caracteres
        - Normaliza a 28x28
        - Predice con CNN
        - Devuelve string

        Si debug=True, guarda:
        - boxes.png con cajas y predicción
        - char_*.png: las 28x28 que entran al modelo
        """
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"No se pudo leer la imagen: {image_path}")

        binary = to_binary(img)  # fondo 255, tinta 0
        boxes = segment_characters(binary)

        result_chars = []

        if debug:
            os.makedirs(debug_dir, exist_ok=True)
            dbg = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

        for i, (x, y, w, h) in enumerate(boxes):
            roi = binary[y:y + h, x:x + w]

            # Normaliza a 28x28 con tinta=1, fondo=0 (float)
            x28 = normalize_28x28(roi)

            # Tensor para el modelo
            inp = x28[None, ..., None]  # (1,28,28,1)

            probs = self.model.predict(inp, verbose=0)[0]
            idx = int(np.argmax(probs))
            char = self.idx_to_char.get(idx, "?")
            result_chars.append(char)

            if debug:
                # Dibujar caja y letra
                cv2.rectangle(dbg, (x, y), (x + w, y + h), (0, 255, 0), 1)
                cv2.putText(
                    dbg,
                    char,
                    (x, max(0, y - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2
                )

                # Guardar exactamente la 28x28 que entra al modelo (en uint8)
                out28 = (1.0 - x28) * 255.0  # volver a "fondo blanco, tinta negra"
                out28 = np.clip(out28, 0, 255).astype(np.uint8)
                cv2.imwrite(os.path.join(debug_dir, f"char_{i}.png"), out28)

        if debug:
            cv2.imwrite(os.path.join(debug_dir, "boxes.png"), dbg)

        return "".join(result_chars)
