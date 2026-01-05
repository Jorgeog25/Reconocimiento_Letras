## python probar_caracter.py pruebas\A.png


import json
import cv2
import numpy as np
import tensorflow as tf
from utils_preprocess import to_binary, normalize_28x28

MODEL_PATH = "models/ocr_cnn.keras"
LABEL_PATH = "models/label_map.json"

model = tf.keras.models.load_model(MODEL_PATH)
with open(LABEL_PATH, "r", encoding="utf-8") as f:
    label_map = {int(k): v for k, v in json.load(f).items()}

def predict_char(img_path: str, topk: int = 5):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"No se pudo leer la imagen: {img_path}")

    binary = to_binary(img)           # fondo blanco, tinta negra
    x28 = normalize_28x28(binary)     # (28,28) tinta=1
    inp = x28[None, ..., None]        # (1,28,28,1)

    probs = model.predict(inp, verbose=0)[0]
    top_idx = np.argsort(probs)[::-1][:topk]

    print(f"Imagen: {img_path}")
    for i, idx in enumerate(top_idx, 1):
        print(f"Top {i}: '{label_map[int(idx)]}'  prob={probs[idx]:.4f}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Uso: python probar_caracter.py ruta_imagen.png")
    else:
        predict_char(sys.argv[1])
