import json
import cv2
import numpy as np
import tensorflow as tf

MODEL_PATH = "models/ocr_cnn.keras"
LABEL_PATH = "models/label_map.json"

model = tf.keras.models.load_model(MODEL_PATH)
with open(LABEL_PATH, "r", encoding="utf-8") as f:
    label_map = {int(k): v for k, v in json.load(f).items()}

def predict_28x28(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("No se pudo leer: " + path)
    img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)

    # img viene con tinta negra (0) y fondo blanco (255)
    x28 = 1.0 - (img.astype(np.float32) / 255.0)

    probs = model.predict(x28[None, ..., None], verbose=0)[0]
    idx = int(np.argmax(probs))
    print("Pred:", label_map[idx], "prob:", float(probs[idx]))

if __name__ == "__main__":
    import sys
    predict_28x28(sys.argv[1])
