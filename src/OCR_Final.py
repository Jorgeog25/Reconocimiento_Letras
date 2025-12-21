import cv2, json
import numpy as np
from tensorflow.keras.models import load_model
from utils_preprocess import to_binary, normalize_28x28

model = load_model("models/ocr_cnn.h5")

with open("models/label_map.json") as f:
    label_map = {int(k): v for k,v in json.load(f).items()}

def ocr_image(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    binary = to_binary(img)

    inv = 255 - binary
    contours, _ = cv2.findContours(inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for c in contours:
        x,y,w,h = cv2.boundingRect(c)
        if w*h > 50:
            boxes.append((x,y,w,h))

    boxes.sort(key=lambda b: b[0])

    text = ""
    for x,y,w,h in boxes:
        roi = binary[y:y+h, x:x+w]
        norm = normalize_28x28(roi)
        pred = model.predict(norm[None,...,None], verbose=0)
        text += label_map[int(np.argmax(pred))]

    return text

print(ocr_image("imagen_prueba.png"))

# python OCR_Final.py
