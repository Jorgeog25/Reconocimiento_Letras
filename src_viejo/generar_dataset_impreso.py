import os
import cv2
import numpy as np
import random

CLASSES = list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZÑabcdefghijklmnopqrstuvwxyzñ")

FONTS = [
    cv2.FONT_HERSHEY_SIMPLEX,
    cv2.FONT_HERSHEY_PLAIN,
    cv2.FONT_HERSHEY_DUPLEX,
    cv2.FONT_HERSHEY_COMPLEX,
    cv2.FONT_HERSHEY_TRIPLEX,
    cv2.FONT_HERSHEY_COMPLEX_SMALL,
]

def render_char(ch, img_size=64):
    img = np.ones((img_size, img_size), dtype=np.uint8) * 255

    font = random.choice(FONTS)
    scale = random.uniform(1.4, 2.2)
    thickness = random.randint(2, 4)

    # posición aproximada centrada
    (tw, th), _ = cv2.getTextSize(ch, font, scale, thickness)
    x = (img_size - tw) // 2 + random.randint(-2, 2)
    y = (img_size + th) // 2 + random.randint(-2, 2)

    cv2.putText(img, ch, (x, y), font, scale, (0,), thickness, lineType=cv2.LINE_AA)

    # pequeñas transformaciones (parecido a fotos/escaneos)
    angle = random.uniform(-8, 8)
    M = cv2.getRotationMatrix2D((img_size/2, img_size/2), angle, 1.0)
    img = cv2.warpAffine(img, M, (img_size, img_size), borderValue=255)

    return img

def main(out_root="dataset_impreso", n_per_class=800):
    os.makedirs(out_root, exist_ok=True)
    for ch in CLASSES:
        folder = os.path.join(out_root, ch)
        os.makedirs(folder, exist_ok=True)
        for i in range(n_per_class):
            img = render_char(ch)
            cv2.imwrite(os.path.join(folder, f"{ch}_{i:05d}.png"), img)

    print(f"✅ Dataset impreso generado en: {out_root}")

if __name__ == "__main__":
    main(out_root="dataset_impreso", n_per_class=600)  # 600 por clase es suficiente
