import os
import cv2
import numpy as np
import random
import string

# =============================
# CONFIGURACIÓN
# =============================
OUT_ROOT = "src/data/impreso"
IMG_SIZE = 64
N_PER_CLASS = 600   # 400–800 es buen rango

FONTS = [
    cv2.FONT_HERSHEY_SIMPLEX,
    cv2.FONT_HERSHEY_PLAIN,
    cv2.FONT_HERSHEY_DUPLEX,
    cv2.FONT_HERSHEY_COMPLEX,
    cv2.FONT_HERSHEY_TRIPLEX,
    cv2.FONT_HERSHEY_COMPLEX_SMALL,
]

DIGITS = list("0123456789")
UPPER = list(string.ascii_uppercase)
LOWER = list(string.ascii_lowercase)


# =============================
# RENDER DE CARÁCTER
# =============================
def render_char(ch, img_size=IMG_SIZE):
    img = np.ones((img_size, img_size), dtype=np.uint8) * 255

    font = random.choice(FONTS)
    scale = random.uniform(1.4, 2.2)
    thickness = random.randint(2, 4)

    (tw, th), _ = cv2.getTextSize(ch, font, scale, thickness)
    x = (img_size - tw) // 2 + random.randint(-2, 2)
    y = (img_size + th) // 2 + random.randint(-2, 2)

    cv2.putText(img, ch, (x, y), font, scale, (0,), thickness, cv2.LINE_AA)

    # pequeñas deformaciones (simula escaneo/foto)
    angle = random.uniform(-8, 8)
    M = cv2.getRotationMatrix2D((img_size / 2, img_size / 2), angle, 1.0)
    img = cv2.warpAffine(img, M, (img_size, img_size), borderValue=255)

    return img


# =============================
# GENERADOR PRINCIPAL
# =============================
def main():
    print("📦 Generando dataset impreso estructurado...")

    # Crear carpetas base
    for group in ["numeros", "mayusculas", "minusculas"]:
        os.makedirs(os.path.join(OUT_ROOT, group), exist_ok=True)

    # ---------- NÚMEROS ----------
    for ch in DIGITS:
        folder = os.path.join(OUT_ROOT, "numeros", ch)
        os.makedirs(folder, exist_ok=True)

        for i in range(N_PER_CLASS):
            img = render_char(ch)
            cv2.imwrite(os.path.join(folder, f"{ch}_{i:05d}.png"), img)

    # ---------- MAYÚSCULAS ----------
    for ch in UPPER:
        folder = os.path.join(OUT_ROOT, "mayusculas", ch)
        os.makedirs(folder, exist_ok=True)

        for i in range(N_PER_CLASS):
            img = render_char(ch)
            cv2.imwrite(os.path.join(folder, f"{ch}_{i:05d}.png"), img)

    # ---------- MINÚSCULAS ----------
    for ch in LOWER:
        folder = os.path.join(OUT_ROOT, "minusculas", ch)
        os.makedirs(folder, exist_ok=True)

        for i in range(N_PER_CLASS):
            img = render_char(ch)
            cv2.imwrite(os.path.join(folder, f"{ch}_{i:05d}.png"), img)

    print("✅ Dataset impreso generado correctamente en:")
    print(f"   {OUT_ROOT}")
    print("   Estructura: numeros / mayusculas / minusculas")


if __name__ == "__main__":
    main()
