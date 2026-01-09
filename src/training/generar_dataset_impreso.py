import os
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFont

FONT_NAMES = [
    "arial.ttf",
    "times.ttf",
    "cour.ttf"
]

CLASSES_NUM = list("0123456789")
CLASSES_UPPER = list("ABCDEFGHIJKLMNOPQRSTUVWXYZÑ")
CLASSES_LOWER = list("abcdefghijklmnopqrstuvwxyzñ")


def render_char(ch, size=64):
    img = Image.new("L", (size, size), 255)
    draw = ImageDraw.Draw(img)

    font_size = random.randint(40, 50)
    font_name = random.choice(FONT_NAMES)
    font = ImageFont.truetype(font_name, font_size)

    bbox = draw.textbbox((0, 0), ch, font=font)
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]

    x = (size - w) // 2
    y = (size - h) // 2 + random.randint(2, 6)

    draw.text((x, y), ch, font=font, fill=0)
    return np.array(img)


def generate_group(chars, out_dir, n=800):
    os.makedirs(out_dir, exist_ok=True)
    for ch in chars:
        folder = os.path.join(out_dir, ch)
        os.makedirs(folder, exist_ok=True)
        for i in range(n):
            img = render_char(ch)
            Image.fromarray(img).save(
                os.path.join(folder, f"{ch}_{i:05d}.png")
            )


def main():
    root = "src/data/impreso"
    generate_group(CLASSES_NUM, os.path.join(root, "numeros"))
    generate_group(CLASSES_UPPER, os.path.join(root, "mayusculas"))
    generate_group(CLASSES_LOWER, os.path.join(root, "minusculas"))
    print("✅ Dataset impreso generado correctamente (Ñ incluida)")


if __name__ == "__main__":
    main()
