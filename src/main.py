import os
import sys
from src.inference.ocr_infer import OCRInferencer

def resolve_path(p: str) -> str:
    if os.path.exists(p):
        return p

    p2 = os.path.join("src", p)
    if os.path.exists(p2):
        return p2

    p3 = os.path.join("src", "pruebas", p)
    if os.path.exists(p3):
        return p3

    return p

def main():
    if len(sys.argv) < 2:
        print("Uso: python -m src.main <imagen.png>")
        print("Ejemplos:")
        print("  python -m src.main src\\pruebas\\Hola.png")
        print("  python -m src.main Hola.png  (si está en src\\pruebas\\)")
        raise SystemExit(1)

    image_path = resolve_path(sys.argv[1])

    ocr = OCRInferencer(
        model_path="src/models/ocr_cnn.keras",
        label_map_path="src/models/label_map.json"
    )

    text = ocr.predict_image(image_path, debug=True, debug_dir="src/debug_out")

    print("\n--- TEXTO RECONOCIDO ---")
    print(text)

if __name__ == "__main__":
    main()
