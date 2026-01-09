import sys
import os

from src.inference.ocr_infer import OCRInferencer


def resolve_image_path(path: str) -> str:
    """
    Permite pasar rutas relativas de forma flexible.
    Prueba:
    - ruta tal cual
    - relativa a src/
    - relativa a src/pruebas/
    """
    if os.path.exists(path):
        return path

    p1 = os.path.join("src", path)
    if os.path.exists(p1):
        return p1

    p2 = os.path.join("src", "pruebas", path)
    if os.path.exists(p2):
        return p2

    return path  # fallará más abajo con mensaje claro


def main():
    if len(sys.argv) < 2:
        print("❌ Uso incorrecto")
        print("✔ Uso correcto:")
        print("   python -m src.main <imagen.png>")
        print("   python -m src.main src/pruebas/Hola.png")
        sys.exit(1)

    image_path = resolve_image_path(sys.argv[1])

    if not os.path.exists(image_path):
        print(f"❌ No se encontró la imagen: {image_path}")
        sys.exit(1)

    # Inicializar OCR
    ocr = OCRInferencer(
        model_path="src/models/ocr_cnn.keras",
        label_path="src/models/label_map.json"
    )

    # Ejecutar OCR
    texto = ocr.predict_image(image_path, debug=True)

    print("\n--- TEXTO RECONOCIDO ---")
    print(texto)


if __name__ == "__main__":
    main()
