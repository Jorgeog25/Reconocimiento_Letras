import sys
from src.inference.ocr_infer import OCRInferencer


def main():
    if len(sys.argv) < 2:
        print("Uso: python main.py imagen.png")
        return

    image_path = sys.argv[1]

    ocr = OCRInferencer()
    text = ocr.predict_image(image_path, debug=True)

    print("\n--- TEXTO RECONOCIDO ---")
    print(text)

if __name__ == "__main__":
    main()
