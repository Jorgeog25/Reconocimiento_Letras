import os
import json
import cv2
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

from src.preprocessing.preprocess import to_binary, normalize_28x28
from src.training.build_model import build_ocr_model


# =============================
# RUTAS
# =============================
DATA_MANUS = "src/data/manuscrito"
DATA_PRINT = "src/data/impreso"
MODEL_OUT = "src/models/ocr_cnn.keras"
LABEL_MAP = "src/models/label_map.json"

os.makedirs("src/models", exist_ok=True)

# =============================
# CLASES (DIFERENCIA MAY/MIN)
# =============================
# Recomendación: empezar sin ñ/Ñ para evitar problemas de datos insuficientes
CLASSES = list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz")
# Si de verdad tienes datos suficientes, puedes usar:
# CLASSES = list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZÑabcdefghijklmnopqrstuvwxyzñ")

char_to_idx = {c: i for i, c in enumerate(CLASSES)}
num_classes = len(CLASSES)

with open(LABEL_MAP, "w", encoding="utf-8") as f:
    json.dump({str(i): c for i, c in enumerate(CLASSES)}, f, ensure_ascii=False, indent=2)

print("Clases:", num_classes)


# =============================
# UTIL: mapping carpetas case-safe -> caracter real
# =============================
def folder_to_char(folder_name: str):
    # Digitos: D_0 ... D_9
    if folder_name.startswith("D_") and len(folder_name) == 3:
        return folder_name[2]

    # Upper: U_A ... U_Z
    if folder_name.startswith("U_") and len(folder_name) == 3:
        return folder_name[2]

    # Lower: L_a ... L_z
    if folder_name.startswith("L_") and len(folder_name) == 3:
        return folder_name[2]

    # Ñ / ñ (si usas estas clases)
    if folder_name == "U_Ñ":
        return "Ñ"
    if folder_name == "L_ñ":
        return "ñ"

    return None


# =============================
# CARGA IMPRESO (case-safe)
# =============================
def load_printed_dataset(root):
    X, y = [], []

    if not os.path.exists(root):
        print(f"⚠️ No existe: {root}")
        return np.zeros((0, 28, 28, 1), np.float32), np.zeros((0,), np.int32)

    for folder in os.listdir(root):
        folder_path = os.path.join(root, folder)
        if not os.path.isdir(folder_path):
            continue

        ch = folder_to_char(folder)
        if ch is None or ch not in char_to_idx:
            continue

        label = char_to_idx[ch]

        for name in os.listdir(folder_path):
            path = os.path.join(folder_path, name)
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue

            b = to_binary(img)
            x28 = normalize_28x28(b)

            if x28 is None or x28.shape != (28, 28):
                continue

            X.append(x28)
            y.append(label)

    if len(X) == 0:
        return np.zeros((0, 28, 28, 1), np.float32), np.zeros((0,), np.int32)

    X = np.array(X, dtype=np.float32)[..., None]
    y = np.array(y, dtype=np.int32)
    return X, y


# =============================
# CARGA MANUSCRITO (tu estructura numeros/mayusculas/minusculas)
# =============================
def load_manuscript_dataset(root):
    X, y = [], []

    if not os.path.exists(root):
        print(f"⚠️ No existe: {root}")
        return np.zeros((0, 28, 28, 1), np.float32), np.zeros((0,), np.int32)

    for group in ["numeros", "mayusculas", "minusculas"]:
        group_path = os.path.join(root, group)
        if not os.path.exists(group_path):
            continue

        for cls in os.listdir(group_path):
            cls_path = os.path.join(group_path, cls)
            if not os.path.isdir(cls_path):
                continue

            # cls debe ser exactamente "A" o "a" o "0"
            if cls not in char_to_idx:
                continue

            label = char_to_idx[cls]

            for name in os.listdir(cls_path):
                path = os.path.join(cls_path, name)
                img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue

                b = to_binary(img)
                x28 = normalize_28x28(b)
                if x28 is None or x28.shape != (28, 28):
                    continue

                X.append(x28)
                y.append(label)

    if len(X) == 0:
        return np.zeros((0, 28, 28, 1), np.float32), np.zeros((0,), np.int32)

    X = np.array(X, dtype=np.float32)[..., None]
    y = np.array(y, dtype=np.int32)
    return X, y


# =============================
# CARGA EMNIST (opcional, limitado)
# =============================
def load_emnist(max_samples=40000):
    """
    Carga EMNIST/byclass, lo filtra a nuestras clases, y lo limita a max_samples.
    Nota: EMNIST label->char depende del dataset; esto es aproximación práctica.
    """
    ds = tfds.load("emnist/byclass", split="train", as_supervised=True)

    # builder para convertir label a string
    builder = tfds.builder("emnist/byclass")
    builder.download_and_prepare()
    int2str = builder.info.features["label"].int2str

    X, y = [], []
    count = 0

    for img, label in tfds.as_numpy(ds):
        # img: (28,28,1) uint8, invertimos a fondo blanco tinta negra
        img = img.squeeze().astype(np.uint8)
        img = 255 - img  # EMNIST suele venir invertido

        # Normalizamos como el resto
        x28 = normalize_28x28(img)

        ch = int2str(int(label))  # puede dar 'A', 'a', '0', etc.

        if ch in char_to_idx:
            X.append(x28)
            y.append(char_to_idx[ch])
            count += 1
            if count >= max_samples:
                break

    if len(X) == 0:
        return np.zeros((0, 28, 28, 1), np.float32), np.zeros((0,), np.int32)

    X = np.array(X, dtype=np.float32)[..., None]
    y = np.array(y, dtype=np.int32)
    return X, y


# =============================
# ENTRENAMIENTO
# =============================
def main():
    print("Cargando manuscrito...")
    Xh, yh = load_manuscript_dataset(DATA_MANUS)
    print("Manuscrito:", Xh.shape)

    print("Cargando impreso (case-safe)...")
    Xp, yp = load_printed_dataset(DATA_PRINT)
    print("Impreso:", Xp.shape)

    print("Cargando EMNIST...")
    Xe, ye = load_emnist(max_samples=30000)
    print("EMNIST:", Xe.shape)

    # Concatenar solo los que tengan datos
    datasets_X, datasets_y = [], []
    for X_, y_ in [(Xh, yh), (Xp, yp), (Xe, ye)]:
        if X_.shape[0] > 0:
            datasets_X.append(X_)
            datasets_y.append(y_)

    if len(datasets_X) == 0:
        raise RuntimeError("No hay datos para entrenar (manuscrito/impreso/emnist vacíos).")

    X = np.concatenate(datasets_X, axis=0)
    y = np.concatenate(datasets_y, axis=0)

    # Shuffle
    perm = np.random.permutation(len(X))
    X, y = X[perm], y[perm]

    # Modelo
    model = build_ocr_model(num_classes)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            MODEL_OUT, monitor="val_loss", save_best_only=True, verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6, verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=6, restore_best_weights=True, verbose=1
        ),
    ]

    print("\nEntrenando desde cero...")
    model.fit(
        X, y,
        epochs=40,
        batch_size=128,
        validation_split=0.1,
        callbacks=callbacks
    )

    # Guardar (por si el último epoch era mejor que el checkpoint)
    model.save(MODEL_OUT)
    print("✅ Modelo entrenado y guardado:", MODEL_OUT)
    print("✅ label_map:", LABEL_MAP)


if __name__ == "__main__":
    main()
