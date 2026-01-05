import os, json, cv2
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

from src.preprocessing.preprocess import to_binary, normalize_28x28
from src.training.build_model import build_ocr_model


# =============================
# CONFIG
# =============================
DATA_MANUS = "src/data/manuscrito"
DATA_PRINT = "src/data/impreso"
MODEL_OUT = "src/models/ocr_cnn.keras"
LABEL_MAP = "src/models/label_map.json"

EPOCHS = 40
BATCH = 128

# =============================
# CLASES (definitivas)
# =============================
CLASSES = list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz")
char_to_idx = {c: i for i, c in enumerate(CLASSES)}

os.makedirs("models", exist_ok=True)
with open(LABEL_MAP, "w") as f:
    json.dump({i: c for i, c in enumerate(CLASSES)}, f)

# =============================
# CARGA DATASET DESDE CARPETAS
# =============================
def load_folder_dataset(root):
    X, y = [], []
    for cls in os.listdir(root):
        if cls not in char_to_idx:
            continue
        for name in os.listdir(os.path.join(root, cls)):
            path = os.path.join(root, cls, name)
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            b = to_binary(img)
            x28 = normalize_28x28(b)
            X.append(x28)
            y.append(char_to_idx[cls])
    return np.array(X)[..., None], np.array(y)

# =============================
# EMNIST
# =============================
def load_emnist():
    ds = tfds.load("emnist/byclass", split="train", as_supervised=True)
    X, y = [], []
    for img, label in tfds.as_numpy(ds.take(60000)):
        img = img.squeeze()
        img = 255 - img
        img = normalize_28x28(img)
        ch = tfds.builder("emnist/byclass").info.features["label"].int2str(label)
        if ch in char_to_idx:
            X.append(img)
            y.append(char_to_idx[ch])
    return np.array(X)[..., None], np.array(y)

# =============================
# MAIN
# =============================
def load_manuscript_dataset(root):
    X, y = [], []

    for group in ["numeros", "mayusculas", "minusculas"]:
        group_path = os.path.join(root, group)
        if not os.path.exists(group_path):
            continue

        for cls in os.listdir(group_path):
            if cls not in char_to_idx:
                continue

            cls_path = os.path.join(group_path, cls)
            if not os.path.isdir(cls_path):
                continue

            for name in os.listdir(cls_path):
                path = os.path.join(cls_path, name)
                img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue

                b = to_binary(img)

                # Si la imagen quedó casi vacía, saltar
                if b is None or b.size == 0:
                    continue

                x28 = normalize_28x28(b)
                # Protección extra
                if x28 is None or x28.shape != (28, 28):
                    continue

                X.append(x28)
                y.append(char_to_idx[cls])

    if len(X) == 0:
        return np.zeros((0, 28, 28, 1), dtype=np.float32), np.zeros((0,), dtype=np.int32)

    X = np.array(X, dtype=np.float32)[..., None]
    y = np.array(y, dtype=np.int32)
    return X, y


print("Cargando manuscrito...")
Xh, yh = load_manuscript_dataset(DATA_MANUS)
print("Manuscrito:", Xh.shape)


print("Cargando impreso...")
Xp, yp = load_folder_dataset(DATA_PRINT)
print("Impreso:", Xp.shape)

print("Cargando EMNIST...")
Xe, ye = load_emnist()
print("EMNIST:", Xe.shape)

datasets_X = []
datasets_y = []

if Xh.shape[0] > 0:
    datasets_X.append(Xh)
    datasets_y.append(yh)

if Xp.shape[0] > 0:
    datasets_X.append(Xp)
    datasets_y.append(yp)

if Xe.shape[0] > 0:
    datasets_X.append(Xe)
    datasets_y.append(ye)

X = np.concatenate(datasets_X, axis=0)
y = np.concatenate(datasets_y, axis=0)


perm = np.random.permutation(len(X))
X, y = X[perm], y[perm]

model = build_ocr_model(len(CLASSES))
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.fit(
    X, y,
    epochs=EPOCHS,
    batch_size=BATCH,
    validation_split=0.1,
    callbacks=[
        tf.keras.callbacks.ReduceLROnPlateau(patience=3),
        tf.keras.callbacks.EarlyStopping(patience=6, restore_best_weights=True)
    ]
)

model.save(MODEL_OUT)
print("✅ Modelo entrenado y guardado:", MODEL_OUT)
