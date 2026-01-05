import os, json
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from utils_preprocess import to_binary, normalize_28x28

MODEL_PATH = "models/ocr_cnn.keras"
LABEL_PATH = "models/label_map.json"

EPOCHS = 25
BATCH_SIZE = 128
VAL_SPLIT = 0.1
AUTOTUNE = tf.data.AUTOTUNE

# =============================
# CARGAR LABEL MAP EXISTENTE (CLAVE)
# =============================
if not os.path.exists(LABEL_PATH):
    raise FileNotFoundError(f"No existe {LABEL_PATH}. Necesito el label_map del modelo original.")

with open(LABEL_PATH, "r", encoding="utf-8") as f:
    idx_to_char = {int(k): v for k, v in json.load(f).items()}

# CLASSES en el orden exacto del modelo
CLASSES = [idx_to_char[i] for i in range(len(idx_to_char))]
char_to_idx = {c: i for i, c in enumerate(CLASSES)}

print("Clases del modelo:", len(CLASSES))
print("Ejemplo primeras 10:", CLASSES[:10])

# =============================
# CARGA DATASET: carpeta por clase
# =============================
def load_folder_dataset(root):
    X, y = [], []
    if not os.path.exists(root):
        print(f"⚠️ No existe {root}")
        return np.zeros((0,28,28,1), np.float32), np.zeros((0,), np.int32)

    for cls in os.listdir(root):
        cls_path = os.path.join(root, cls)
        if not os.path.isdir(cls_path):
            continue
        if cls not in char_to_idx:
            # carpeta que el modelo no conoce
            continue

        label = char_to_idx[cls]

        for name in os.listdir(cls_path):
            path = os.path.join(cls_path, name)
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue

            binary = to_binary(img)
            x28 = normalize_28x28(binary)  # float 0..1 (tinta=1)
            X.append(x28)
            y.append(label)

    if len(X) == 0:
        return np.zeros((0,28,28,1), np.float32), np.zeros((0,), np.int32)

    X = np.array(X, dtype=np.float32)[..., None]
    y = np.array(y, dtype=np.int32)
    return X, y

def load_handwritten_dataset(root="dataset"):
    X, y = [], []
    for group in ["numeros", "mayusculas", "minusculas"]:
        group_path = os.path.join(root, group)
        if not os.path.exists(group_path):
            continue

        for cls in os.listdir(group_path):
            cls_path = os.path.join(group_path, cls)
            if not os.path.isdir(cls_path):
                continue
            if cls not in char_to_idx:
                continue

            label = char_to_idx[cls]

            for name in os.listdir(cls_path):
                path = os.path.join(cls_path, name)
                img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue

                binary = to_binary(img)
                x28 = normalize_28x28(binary)
                X.append(x28)
                y.append(label)

    if len(X) == 0:
        return np.zeros((0,28,28,1), np.float32), np.zeros((0,), np.int32)

    X = np.array(X, dtype=np.float32)[..., None]
    y = np.array(y, dtype=np.int32)
    return X, y

# =============================
# AUGMENTATION LIGERA (NO agresiva en fine-tuning)
# =============================
augment = tf.keras.Sequential([
    layers.RandomRotation(0.05),
    layers.RandomTranslation(0.05, 0.05),
    layers.RandomZoom(0.05),
])

print("Cargando dataset impreso...")
X_print, y_print = load_folder_dataset("dataset_impreso")
print("Impreso:", X_print.shape)

print("Cargando dataset manuscrito...")
X_hand, y_hand = load_handwritten_dataset("dataset")
print("Manuscrito:", X_hand.shape)

# Si impreso es enorme, puedes reducirlo (opcional)
# Ej: quedarte con 200 por clase para no desbalancear
# (ahora lo dejamos tal cual)

# Mezcla
X = np.concatenate([X_print, X_hand], axis=0)
y = np.concatenate([y_print, y_hand], axis=0)

if len(X) == 0:
    raise RuntimeError("No hay datos cargados. Revisa rutas dataset_impreso/ y dataset/.")

# Shuffle
perm = np.random.permutation(len(X))
X, y = X[perm], y[perm]

# Split train/val
val_size = int(len(X) * VAL_SPLIT)
X_val, y_val = X[:val_size], y[:val_size]
X_train, y_train = X[val_size:], y[val_size:]

ds_train = tf.data.Dataset.from_tensor_slices((X_train, y_train))
ds_train = ds_train.shuffle(50000)
ds_train = ds_train.map(lambda x, yy: (augment(x, training=True), yy), num_parallel_calls=AUTOTUNE)
ds_train = ds_train.batch(BATCH_SIZE).prefetch(AUTOTUNE)

ds_val = tf.data.Dataset.from_tensor_slices((X_val, y_val))
ds_val = ds_val.batch(BATCH_SIZE).prefetch(AUTOTUNE)

# =============================
# CARGAR MODELO Y FINE-TUNE CON LR BAJO
# =============================
print("Cargando modelo existente...")
model = tf.keras.models.load_model(MODEL_PATH)

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),  # LR bajo pero no microscópico
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# Guardar un backup antes de tocar nada
os.makedirs("models", exist_ok=True)
backup_path = "models/ocr_cnn_backup_before_finetune.keras"
try:
    model.save(backup_path)
    print("✅ Backup guardado:", backup_path)
except Exception as e:
    print("⚠️ No pude guardar backup:", e)

callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        MODEL_PATH, monitor="val_loss", save_best_only=True, verbose=1
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6, verbose=1
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=7, restore_best_weights=True, verbose=1
    )
]

print("Fine-tuning...")
model.fit(ds_train, validation_data=ds_val, epochs=EPOCHS, callbacks=callbacks)

print("✅ Fine-tuning terminado. Modelo guardado en:", MODEL_PATH)
print("✅ label_map NO se ha modificado (se mantiene consistente).")
