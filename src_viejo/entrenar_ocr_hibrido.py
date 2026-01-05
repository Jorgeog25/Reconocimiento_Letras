import os, json
import cv2
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
from utils_preprocess import to_binary, normalize_28x28

# =============================
# CONFIGURACIÓN GENERAL
# =============================

EPOCHS = 200                # épocas por sesión
BATCH_SIZE = 128
VAL_SPLIT = 0.1             # 10% validación
MODEL_PATH = "models/ocr_cnn.keras"

AUTOTUNE = tf.data.AUTOTUNE

CLASSES = list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZÑabcdefghijklmnopqrstuvwxyzñ")
char_to_idx = {c: i for i, c in enumerate(CLASSES)}

# =============================
# DATASET LOCAL (MANUSCRITO)
# =============================

def load_local_dataset(root):
    X, y = [], []

    for group in ["numeros", "mayusculas", "minusculas"]:
        group_path = os.path.join(root, group)

        for cls in os.listdir(group_path):
            if cls not in char_to_idx:
                continue

            label = char_to_idx[cls]
            cls_path = os.path.join(group_path, cls)

            for img_name in os.listdir(cls_path):
                img_path = os.path.join(cls_path, img_name)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue

                binary = to_binary(img)
                norm = normalize_28x28(binary)
                X.append(norm)
                y.append(label)

    X = np.array(X, dtype=np.float32)[..., None]
    y = np.array(y, dtype=np.int32)
    return X, y

# =============================
# MNIST (DÍGITOS)
# =============================

def load_mnist():
    (Xtr, ytr), (Xte, yte) = mnist.load_data()
    X = np.concatenate([Xtr, Xte])
    y = np.concatenate([ytr, yte])

    X = 1.0 - (X.astype(np.float32) / 255.0)
    X = X[..., None]
    return X, y

# =============================
# EMNIST (LETRAS)
# =============================

def load_emnist():
    ds = tfds.load("emnist/byclass", split="train", as_supervised=True)

    X, y = [], []

    for img, label in tfds.as_numpy(ds):
        char = chr(label + 48)
        if char not in char_to_idx:
            continue

        img = img.astype(np.float32)
        img = 1.0 - (img / 255.0)  # normalización
        X.append(img)              # YA es (28,28,1)
        y.append(char_to_idx[char])

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)
    return X, y

# =============================
# MODELO CNN
# =============================

def build_model():
    model = models.Sequential([
        layers.Input(shape=(28,28,1)),

        layers.Conv2D(32, 3, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),

        layers.Conv2D(64, 3, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),

        layers.Conv2D(128, 3, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),

        layers.Flatten(),
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.4),

        layers.Dense(len(CLASSES), activation="softmax")
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

# =============================
# DATA AUGMENTATION
# =============================

augment = tf.keras.Sequential([
    layers.RandomRotation(0.1),
    layers.RandomTranslation(0.1, 0.1),
    layers.RandomZoom(0.15),
    layers.GaussianNoise(0.03),
])

# =============================
# MAIN
# =============================

print("Cargando datasets...")
X_local, y_local = load_local_dataset("dataset")
X_mnist, y_mnist = load_mnist()
X_emnist, y_emnist = load_emnist()

X = np.concatenate([X_local, X_mnist, X_emnist])
y = np.concatenate([y_local, y_mnist, y_emnist])

print("Total muestras:", X.shape[0])

# =============================
# TRAIN / VALIDATION SPLIT
# =============================

indices = np.random.permutation(len(X))
X = X[indices]
y = y[indices]

val_size = int(len(X) * VAL_SPLIT)

X_val, y_val = X[:val_size], y[:val_size]
X_train, y_train = X[val_size:], y[val_size:]

ds_train = tf.data.Dataset.from_tensor_slices((X_train, y_train))
ds_train = ds_train.shuffle(50000)
ds_train = ds_train.map(
    lambda x,y: (augment(x, training=True), y),
    num_parallel_calls=AUTOTUNE
)
ds_train = ds_train.batch(BATCH_SIZE).prefetch(AUTOTUNE)

ds_val = tf.data.Dataset.from_tensor_slices((X_val, y_val))
ds_val = ds_val.batch(BATCH_SIZE).prefetch(AUTOTUNE)

# =============================
# MODELO (CREAR O CONTINUAR)
# =============================

os.makedirs("models", exist_ok=True)

if os.path.exists(MODEL_PATH):
    print("Modelo existente encontrado → continuando entrenamiento")
    model = tf.keras.models.load_model(MODEL_PATH)
else:
    print("Creando modelo nuevo")
    model = build_model()

# =============================
# CALLBACKS
# =============================

callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        MODEL_PATH,
        monitor="val_loss",
        save_best_only=True,
        verbose=1
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=5,
        min_lr=1e-5,
        verbose=1
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=12,
        restore_best_weights=True,
        verbose=1
    )
]

# =============================
# ENTRENAMIENTO
# =============================

print("Entrenando...")
model.fit(
    ds_train,
    validation_data=ds_val,
    epochs=EPOCHS,
    callbacks=callbacks
)

# =============================
# GUARDAR MAPA DE ETIQUETAS
# =============================

with open("models/label_map.json", "w", encoding="utf-8") as f:
    json.dump({str(i): c for i,c in enumerate(CLASSES)}, f, ensure_ascii=False, indent=2)

print("Entrenamiento finalizado correctamente")
