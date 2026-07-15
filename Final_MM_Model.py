# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

# Use the kagglehub client library to attach Kaggle resources like competitions, datasets, and models to your session
# Learn more about kagglehub: https://github.com/Kaggle/kagglehub/blob/main/README.md

import kagglehub
# kagglehub.dataset_download('<owner>/<dataset-slug>')

# ---------------------------------------------------------------------

import os
import random
import numpy as np
from pathlib import Path

import kagglehub
from PIL import Image

import tensorflow as tf
from tensorflow.keras import layers, models

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# ---------------------------------------------------------------------

SEED = 42
VALID_EXTS = {".png", ".jpg", ".jpeg"}

CLASS_NAMES = ["Healthy", "ALL", "AML", "CLL", "CML"]
NUM_CLASSES = len(CLASS_NAMES)
class_to_idx = {name: idx for idx, name in enumerate(CLASS_NAMES)}
print("Class mapping:", class_to_idx)

# Fixed image size — chosen instead of the dataset's true average (~224x224)
# because loading the full dataset into memory at 224x224 was exhausting
# Kaggle's RAM. 128x128 keeps enough detail for cell morphology while
# staying memory-safe when combined with the tf.data pipeline in Phase 4.
IMG_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 5

ALL_KEEP_COUNT = 4500          # downsample ALL to this many images
HEALTHY_B_KEEP_COUNT = 1500    # downsample unclesamulus healthy source

random.seed(SEED)
tf.random.set_seed(SEED)

# ---------------------------------------------------------------------

# =====================================================================
# PHASE 2 — DATASET COLLECTION
# (gathers file paths + labels from every source; no images loaded yet)
# =====================================================================
def collect_images(root_dir, include_keywords=None, exclude_keywords=None):
    """Walk a folder tree and return image paths, optionally filtered by
    keywords found in the folder path (case-insensitive)."""
    collected = []
    for root, _, files in os.walk(root_dir):
        folder = root.lower()

        if include_keywords is not None:
            if not any(k in folder for k in include_keywords):
                continue
        if exclude_keywords is not None:
            if any(k in folder for k in exclude_keywords):
                continue

        for f in files:
            if f.startswith("."):
                continue  # skip hidden/junk files like .DS_Store, ._xxx.jpg
            if Path(f).suffix.lower() in VALID_EXTS:
                collected.append(os.path.join(root, f))
    return collected

def collect_dataset_paths():
    """Runs every dataset source and returns combined (paths, labels)."""

    # ---- HEALTHY — Source A: kylewang1999/pbc-dataset — REMOVED ----
    # (dropped: caused severe class imbalance)
    paths_healthy_a, labels_healthy_a = [], []

    # ---- HEALTHY — Source B: unclesamulus/blood-cells-image-dataset ----
    path_healthy_b = kagglehub.dataset_download("unclesamulus/blood-cells-image-dataset")
    print("Healthy source B path:", path_healthy_b)

    relevant_wbc_keywords = [
        "neutrophil", "eosinophil", "basophil",
        "lymphocyte", "monocyte", "granulocyte"
    ]
    paths_healthy_b_all = collect_images(
        path_healthy_b,
        include_keywords=relevant_wbc_keywords,
        exclude_keywords=["erythroblast", "platelet", "thrombocyte"]
    )
    print(f"Healthy (unclesamulus, relevant WBC classes, before downsampling) -> {len(paths_healthy_b_all)} images")

    random.shuffle(paths_healthy_b_all)
    paths_healthy_b = paths_healthy_b_all[:HEALTHY_B_KEEP_COUNT]
    labels_healthy_b = [class_to_idx["Healthy"]] * len(paths_healthy_b)
    print(f"Healthy (unclesamulus, downsampled) -> {len(paths_healthy_b)} images")

    # ---- HEALTHY — Source C: mehradaria/leukemia (benign only) ----
    path_mehradaria = kagglehub.dataset_download("mehradaria/leukemia")
    print("Mehradaria dataset path:", path_mehradaria)

    paths_healthy_c = collect_images(path_mehradaria, include_keywords=["benign"])
    labels_healthy_c = [class_to_idx["Healthy"]] * len(paths_healthy_c)
    print(f"Healthy (mehradaria benign) -> {len(paths_healthy_c)} images")

    # ---- ALL — Source A: mehradaria/leukemia (early + pre + pro) ----
    paths_all_a = collect_images(path_mehradaria, include_keywords=["early", "pre", "pro"])
    print(f"ALL (mehradaria early/pre/pro) -> {len(paths_all_a)} images")

    # ---- ALL — Source B: mohammadamireshraghi (benign excluded) ----
    path_extra_all = kagglehub.dataset_download("mohammadamireshraghi/blood-cell-cancer-all-4class")
    print("Extra ALL dataset path:", path_extra_all)

    paths_all_b = collect_images(
        path_extra_all, include_keywords=["early", "pre", "pro"], exclude_keywords=["benign"]
    )
    print(f"ALL (mohammadamireshraghi malignant Pre-B/Pro-B/early Pre-B) -> {len(paths_all_b)} images")

    # ---- AML / CLL / CML / Healthy(D) / ALL(C) — priyaadharshinivs062 ----
    path_leuk_types = kagglehub.dataset_download("priyaadharshinivs062/leukemia-dataset")
    print("Leukemia types dataset path:", path_leuk_types)

    paths_aml = collect_images(path_leuk_types, include_keywords=["aml"])
    labels_aml = [class_to_idx["AML"]] * len(paths_aml)
    print(f"AML -> {len(paths_aml)} images")

    paths_cll = collect_images(path_leuk_types, include_keywords=["cll"])
    labels_cll = [class_to_idx["CLL"]] * len(paths_cll)
    print(f"CLL -> {len(paths_cll)} images")

    paths_cml = collect_images(path_leuk_types, include_keywords=["cml"])
    labels_cml = [class_to_idx["CML"]] * len(paths_cml)
    print(f"CML -> {len(paths_cml)} images")

    paths_healthy_d = collect_images(path_leuk_types, include_keywords=["h train", "h test"])
    labels_healthy_d = [class_to_idx["Healthy"]] * len(paths_healthy_d)
    print(f"Healthy (leukemia-dataset h train/test) -> {len(paths_healthy_d)} images")

    paths_all_c = collect_images(path_leuk_types, include_keywords=["all train", "all test"])
    print(f"ALL (leukemia-dataset all train/test) -> {len(paths_all_c)} images")

    # ---- Combine + downsample ALL ----
    paths_all_combined = paths_all_a + paths_all_b + paths_all_c
    print(f"\nALL combined (before downsampling) -> {len(paths_all_combined)} images")

    random.shuffle(paths_all_combined)
    paths_all_final = paths_all_combined[:ALL_KEEP_COUNT]
    labels_all_final = [class_to_idx["ALL"]] * len(paths_all_final)
    print(f"ALL (downsampled) -> {len(paths_all_final)} images")

    # ---- Final concatenation ----
    all_paths = (
        paths_healthy_a + paths_healthy_b + paths_healthy_c + paths_healthy_d
        + paths_all_final
        + paths_aml + paths_cll + paths_cml
    )
    all_labels = (
        labels_healthy_a + labels_healthy_b + labels_healthy_c + labels_healthy_d
        + labels_all_final
        + labels_aml + labels_cll + labels_cml
    )

    print("\nCombined class counts:")
    for name, idx in class_to_idx.items():
        print(f"  {name}: {all_labels.count(idx)}")
    print(f"Combined total images: {len(all_paths)}")

    return all_paths, all_labels

paths, labels = collect_dataset_paths()

# ---------------------------------------------------------------------

# =====================================================================
# PHASE 3 — CLEAN & SPLIT
# (drop unreadable files, then split into train/val/test file lists —
#  still just paths + labels, nothing loaded into memory yet)
# =====================================================================
def filter_unreadable(paths, labels):
    good_paths, good_labels, bad_files = [], [], []
    for p, l in zip(paths, labels):
        try:
            with Image.open(p) as img:
                img.verify()
            good_paths.append(p)
            good_labels.append(l)
        except Exception:
            bad_files.append(p)
    if bad_files:
        print(f"Skipping {len(bad_files)} unreadable files (sample): {bad_files[:5]}")
    return good_paths, good_labels

paths, labels = filter_unreadable(paths, labels)

train_paths, test_paths, train_labels, test_labels = train_test_split(
    paths, labels, test_size=0.2, stratify=labels, random_state=SEED
)
train_paths, val_paths, train_labels, val_labels = train_test_split(
    train_paths, train_labels, test_size=0.2, stratify=train_labels, random_state=SEED
)

print(f"\nTrain: {len(train_paths)}  Val: {len(val_paths)}  Test: {len(test_paths)}")

# ---------------------------------------------------------------------

def load_and_preprocess(path, label):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IMG_SIZE)
    img = img / 255.0
    label = tf.one_hot(label, NUM_CLASSES)
    return img, label

def make_dataset(paths, labels, shuffle=False):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(paths), seed=SEED)
    ds = ds.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(BATCH_SIZE)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds

train_ds = make_dataset(train_paths, train_labels, shuffle=True)
val_ds = make_dataset(val_paths, val_labels, shuffle=False)
test_ds = make_dataset(test_paths, test_labels, shuffle=False)

# ---------------------------------------------------------------------

# =====================================================================
# PHASE 5 — CLASS WEIGHTS
# (computed from train labels so under-represented classes aren't ignored)
# =====================================================================
class_counts = {i: train_labels.count(i) for i in range(NUM_CLASSES)}
total_train = len(train_labels)
class_weight = {
    i: total_train / (NUM_CLASSES * count) for i, count in class_counts.items() if count > 0
}
print("Class weights:", {CLASS_NAMES[i]: round(w, 2) for i, w in class_weight.items()})

# ---------------------------------------------------------------------

def build_model():
    model = models.Sequential([
        layers.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
        layers.Conv2D(32, 3, activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),
        # layers.Conv2D(128, 3, activation="relu"),
        # layers.BatchNormalization(),
        # layers.MaxPooling2D(),
        # layers.Conv2D(256, 3, activation="relu"),
        # layers.BatchNormalization(),
        # layers.MaxPooling2D(),
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.4),
        layers.Dense(NUM_CLASSES, activation="softmax"),
    ])
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model

model = build_model()
model.summary()

# ---------------------------------------------------------------------

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    class_weight=class_weight,
)

# ---------------------------------------------------------------------

test_loss, test_acc = model.evaluate(test_ds)
print(f"\nTest Accuracy: {test_acc:.4f}")

y_true, y_pred = [], []
for images, batch_labels in test_ds:
    preds = model.predict(images, verbose=0)
    y_true.extend(np.argmax(batch_labels.numpy(), axis=1))
    y_pred.extend(np.argmax(preds, axis=1))

print("\n================== CLASSIFICATION REPORT ==================")
print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))
print("=============================================================\n")

print("Confusion Matrix:")
print(confusion_matrix(y_true, y_pred))

# ---------------------------------------------------------------------

model.save("model.keras")

# ---------------------------------------------------------------------

checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath="/kaggle/working/checkpoints/model_epoch_{epoch:02d}.keras",
    save_weights_only=False,
    save_best_only=False,
    verbose=1
)

history = model.fit(
    train_ds,
    validation_data=val_ds,
    initial_epoch=5,   # epochs 1–5 already completed
    epochs=10,         # trains epochs 6–10
    class_weight=class_weight,
    callbacks=[checkpoint_callback]
)

# ---------------------------------------------------------------------

import os
os.listdir("/kaggle/working/")
