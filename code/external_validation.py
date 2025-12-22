"""
generalization_external.py
==========================

Generalization experiment on an external pneumonia dataset.

The model is NOT retrained.
The best model from the paper (fold 1) is evaluated on a
balanced external dataset (500 NORMAL + 500 PNEUMONIA).

Paper version – frozen configuration.
"""

# ======================================================
# IMPORTS
# ======================================================
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'false'

import tensorflow as tf


import random
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, roc_curve
)

# ======================================================
# FIXED PAPER CONFIGURATION
# ======================================================

MODEL_NAME = "DenseNet121"
FOLD = 2                       # fixed fold used in the paper
SEED = 42

IMG_SIZE = (224, 224)
BATCH_SIZE = 64

MAIN_RESULTS_DIR = f"Results"
MODEL_FILENAME = f"best_fold{FOLD}.keras"

EXTERNAL_DATASET_DIR = "/Users/franciscoantoniogomezvela/Git-wrokspace/GitHub/proyectosTransferencia/pneumoniaCNN/ExternalDataset"
OUTPUT_DIR = f"{MAIN_RESULTS_DIR}/Generalization"

ARCHITECTURES = [
    "DenseNet121",
    "MobileNetV2",
    "ResNet50",
    "EfficientNetB0",
    "ConvNeXtTiny"
]

N_SAMPLES_PER_CLASS = 500

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ======================================================
# REPRODUCIBILITY
# ======================================================

tf.random.set_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

AUTOTUNE = tf.data.AUTOTUNE

# ======================================================
# DATA LOADING
# ======================================================

def build_dataframe(dataset_dir):
    data = []
    for cls in ["NORMAL", "PNEUMONIA"]:
        class_dir = os.path.join(dataset_dir, cls)
        for f in os.listdir(class_dir):
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                data.append([os.path.join(class_dir, f), cls])

    df = pd.DataFrame(data, columns=["filepath", "label"])
    df["label_id"] = df["label"].apply(
        lambda x: 1 if "pneu" in x.lower() else 0
    ).astype(int)
    return df

# ======================================================
# DATASET PIPELINE
# ======================================================

def _load_image(path, label):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IMG_SIZE)
    img = tf.cast(img, tf.float32) / 255.0
    return img, tf.cast(label, tf.float32)

def build_dataset(paths, labels):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.map(_load_image, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)
    return ds

# ======================================================
# BALANCED SUBSAMPLING (EXTERNAL DATASET)
# ======================================================

def balanced_external_split(df):
    df0 = df[df["label_id"] == 0]
    df1 = df[df["label_id"] == 1]

    df0 = df0.sample(
        n=min(N_SAMPLES_PER_CLASS, len(df0)),
        random_state=SEED
    )
    df1 = df1.sample(
        n=min(N_SAMPLES_PER_CLASS, len(df1)),
        random_state=SEED
    )

    df_balanced = pd.concat([df0, df1]).sample(
        frac=1, random_state=SEED
    )

    return df_balanced

# ======================================================
# GENERALIZATION EVALUATION
# ======================================================

def main():

    print("Loading external dataset...")
    df_ext = build_dataframe(EXTERNAL_DATASET_DIR)
    df_ext = balanced_external_split(df_ext)

    print(df_ext["label"].value_counts())

    ds_ext = build_dataset(
        df_ext["filepath"].values,
        df_ext["label_id"].values
    )
    
    print(f"\nExternal test dataset ready: {len(ds_ext)} samples")
    
    external_results = {}
    results_external = []
    
    for arch in ARCHITECTURES:
        model_path = os.path.join(MAIN_RESULTS_DIR, arch, MODEL_FILENAME)

        if not os.path.exists(model_path):
            print(f"[WARNING] Model not found: {model_path}")
            continue
    
        print("Loading frozen paper model...")
        model = tf.keras.models.load_model(model_path)

        y_true = np.concatenate([y.numpy() for _, y in ds_ext])
        y_prob = np.concatenate([
            model.predict(x).ravel() for x, _ in ds_ext
        ])
        y_pred = (y_prob >= 0.5).astype(int)

        # --- Metrics ---
        acc  = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec  = recall_score(y_true, y_pred)
        f1   = f1_score(y_true, y_pred)
        auc  = roc_auc_score(y_true, y_prob)

        # -------------------------
        # Store for ROC
        # -------------------------
        external_results[arch] = (y_true, y_prob)

        results_external.append({
            "Model": arch,
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1-score": f1,
            "AUC": auc
        })
    # -----------------------------------------------------
    # RESULTS TABLE
    # -----------------------------------------------------
    df_external_results = pd.DataFrame(results_external)
    print("\n===== EXTERNAL GENERALIZATION RESULTS =====")
    print(df_external_results.to_string(index=False))

    # Optional: save to disk
    df_external_results.to_csv(
        f"{OUTPUT_DIR}/external_validation_results.csv",
        index=False
    )

    print("\nExternal validation results saved.")
    plt.figure(figsize=(7,7))

    for model_name, (y_true, y_prob) in external_results.items():
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc = roc_auc_score(y_true, y_prob)
        plt.plot(fpr, tpr, label=f"{model_name} (AUC = {auc:.2f})")

    # Diagonal
    plt.plot([0,1], [0,1], 'k--', label="Random")

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves on External Validation Dataset")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/roc_external_validation.png", dpi=300)
    plt.show()
    
    
if __name__ == "__main__":
    # Force CPU
    with tf.device('/CPU:0'):
        # Tu código aquí
        main()
