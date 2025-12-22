"""
train_kfold.py
==============

CNN training and predictive performance evaluation using
5-fold stratified cross-validation with class-balanced undersampling.

Paper version – frozen configuration.
"""

# ======================================================
# IMPORTS
# ======================================================

import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, roc_curve
)

from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import (
    ResNet50, MobileNetV2, DenseNet121, EfficientNetB0,ConvNeXtTiny
)

# ======================================================
# FIXED PAPER CONFIGURATION
# ======================================================

MODEL_NAME   = "DenseNet121" # Model architecture- Change here to try others ResNet50, MobileNetV2, DenseNet121, EfficientNetB0,ConvNeXtTiny
DATASET_DIR = "/Users/franciscoantoniogomezvela/Git-wrokspace/GitHub/proyectosTransferencia/pneumoniaCNN/Images"   # <-- change this for change the dataset path
OUTPUT_DIR  = f"Results/{MODEL_NAME}"

IMG_SIZE    = (224, 224)
BATCH_SIZE = 64
EPOCHS     = 25
N_SPLITS   = 5
SEED       = 42

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
    for cls in sorted(os.listdir(dataset_dir)):
        class_dir = os.path.join(dataset_dir, cls)
        if not os.path.isdir(class_dir):
            continue
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

def augment_fn(img, label):
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_brightness(img, 0.05)
    img = tf.image.random_contrast(img, 0.9, 1.1)
    return img, label

def build_dataset(paths, labels, train=True):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.map(_load_image, num_parallel_calls=AUTOTUNE)
    if train:
        ds = ds.shuffle(512, seed=SEED)
        ds = ds.map(augment_fn, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)
    return ds

# ======================================================
# MODEL
# ======================================================

def build_model(architecture="ResNet50", lr=1e-4):
    """
    Creates a CNN classifier using a selectable backbone.
    Easily extensible to new architectures.
    """
    print(f"Building model with architecture: {architecture}")
    # Default strategy for single GPU and CPU
    strategy = tf.distribute.get_strategy()
    with strategy.scope():
        if architecture == "ResNet50":
            base = ResNet50(weights="imagenet", include_top=False, input_shape=(*IMG_SIZE,3))
        elif architecture == "MobileNetV2":
            base = MobileNetV2(weights="imagenet", include_top=False, input_shape=(*IMG_SIZE,3))
        elif architecture == "DenseNet121":
            base = DenseNet121(weights="imagenet", include_top=False, input_shape=(*IMG_SIZE,3))
        elif architecture == "EfficientNetB0":
            base = EfficientNetB0(weights="imagenet", include_top=False, input_shape=(*IMG_SIZE,3))
        elif architecture == "ConvNeXtTiny":
            base = ConvNeXtTiny(weights="imagenet", include_top=False, input_shape=(*IMG_SIZE, 3))
        else:
            raise ValueError("Unknown architecture")
        # ---- CLASSIFICATION HEAD ----
        base.trainable = False  # Freeze base

        x = layers.GlobalAveragePooling2D()(base.output)
        x = layers.Dense(512, activation="relu")(x)
        x = layers.Dropout(0.4)(x)
        x = layers.Dense(256, activation="relu")(x)
        x = layers.Dropout(0.3)(x)
        outputs = layers.Dense(1, activation="sigmoid")(x)

        model = models.Model(inputs=base.input, outputs=outputs)
        model.compile(optimizer=Adam(lr), loss="binary_crossentropy", metrics=["accuracy"])

    return model

# ======================================================
# TRAINING
# ======================================================

def main():

    df = build_dataframe(DATASET_DIR)
    image_paths = df["filepath"].values
    labels = df["label_id"].values

    skf = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=SEED
    )

    results = []

    for fold, (tr, va) in enumerate(skf.split(image_paths, labels), 1):
        print(f"\n===== FOLD {fold} =====")

        # ----- undersampling -----
        y_tr = labels[tr]
        idx0 = tr[y_tr == 0]
        idx1 = tr[y_tr == 1]
        n = min(len(idx0), len(idx1))

        balanced_idx = np.concatenate([
            np.random.choice(idx0, n, replace=False),
            np.random.choice(idx1, n, replace=False)
        ])
        np.random.shuffle(balanced_idx)

        train_ds = build_dataset(image_paths[balanced_idx], labels[balanced_idx], True)
        val_ds   = build_dataset(image_paths[va], labels[va], False)

        model = build_model(architecture=MODEL_NAME)

        ckpt = f"{OUTPUT_DIR}/best_fold{fold}.keras"
        model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=EPOCHS,
            callbacks=[
                tf.keras.callbacks.ModelCheckpoint(
                    ckpt, save_best_only=True, monitor="val_loss"
                ),
                tf.keras.callbacks.EarlyStopping(
                    patience=10, restore_best_weights=True
                )
            ],
            verbose=1
        )

        y_true = np.concatenate([y.numpy() for _, y in val_ds])
        y_prob = np.concatenate([model.predict(x).ravel() for x, _ in val_ds])
        y_pred = (y_prob >= 0.5).astype(int)

        results.append({
            "fold": fold,
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred),
            "recall": recall_score(y_true, y_pred),
            "f1": f1_score(y_true, y_pred),
            "auc": roc_auc_score(y_true, y_prob)
        })
        # ------------------------------------------------------
        #                ROC CURVE PER FOLD
        # ------------------------------------------------------
        fpr, tpr, _ = roc_curve(y_true, y_prob)

        plt.figure(figsize=(6,6))
        plt.plot(fpr, tpr, label=f"Fold {fold} (AUC={results[-1]['auc']:.3f})")
        plt.plot([0,1],[0,1],"k--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC Curve – Fold {fold}")
        plt.legend()
        plt.savefig(f"{OUTPUT_DIR}/roc_fold{fold}.png")
        plt.close()

         # Save predictions for this fold
        pd.DataFrame({
            "filepath": image_paths[va],
            "y_true": y_true,
            "y_prob": y_prob,
            "y_pred": y_pred
        }).to_csv(f"{OUTPUT_DIR}/predictions_fold{fold}.csv", index=False)

    # ------------------------------------------------------
    #                 SAVE SUMMARY STATISTICS
    # ------------------------------------------------------

    df_res = pd.DataFrame(results)
    df_res.loc["mean"] = df_res.mean()
    df_res.loc["std"]  = df_res.std()
    df_res.to_csv(f"{OUTPUT_DIR}/kfold_results.csv", index=False)
    print("\n===== FINAL SUMMARY =====")
    print(df_res)
        
    # ------------------------------------------------------
    #             PLOT AVERAGE ROC CURVES
    # ------------------------------------------------------
    plt.figure(figsize=(7,7))

    for fold in range(1, N_SPLITS+1):
        pred_df = pd.read_csv(f"{OUTPUT_DIR}/predictions_fold{fold}.csv")
        fpr, tpr, _ = roc_curve(pred_df["y_true"], pred_df["y_prob"])
        auc = roc_auc_score(pred_df["y_true"], pred_df["y_prob"])
        plt.plot(fpr, tpr, label=f"Fold {fold} (AUC={auc:.3f})")

    plt.plot([0,1],[0,1],"k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves Across All Folds")
    plt.legend()
    plt.savefig(f"{OUTPUT_DIR}/roc_all_folds.png")
    plt.close()

    print(f"\nAll results saved to: {OUTPUT_DIR}")   

if __name__ == "__main__":
    main()
