"""
train_kfold.py
==============

CNN training and predictive performance evaluation using
5-fold stratified cross-validation with class-balanced undersampling.

Updated for JCR Review:
- Hyperparameters read from config.json.
- Evaluates across multiple seeds for robustness.
- Saves exact data splits for reproducibility.
"""

# ======================================================
# IMPORTS
# ======================================================

import os
import json
import random
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, roc_curve,
    precision_recall_curve, auc, matthews_corrcoef
)

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import (
    ResNet50, MobileNetV2, DenseNet121, EfficientNetB0, ConvNeXtTiny, VGG16
)

# ======================================================
# CONFIGURATION
# ======================================================
config_path = os.path.join(os.path.dirname(__file__), "config.json")
with open(config_path, 'r') as f:
    config = json.load(f)

MODEL_NAME   = config["model_name"]
DATASET_DIR  = config["dataset_dir"]
OUTPUT_BASE_DIR = os.path.join(config["output_dir"], MODEL_NAME)

IMG_SIZE    = tuple(config["img_size"])
BATCH_SIZE = config["batch_size"]
EPOCHS_PHASE1 = config.get("epochs_phase1", 15)
EPOCHS_PHASE2 = config.get("epochs_phase2", 10)
N_SPLITS   = config["n_splits"]
SEEDS      = config["seeds"]
HP         = config["hyperparameters"]

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

def build_dataset(paths, labels, train=True, seed=42):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.map(_load_image, num_parallel_calls=AUTOTUNE)
    if train:
        ds = ds.shuffle(512, seed=seed)
        ds = ds.map(augment_fn, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)
    return ds

# ======================================================
# MODEL
# ======================================================

def build_model(architecture="ResNet50", hp=None):
    """
    Creates a CNN classifier using hyperparameters defined in config.
    """
    if hp is None:
        hp = HP
        
    print(f"Building model with architecture: {architecture}")
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
        elif architecture == "VGG16":
            base = VGG16(weights="imagenet", include_top=False, input_shape=(*IMG_SIZE, 3))
        else:
            raise ValueError("Unknown architecture")
            
        # ---- CLASSIFICATION HEAD ----
        base.trainable = True

        x = layers.GlobalAveragePooling2D()(base.output)
        x = layers.Dense(hp["dense_1_units"], activation="relu")(x)
        x = layers.Dropout(hp["dropout_1_rate"])(x)
        x = layers.Dense(hp["dense_2_units"], activation="relu")(x)
        x = layers.Dropout(hp["dropout_2_rate"])(x)
        outputs = layers.Dense(1, activation="sigmoid")(x)

        model = models.Model(inputs=base.input, outputs=outputs)
        model.compile(optimizer=Adam(hp["learning_rate"]), loss="binary_crossentropy", metrics=["accuracy"])

    return model, base

# ======================================================
# TRAINING
# ======================================================

def main():

    df = build_dataframe(DATASET_DIR)
    image_paths = df["filepath"].values
    labels = df["label_id"].values

    all_seed_results = []

    for seed in SEEDS:
        print(f"\n======================================")
        print(f"      STARTING EVALUATION - SEED {seed}")
        print(f"======================================")
        
        # Set reproducibility per seed
        tf.random.set_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        
        OUTPUT_DIR = os.path.join(OUTPUT_BASE_DIR, f"seed_{seed}")
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
        results = []

        for fold, (tr, va) in enumerate(skf.split(image_paths, labels), 1):
            print(f"\n===== SEED {seed} | FOLD {fold} =====")

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
            
            # --- SAVE EXACT TRAINING DATA SPLIT FOR REPRODUCIBILITY ---
            train_split_df = pd.DataFrame({
                "filepath": image_paths[balanced_idx],
                "label_id": labels[balanced_idx],
                "fold": fold,
                "seed": seed,
                "split": "train"
            })
            train_split_df.to_csv(f"{OUTPUT_DIR}/train_split_fold{fold}_seed{seed}.csv", index=False)

            train_ds = build_dataset(image_paths[balanced_idx], labels[balanced_idx], train=True, seed=seed)
            val_ds   = build_dataset(image_paths[va], labels[va], train=False)

            model, base = build_model(architecture=MODEL_NAME)

            ckpt_p1 = f"{OUTPUT_DIR}/best_p1_fold{fold}.keras"
            ckpt_p2 = f"{OUTPUT_DIR}/best_p2_fold{fold}.keras"
            final_ckpt = f"{OUTPUT_DIR}/final_fold{fold}.keras"
            
            print(f"--- PHASE 1: Training Classification Head (Fold {fold}) ---")
            reduce_lr_p1 = tf.keras.callbacks.ReduceLROnPlateau(patience=3, factor=0.5, monitor="val_loss", verbose=1)
            model.fit(
                train_ds,
                validation_data=val_ds,
                epochs=EPOCHS_PHASE1,
                callbacks=[
                    tf.keras.callbacks.ModelCheckpoint(ckpt_p1, save_best_only=True, monitor="val_loss"),
                    tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
                    reduce_lr_p1
                ],
                verbose=1
            )
            
            print(f"--- EVALUATING PHASE 1 (Frozen) ---")
            y_true = np.concatenate([y.numpy() for _, y in val_ds])
            y_prob_frozen = np.concatenate([model.predict(x).ravel() for x, _ in val_ds])
            y_pred_frozen = (y_prob_frozen >= 0.5).astype(int)
            
            prec_f, rec_f, _ = precision_recall_curve(y_true, y_prob_frozen)
            results.append({
                "seed": seed,
                "fold": fold,
                "protocol": "Frozen",
                "accuracy": accuracy_score(y_true, y_pred_frozen),
                "precision": precision_score(y_true, y_pred_frozen, zero_division=0),
                "recall": recall_score(y_true, y_pred_frozen),
                "f1": f1_score(y_true, y_pred_frozen),
                "auc": roc_auc_score(y_true, y_prob_frozen),
                "pr_auc": auc(rec_f, prec_f),
                "mcc": matthews_corrcoef(y_true, y_pred_frozen)
            })

            print(f"--- PHASE 2: Fine-Tuning Base Model (Fold {fold}) ---")
            
            UNFREEZE_PER_MODEL = {
                "ResNet50":      15,
                "DenseNet121":   20,
                "MobileNetV2":   15,
                "EfficientNetB0":15,
                "ConvNeXtTiny":  20,
                "VGG16":         8
            }
            unfreeze_layers = UNFREEZE_PER_MODEL.get(MODEL_NAME, 15)

            # Enable layer-wise trainability
            base.trainable = True
            
            # Freeze all base layers first
            for layer in base.layers:
                layer.trainable = False
                
            # Unfreeze only the last N layers
            for layer in base.layers[-unfreeze_layers:]:
                if not isinstance(layer, layers.BatchNormalization):
                    layer.trainable = True

            # Recompile with fine tuning learning rate
            ft_lr = HP.get("fine_tune_learning_rate", 1e-5)
            model.compile(
                optimizer=Adam(ft_lr), 
                loss="binary_crossentropy", 
                metrics=["accuracy"]
            )
            
            reduce_lr_p2 = tf.keras.callbacks.ReduceLROnPlateau(patience=3, factor=0.5, monitor="val_loss", verbose=1)
            model.fit(
                train_ds,
                validation_data=val_ds,
                epochs=EPOCHS_PHASE2,
                callbacks=[
                    tf.keras.callbacks.ModelCheckpoint(ckpt_p2, save_best_only=True, monitor="val_loss"),
                    tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
                    reduce_lr_p2
                ],
                verbose=1
            )
            
            model.save(final_ckpt)

            print(f"--- EVALUATING PHASE 2 (Fine-tuned) ---")
            y_prob_ft = np.concatenate([model.predict(x).ravel() for x, _ in val_ds])
            y_pred_ft = (y_prob_ft >= 0.5).astype(int)

            prec_ft, rec_ft, _ = precision_recall_curve(y_true, y_prob_ft)
            results.append({
                "seed": seed,
                "fold": fold,
                "protocol": "Fine-tuned",
                "accuracy": accuracy_score(y_true, y_pred_ft),
                "precision": precision_score(y_true, y_pred_ft, zero_division=0),
                "recall": recall_score(y_true, y_pred_ft),
                "f1": f1_score(y_true, y_pred_ft),
                "auc": roc_auc_score(y_true, y_prob_ft),
                "pr_auc": auc(rec_ft, prec_ft),
                "mcc": matthews_corrcoef(y_true, y_pred_ft)
            })
            
            # The ROC plots will use the fine-tuned probabilities (y_prob_ft)
            all_seed_results.append(results[-1]) # Appending only fine-tuned to global summary for consistency with previous scripts
            
            # ------------------------------------------------------
            #                ROC CURVE PER FOLD
            # ------------------------------------------------------
            fpr, tpr, _ = roc_curve(y_true, y_prob_ft)
            plt.figure(figsize=(6,6))
            plt.plot(fpr, tpr, label=f"Fold {fold} (AUC={results[-1]['auc']:.3f})")
            plt.plot([0,1],[0,1],"k--")
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title(f"ROC Curve – Seed {seed} Fold {fold}")
            plt.legend()
            plt.savefig(f"{OUTPUT_DIR}/roc_fold{fold}_seed{seed}.png")
            plt.close()

            # Save predictions for this fold (Fine-tuned)
            pd.DataFrame({
                "filepath": image_paths[va],
                "y_true": y_true,
                "y_prob_frozen": y_prob_frozen,
                "y_prob_ft": y_prob_ft,
                "y_pred_ft": y_pred_ft,
                "fold": fold,
                "seed": seed
            }).to_csv(f"{OUTPUT_DIR}/predictions_fold{fold}_seed{seed}.csv", index=False)

        # ------------------------------------------------------
        #                 SAVE SUMMARY STATISTICS PER SEED
        # ------------------------------------------------------
        df_res = pd.DataFrame(results)
        df_res.to_csv(f"{OUTPUT_DIR}/kfold_results_seed{seed}.csv", index=False)
        
        # Calculate mean and std per protocol
        df_mean = df_res.groupby('protocol').mean(numeric_only=True)
        df_std = df_res.groupby('protocol').std(numeric_only=True)
        
        # Detailed stats per protocol
        df_frozen = df_res[df_res['protocol'] == 'Frozen'].drop(columns=['protocol']).reset_index(drop=True)
        df_frozen.loc['mean'] = df_frozen.mean()
        df_frozen.loc['std'] = df_frozen.std()
        
        df_ft = df_res[df_res['protocol'] == 'Fine-tuned'].drop(columns=['protocol']).reset_index(drop=True)
        df_ft.loc['mean'] = df_ft.mean()
        df_ft.loc['std'] = df_ft.std()
        
        summary_text = [
            f"\n===== SUMMARY SEED {seed} (FROZEN) =====",
            df_frozen.to_string(),
            f"\n===== SUMMARY SEED {seed} (FINE-TUNED) =====",
            df_ft.to_string(),
            f"\n===== SUMMARY SEED {seed} (AGGREGATED MEAN) =====",
            df_mean.to_string(),
            f"\n===== SUMMARY SEED {seed} (AGGREGATED STD) =====",
            df_std.to_string()
        ]
        
        final_summary_str = "\n".join(summary_text)
        print(final_summary_str)
        
        with open(f"{OUTPUT_DIR}/results_seed_{seed}.txt", "w") as f:
            f.write(final_summary_str)
        
        # ------------------------------------------------------
        #             PLOT AVERAGE ROC CURVES PER SEED
        # ------------------------------------------------------
        
        # 1. ROC Curves for Frozen
        plt.figure(figsize=(7,7))
        for fold in range(1, N_SPLITS+1):
            pred_df = pd.read_csv(f"{OUTPUT_DIR}/predictions_fold{fold}_seed{seed}.csv")
            fpr, tpr, _ = roc_curve(pred_df["y_true"], pred_df["y_prob_frozen"])
            auc_score = roc_auc_score(pred_df["y_true"], pred_df["y_prob_frozen"])
            plt.plot(fpr, tpr, label=f"Fold {fold} (AUC={auc_score:.3f})")

        plt.plot([0,1],[0,1],"k--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC Curves Across Folds - Frozen (Seed {seed})")
        plt.legend(loc="lower right")
        plt.savefig(f"{OUTPUT_DIR}/roc_all_folds_frozen_seed{seed}.png")
        plt.close()

        # 2. ROC Curves for Fine-Tuned
        plt.figure(figsize=(7,7))
        for fold in range(1, N_SPLITS+1):
            pred_df = pd.read_csv(f"{OUTPUT_DIR}/predictions_fold{fold}_seed{seed}.csv")
            fpr, tpr, _ = roc_curve(pred_df["y_true"], pred_df["y_prob_ft"])
            auc_score = roc_auc_score(pred_df["y_true"], pred_df["y_prob_ft"])
            plt.plot(fpr, tpr, label=f"Fold {fold} (AUC={auc_score:.3f})")

        plt.plot([0,1],[0,1],"k--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC Curves Across Folds - Fine-Tuned (Seed {seed})")
        plt.legend(loc="lower right")
        plt.savefig(f"{OUTPUT_DIR}/roc_all_folds_finetuned_seed{seed}.png")
        plt.close()

        # 3. ROC Curves for Both
        plt.figure(figsize=(8,8))
        for fold in range(1, N_SPLITS+1):
            pred_df = pd.read_csv(f"{OUTPUT_DIR}/predictions_fold{fold}_seed{seed}.csv")
            
            # Frozen
            fpr_f, tpr_f, _ = roc_curve(pred_df["y_true"], pred_df["y_prob_frozen"])
            auc_score_f = roc_auc_score(pred_df["y_true"], pred_df["y_prob_frozen"])
            plt.plot(fpr_f, tpr_f, linestyle=':', alpha=0.7, label=f"Fold {fold} Frozen (AUC={auc_score_f:.3f})")
            
            # Fine-Tuned
            fpr_ft, tpr_ft, _ = roc_curve(pred_df["y_true"], pred_df["y_prob_ft"])
            auc_score_ft = roc_auc_score(pred_df["y_true"], pred_df["y_prob_ft"])
            plt.plot(fpr_ft, tpr_ft, linestyle='-', label=f"Fold {fold} FT (AUC={auc_score_ft:.3f})")

        plt.plot([0,1],[0,1],"k--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC Curves Comparison - Frozen vs Fine-Tuned (Seed {seed})")
        
        # Move legend outside to avoid clutter
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/roc_all_folds_comparison_seed{seed}.png")
        plt.close()

    # ------------------------------------------------------
    #             GLOBAL MULTI-SEED SUMMARY
    # ------------------------------------------------------
    print(f"\n======================================")
    print(f"      GLOBAL MULTI-SEED SUMMARY       ")
    print(f"======================================")
    df_all = pd.DataFrame(all_seed_results)
    
    # Calculate mean and std grouping by fold, then average across folds? 
    # Usually we average across all folds and all seeds directly, or average fold-performance per seed.
    # Let's group by seed first to get the seed-level performance, then aggregate
    metrics_cols = ['accuracy', 'precision', 'recall', 'f1', 'auc', 'pr_auc', 'mcc']
    seed_summary = df_all.groupby('seed')[metrics_cols].mean().reset_index()
    print("Average performance per seed:")
    print(seed_summary)
    
    global_mean = seed_summary[metrics_cols].mean()
    global_std = seed_summary[metrics_cols].std()
    
    global_df = pd.DataFrame({'Mean': global_mean, 'Std': global_std})
    print("\nGlobal Performance Across All Seeds:")
    print(global_df)
    
    global_df.to_csv(os.path.join(OUTPUT_BASE_DIR, "global_multi_seed_results.csv"))
    print(f"\nAll results saved to: {OUTPUT_BASE_DIR}")

if __name__ == "__main__":
    main()
