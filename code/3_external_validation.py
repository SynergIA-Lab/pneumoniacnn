"""
generalization_external.py
==========================

Generalization experiment on an external pneumonia dataset.

The model is NOT retrained.
The best model from the paper (fold 1 or average) is evaluated on a
balanced external dataset (500 NORMAL + 500 PNEUMONIA).

Updated for JCR Review:
- Evaluates models across all seeds defined in config.json to prove robustness.
- Uses configuration file for paths and parameters.
"""

# ======================================================
# IMPORTS
# ======================================================
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'false'

import json
import random
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, roc_curve,
    precision_recall_curve, auc, matthews_corrcoef
)

# ======================================================
# CONFIGURATION
# ======================================================
config_path = os.path.join(os.path.dirname(__file__), "config.json")
with open(config_path, 'r') as f:
    config = json.load(f)

FOLD = 2                       # fixed fold used in the paper

IMG_SIZE = tuple(config["img_size"])
BATCH_SIZE = config["batch_size"]

MAIN_RESULTS_DIR = config["output_dir"]


EXTERNAL_DATASET_DIR = config["external_dataset_dir"]
OUTPUT_DIR = f"{MAIN_RESULTS_DIR}/Generalization"

ARCHITECTURES = config["architectures"]
SEEDS = config["seeds"]

N_SAMPLES_PER_CLASS = 500

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ======================================================
# REPRODUCIBILITY
# ======================================================
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

def balanced_external_split(df, seed=42):
    df0 = df[df["label_id"] == 0]
    df1 = df[df["label_id"] == 1]

    df0 = df0.sample(
        n=min(N_SAMPLES_PER_CLASS, len(df0)),
        random_state=seed
    )
    df1 = df1.sample(
        n=min(N_SAMPLES_PER_CLASS, len(df1)),
        random_state=seed
    )

    df_balanced = pd.concat([df0, df1]).sample(
        frac=1, random_state=seed
    )

    return df_balanced

# ======================================================
# GENERALIZATION EVALUATION
# ======================================================

def main():

    print("Loading external dataset...")
    df_ext = build_dataframe(EXTERNAL_DATASET_DIR)
    
    # We will use the first seed for external split consistency, or average over splits.
    # To keep the external test set strictly identical across runs, we fix the subsampling seed.
    FIXED_EVAL_SEED = 42 
    df_ext = balanced_external_split(df_ext, seed=FIXED_EVAL_SEED)

    print(df_ext["label"].value_counts())

    ds_ext = build_dataset(
        df_ext["filepath"].values,
        df_ext["label_id"].values
    )
    
    print(f"\nExternal test dataset ready: {len(ds_ext)} samples")
    
    all_results_raw = []
    
    for arch in ARCHITECTURES:
        print(f"\nEvaluating Architecture: {arch}")
        for seed in SEEDS:
            for protocol, model_filename in [("Frozen", f"best_p1_fold{FOLD}.keras"), ("Fine-tuned", f"best_p2_fold{FOLD}.keras")]:
                model_path = os.path.join(MAIN_RESULTS_DIR, arch, f"seed_{seed}", model_filename)

                if not os.path.exists(model_path):
                    print(f"[WARNING] Model not found: {model_path}")
                    continue
            
                print(f"Loading {protocol} model for {arch} (Seed {seed})...")
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
                auc_val = roc_auc_score(y_true, y_prob)
                precision_vals, recall_vals, _ = precision_recall_curve(y_true, y_prob)
                pr_auc_val = auc(recall_vals, precision_vals)
                mcc_val = matthews_corrcoef(y_true, y_pred)

                all_results_raw.append({
                    "Model": arch,
                    "Seed": seed,
                    "Protocol": protocol,
                    "Accuracy": acc,
                    "Precision": prec,
                    "Recall": rec,
                    "F1-score": f1,
                    "AUC": auc_val,
                    "PR-AUC": pr_auc_val,
                    "MCC": mcc_val,
                    "y_true": y_true,
                    "y_prob": y_prob
                })

    if not all_results_raw:
        print("\nNo models were found to evaluate. Please run train_kfold.py first.")
        return

    # -----------------------------------------------------
    # AGGREGATE RESULTS TABLE (MEAN ACROSS SEEDS)
    # -----------------------------------------------------
    df_raw = pd.DataFrame(all_results_raw)
    
    # Save raw detailed results
    df_raw.drop(columns=['y_true', 'y_prob']).to_csv(f"{OUTPUT_DIR}/external_validation_all_seeds.csv", index=False)
    
    # Group by architecture and calculate mean and std across seeds
    metrics_cols = ['Accuracy', 'Precision', 'Recall', 'F1-score', 'AUC', 'PR-AUC', 'MCC']
    df_mean = df_raw.groupby(['Model', 'Protocol'])[metrics_cols].mean().reset_index()
    df_std = df_raw.groupby(['Model', 'Protocol'])[metrics_cols].std().reset_index()

    df_combined = df_mean.copy()
    for col in metrics_cols:
        df_combined[col] = df_mean[col].map('{:.4f}'.format) + " ± " + df_std[col].map('{:.4f}'.format)
    
    print("\n===== EXTERNAL GENERALIZATION RESULTS (MEAN ± STD ACROSS SEEDS) =====")
    print(df_combined.to_string(index=False))

    df_combined.to_csv(f"{OUTPUT_DIR}/external_validation_mean_std.csv", index=False)
    df_mean.to_csv(f"{OUTPUT_DIR}/external_validation_mean.csv", index=False)
    df_std.to_csv(f"{OUTPUT_DIR}/external_validation_std.csv", index=False)
    print("\nExternal validation results saved.")
    
    # -----------------------------------------------------
    # PLOT ROC CURVE
    # -----------------------------------------------------
    for protocol in ["Frozen", "Fine-tuned"]:
        plt.figure(figsize=(8,8))

        for arch in ARCHITECTURES:
            arch_data = df_raw[(df_raw['Model'] == arch) & (df_raw['Protocol'] == protocol)]
            if arch_data.empty:
                continue
                
            y_true_arch = arch_data.iloc[0]['y_true']
            
            y_probs = np.vstack(arch_data['y_prob'].values)
            y_prob_mean = np.mean(y_probs, axis=0)
            
            fpr, tpr, _ = roc_curve(y_true_arch, y_prob_mean)
            auc_score = roc_auc_score(y_true_arch, y_prob_mean)
            
            plt.plot(fpr, tpr, label=f"{arch} (Ensemble AUC = {auc_score:.3f})")

        # Diagonal
        plt.plot([0,1], [0,1], 'k--', label="Random")

        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"Ensemble ROC Curves on External Validation Dataset ({protocol})")
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/roc_external_validation_ensemble_{protocol}.png", dpi=300)
        plt.close()
    
if __name__ == "__main__":
    # Force CPU
    with tf.device('/CPU:0'):
        main()
