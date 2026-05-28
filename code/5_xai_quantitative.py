"""
xai_quantitative.py
===================

Quantitative evaluation of XAI explanations (paper metrics).
"""

import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import random
from skimage.metrics import structural_similarity as ssim
import pandas as pd

import json

config_path = os.path.join(os.path.dirname(__file__), "config.json")
with open(config_path, 'r') as f:
    config = json.load(f)

# ---------------- FIXED SETTINGS ----------------
MODEL_NAME = config["model_name"]
FOLD = 1
SEED = config["seeds"][0]
IMG_SIZE = tuple(config["img_size"])
DATASET_DIR = config["dataset_dir"]
OUTPUT_DIR = f"{config['output_dir']}/{MODEL_NAME}/seed_{SEED}"
BATCH_SIZE = config["batch_size"]
AUTOTUNE = tf.data.AUTOTUNE
# -----------------------------------------------

np.random.seed(SEED)
tf.random.set_seed(SEED)
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

def load_img(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IMG_SIZE)
    return (img / 255.0).numpy()
# ======================================================
# QUALITATIVE XAI METHODS
# ======================================================

def saliency_map(model, img):
    img_tf = tf.convert_to_tensor(img[None], dtype=tf.float32)
    with tf.GradientTape() as tape:
        tape.watch(img_tf)
        score = model(img_tf)[0][0]
    grads = tape.gradient(score, img_tf)[0].numpy()
    sal = np.max(np.abs(grads), axis=-1)
    return (sal - sal.min()) / (sal.max() - sal.min() + 1e-9)

def smoothgrad(model, img, n_samples=30, noise=0.1):
    """
    Computes SmoothGrad using the maximum absolute gradient over channels.
    Returns a 2D heatmap of shape (H,W).
    """
    H, W, C = img.shape
    acc = np.zeros((H, W))   # <-- Correct shape for accumulation

    for _ in range(n_samples):
        noisy = img + np.random.normal(0, noise, img.shape)
        noisy = np.clip(noisy, 0, 1)

        # compute vanilla saliency (2D)
        sal = saliency_map(model, noisy)

        acc += sal

    sm = acc / n_samples
    # normalize
    sm = (sm - sm.min()) / (sm.max() - sm.min() + 1e-9)
    return sm

def gradcam(model, img, layer_name=None):
    if layer_name is None:
        for layer in reversed(model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                layer_name = layer.name
                break

    grad_model = tf.keras.Model(
        model.inputs,
        [model.get_layer(layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv, pred = grad_model(img[None])
        loss = pred[:, 0]

    grads = tape.gradient(loss, conv)[0]
    weights = tf.reduce_mean(grads, axis=(0,1))
    cam = tf.reduce_sum(weights * conv[0], axis=-1)
    cam = np.maximum(cam, 0)
    cam = cam / (cam.max() + 1e-9)
    cam = tf.image.resize(cam[..., None], IMG_SIZE).numpy().squeeze()
    return cam

# ======================================================
# QUANTITATIVE XAI METHODS
# ======================================================
def deletion_metric(model, img, heatmap, steps=50):
    """
    img: (H,W,3) normalized [0,1]
    heatmap: (H,W) normalized [0,1]
    """
    # order pixels by relevance
    flat = heatmap.flatten()
    idxs = np.argsort(-flat)  # desc
    h, w, c = img.shape
    img_copy = img.copy()

    probs = []
    remove_per_step = len(flat) // steps

    for i in range(steps):
        # remove most relevant pixels
        mask_idxs = idxs[i*remove_per_step:(i+1)*remove_per_step]
        mask_2d = np.zeros((h*w,), dtype=bool)
        mask_2d[mask_idxs] = True
        mask_2d = mask_2d.reshape(h, w)

        # delete pixels by setting them to 0
        corrupted = img_copy.copy()
        corrupted[mask_2d] = 0.0

        pred = float(model.predict(corrupted[None])[0][0])
        probs.append(pred)

    # area under curve
    auc = np.trapz(probs)
    return auc, probs


def insertion_metric(model, img, heatmap, steps=50):
    h, w, _ = img.shape
    flat = heatmap.flatten()
    idxs = np.argsort(-flat)  # top pixels first

    probs = []
    blank = np.zeros_like(img)

    add_per_step = len(flat) // steps

    for i in range(steps):
        mask_idxs = idxs[: (i+1)*add_per_step]
        mask_2d = np.zeros((h*w,), dtype=bool)
        mask_2d[mask_idxs] = True
        mask_2d = mask_2d.reshape(h, w)

        inserted = blank.copy()
        inserted[mask_2d] = img[mask_2d]

        pred = float(model.predict(inserted[None])[0][0])
        probs.append(pred)

    auc = np.trapz(probs)
    return auc, probs

def sparsity(heatmap, threshold=0.5):
    total = heatmap.size
    active = np.sum(heatmap > threshold)
    return 1 - active/total

def entropy(heatmap):
    p = heatmap.flatten()
    p = p / (p.sum() + 1e-9)
    return -np.sum(p * np.log(p + 1e-9))

def xai_stability_ssim(model, img, xai_func, noise_std=0.02):
    hmap_clean = xai_func(model, img)

    noisy = img + np.random.normal(0, noise_std, img.shape)
    noisy = np.clip(noisy, 0, 1)
    hmap_noisy = xai_func(model, noisy)

    return ssim(hmap_clean, hmap_noisy, data_range=1.0)



# ======================================================
# MAIN: Calculate XAI metrics for ALL 3 METHODS
# ======================================================

def main():
    # XAI methods dictionary
    xai_methods = {
        'Saliency': saliency_map,
        'SmoothGrad': smoothgrad,
        'Grad-CAM': gradcam
    }
    
    # Select random images
    paths = []
    for cls in ["NORMAL", "PNEUMONIA"]:
        cls_dir = os.path.join(DATASET_DIR, cls)
        label = "0" if cls == "NORMAL" else "1"
        all_imgs = [f for f in os.listdir(cls_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        imgs = random.sample(all_imgs, k=min(4, len(all_imgs)))  # 4 random images per class
        paths += [(os.path.join(cls_dir, i), label) for i in imgs]
    
    print(f"\n{'='*70}")
    print(f"Loaded {len(paths)} images for XAI analysis")
    print(f"{'='*70}\n")
    
    all_comparison_data = []

    for protocol, model_filename in [("Frozen", f"best_p1_fold{FOLD}.keras"), ("Fine-tuned", f"best_p2_fold{FOLD}.keras")]:
        print(f"\n\n{'*'*90}")
        print(f"EVALUATING PROTOCOL: {protocol.upper()}")
        print(f"{'*'*90}\n")

        # Load model
        model = tf.keras.models.load_model(f"{OUTPUT_DIR}/{model_filename}")
        results_all_methods = {}
        
        for method_name, xai_func in xai_methods.items():
            print(f"\n{'='*70}")
            print(f"COMPUTING METRICS FOR: {method_name.upper()}")
            print(f"{'='*70}\n")
            
            all_del_auc = []
            all_ins_auc = []
            all_sparsity = []
            all_entropy = []
            all_stability = []
            
            for idx, (path, label) in enumerate(paths):
                print(f"[{method_name}] Image {idx+1}/{len(paths)}: {os.path.basename(path)}")
                img = load_img(path)
                try:
                    heatmap = xai_func(model, img)
                except Exception as e:
                    print(f"  Error computing {method_name}: {e}")
                    continue
                
                try:
                    del_auc, _ = deletion_metric(model, img, heatmap)
                    ins_auc, _ = insertion_metric(model, img, heatmap)
                    sp = sparsity(heatmap)
                    ent = entropy(heatmap)
                    stab = xai_stability_ssim(model, img, xai_func)
                    
                    all_del_auc.append(del_auc)
                    all_ins_auc.append(ins_auc)
                    all_sparsity.append(sp)
                    all_entropy.append(ent)
                    all_stability.append(stab)
                    
                    print(f"  ✓ Del: {del_auc:.4f} | Ins: {ins_auc:.4f} | Spar: {sp:.4f} | Ent: {ent:.4f} | Stab: {stab:.4f}")
                except Exception as e:
                    print(f" Error computing metrics: {e}")
                    continue
            
            results_all_methods[method_name] = {
                'deletion_auc': np.mean(all_del_auc) if all_del_auc else 0,
                'deletion_auc_std': np.std(all_del_auc) if all_del_auc else 0,
                'insertion_auc': np.mean(all_ins_auc) if all_ins_auc else 0,
                'insertion_auc_std': np.std(all_ins_auc) if all_ins_auc else 0,
                'sparsity': np.mean(all_sparsity) if all_sparsity else 0,
                'sparsity_std': np.std(all_sparsity) if all_sparsity else 0,
                'entropy': np.mean(all_entropy) if all_entropy else 0,
                'entropy_std': np.std(all_entropy) if all_entropy else 0,
                'stability': np.mean(all_stability) if all_stability else 0,
                'stability_std': np.std(all_stability) if all_stability else 0,
            }
            
        for method_name, metrics in results_all_methods.items():
            all_comparison_data.append({
                'Protocol': protocol,
                'Method': method_name,
                'Deletion AUC': f"{metrics['deletion_auc']:.4f}±{metrics['deletion_auc_std']:.4f}",
                'Insertion AUC': f"{metrics['insertion_auc']:.4f}±{metrics['insertion_auc_std']:.4f}",
                'Sparsity': f"{metrics['sparsity']:.4f}±{metrics['sparsity_std']:.4f}",
                'Entropy': f"{metrics['entropy']:.4f}±{metrics['entropy_std']:.4f}",
                'Stability': f"{metrics['stability']:.4f}±{metrics['stability_std']:.4f}",
            })
            
            # Save detailed metrics CSV directly here, with protocol in name
            method_df = pd.DataFrame([{
                'Metric': 'Deletion AUC', 'Mean': metrics['deletion_auc'], 'Std': metrics['deletion_auc_std']
            }, {
                'Metric': 'Insertion AUC', 'Mean': metrics['insertion_auc'], 'Std': metrics['insertion_auc_std']
            }, {
                'Metric': 'Sparsity', 'Mean': metrics['sparsity'], 'Std': metrics['sparsity_std']
            }, {
                'Metric': 'Entropy', 'Mean': metrics['entropy'], 'Std': metrics['entropy_std']
            }, {
                'Metric': 'Stability SSIM', 'Mean': metrics['stability'], 'Std': metrics['stability_std']
            }])
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            method_csv = f"{OUTPUT_DIR}/xai_metrics_{method_name.lower()}_{protocol}.csv"
            method_df.to_csv(method_csv, index=False)
            print(f" {method_name} metrics ({protocol}) saved to: {method_csv}")

    print(f"\n\n{'='*90}")
    print("COMPARISON: ALL 3 XAI METHODS ACROSS PROTOCOLS")
    print(f"{'='*90}\n")
    
    comparison_df = pd.DataFrame(all_comparison_data)
    print(comparison_df.to_string(index=False))
    
    comparison_df.to_csv(f"{OUTPUT_DIR}/xai_metrics_comparison_all_protocols.csv", index=False)
    print(f"\n Comparison saved to: {OUTPUT_DIR}/xai_metrics_comparison_all_protocols.csv")
    
    print(f"\n{'='*90}")
    print(" XAI quantitative evaluation COMPLETE")
    print(f"{'='*90}\n")
    
    return comparison_df

if __name__ == "__main__":
    main()