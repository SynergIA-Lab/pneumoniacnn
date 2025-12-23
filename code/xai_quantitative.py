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

# ---------------- FIXED SETTINGS ----------------
MODEL_NAME = "EfficientNetB0" # Model to evaluate
FOLD = 1
SEED = 42
IMG_SIZE = (224, 224)
DATASET_DIR = "/Users/franciscoantoniogomezvela/Git-wrokspace/GitHub/proyectosTransferencia/pneumoniaCNN/Images"# Dataset path
OUTPUT_DIR = f"Results/{MODEL_NAME}"# Output directory
BATCH_SIZE = 64
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

def saliency(model, img):
    img_tf = tf.convert_to_tensor(img[None])
    with tf.GradientTape() as tape:
        tape.watch(img_tf)
        score = model(img_tf)[0][0]
    grads = tape.gradient(score, img_tf)[0]
    sal = tf.reduce_max(tf.abs(grads), axis=-1)
    return sal.numpy()

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
        sal = saliency(model, noisy)

        acc += sal

    sm = acc / n_samples
    # normalize
    sm = (sm - sm.min()) / (sm.max() - sm.min() + 1e-9)
    return sm

def gradcam(model, img):
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            target = layer.name
            break

    grad_model = tf.keras.Model(
        model.inputs,
        [model.get_layer(target).output, model.output]
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
    # Load model
    model = tf.keras.models.load_model(f"{OUTPUT_DIR}/best_fold{FOLD}.keras")
    
    # XAI methods dictionary
    xai_methods = {
        'Saliency': saliency,
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
    
    # ===============================================
    # Store results for all methods
    # ===============================================
    
    results_all_methods = {}
    
    # ===============================================
    # For each XAI method
    # ===============================================
    
    for method_name, xai_func in xai_methods.items():
        print(f"\n{'='*70}")
        print(f"COMPUTING METRICS FOR: {method_name.upper()}")
        print(f"{'='*70}\n")
        
        # Diccionarios para almacenar métricas
        all_del_auc = []
        all_ins_auc = []
        all_sparsity = []
        all_entropy = []
        all_stability = []
        
        # ===============================================
        # Calcular métricas para cada imagen
        # ===============================================
        
        for idx, (path, label) in enumerate(paths):
            print(f"[{method_name}] Image {idx+1}/{len(paths)}: {os.path.basename(path)}")
            
            # Load image
            img = load_img(path)
            
            # Compute XAI explanation with this method
            try:
                heatmap = xai_func(model, img)
            except Exception as e:
                print(f"  Error computing {method_name}: {e}")
                continue
            
            # ===============================================
            # Calcular métricas
            # ===============================================
            
            try:
                del_auc, _ = deletion_metric(model, img, heatmap)
                ins_auc, _ = insertion_metric(model, img, heatmap)
                sp = sparsity(heatmap)
                ent = entropy(heatmap)
                stab = xai_stability_ssim(model, img, xai_func)
                
                # Guardar
                all_del_auc.append(del_auc)
                all_ins_auc.append(ins_auc)
                all_sparsity.append(sp)
                all_entropy.append(ent)
                all_stability.append(stab)
                
                print(f"  ✓ Del: {del_auc:.4f} | Ins: {ins_auc:.4f} | Spar: {sp:.4f} | Ent: {ent:.4f} | Stab: {stab:.4f}")
                
            except Exception as e:
                print(f" Error computing metrics: {e}")
                continue
        
        # ===============================================
        # Store results for this method
        # ===============================================
        
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
        
        # Print summary for this method
        print(f"\n{method_name.upper()} SUMMARY:")
        print(f"  Deletion AUC:   {results_all_methods[method_name]['deletion_auc']:.4f} ± {results_all_methods[method_name]['deletion_auc_std']:.4f}")
        print(f"  Insertion AUC:  {results_all_methods[method_name]['insertion_auc']:.4f} ± {results_all_methods[method_name]['insertion_auc_std']:.4f}")
        print(f"  Sparsity:       {results_all_methods[method_name]['sparsity']:.4f} ± {results_all_methods[method_name]['sparsity_std']:.4f}")
        print(f"  Entropy:        {results_all_methods[method_name]['entropy']:.4f} ± {results_all_methods[method_name]['entropy_std']:.4f}")
        print(f"  Stability SSIM: {results_all_methods[method_name]['stability']:.4f} ± {results_all_methods[method_name]['stability_std']:.4f}")
    
    # ===============================================
    # CREATE FINAL COMPARISON TABLE
    # ===============================================
    
    print(f"\n\n{'='*90}")
    print("COMPARISON: ALL 3 XAI METHODS (MEAN ± STD)")
    print(f"{'='*90}\n")
    
    # Build comparison DataFrame
    comparison_data = []
    
    for method_name, metrics in results_all_methods.items():
        comparison_data.append({
            'Method': method_name,
            'Deletion AUC': f"{metrics['deletion_auc']:.4f}±{metrics['deletion_auc_std']:.4f}",
            'Insertion AUC': f"{metrics['insertion_auc']:.4f}±{metrics['insertion_auc_std']:.4f}",
            'Sparsity': f"{metrics['sparsity']:.4f}±{metrics['sparsity_std']:.4f}",
            'Entropy': f"{metrics['entropy']:.4f}±{metrics['entropy_std']:.4f}",
            'Stability': f"{metrics['stability']:.4f}±{metrics['stability_std']:.4f}",
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    print(comparison_df.to_string(index=False))
    
    # ===============================================
    # SAVE TO CSV
    # ===============================================
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Save comparison table
    comparison_df.to_csv(f"{OUTPUT_DIR}/xai_metrics_comparison.csv", index=False)
    print(f"\n Comparison saved to: {OUTPUT_DIR}/xai_metrics_comparison.csv")
    
    # Save detailed results for each method
    for method_name, metrics in results_all_methods.items():
        method_df = pd.DataFrame([{
            'Metric': 'Deletion AUC',
            'Mean': metrics['deletion_auc'],
            'Std': metrics['deletion_auc_std']
        }, {
            'Metric': 'Insertion AUC',
            'Mean': metrics['insertion_auc'],
            'Std': metrics['insertion_auc_std']
        }, {
            'Metric': 'Sparsity',
            'Mean': metrics['sparsity'],
            'Std': metrics['sparsity_std']
        }, {
            'Metric': 'Entropy',
            'Mean': metrics['entropy'],
            'Std': metrics['entropy_std']
        }, {
            'Metric': 'Stability SSIM',
            'Mean': metrics['stability'],
            'Std': metrics['stability_std']
        }])
        
        method_csv = f"{OUTPUT_DIR}/xai_metrics_{method_name.lower()}.csv"
        method_df.to_csv(method_csv, index=False)
        print(f" {method_name} metrics saved to: {method_csv}")
    
    print(f"\n{'='*90}")
    print(" XAI quantitative evaluation COMPLETE")
    print(f"{'='*90}\n")
    
    return comparison_df

if __name__ == "__main__":
    main()