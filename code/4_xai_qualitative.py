"""
xai_qualitative.py
==================

Qualitative XAI visualizations used in the paper.
"""

import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

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

def main():
    paths = []
    # Guardamos tuplas (ruta, etiqueta) para saber qué es cada imagen
    for cls in ["NORMAL", "PNEUMONIA"]:
        cls_dir = os.path.join(DATASET_DIR, cls)
        label = "0" if cls == "NORMAL" else "1"
        imgs = sorted(os.listdir(cls_dir))[:4]
        paths += [(os.path.join(cls_dir, i), label) for i in imgs]

    for protocol, model_filename in [("frozen", f"best_p1_fold{FOLD}.keras"), ("finetuned", f"best_p2_fold{FOLD}.keras")]:
        print(f"Generating XAI maps for {protocol} model...")
        model = tf.keras.models.load_model(
            f"{OUTPUT_DIR}/{model_filename}"
        )

        for i, (p, label) in enumerate(paths):
            img = load_img(p)
            sal = saliency_map(model, img)
            smg = smoothgrad(model, img)
            cam = gradcam(model, img)

            plt.figure(figsize=(15, 5)) # Aumentamos un poco el ancho para los títulos

            # 1. Imagen Original con Label
            plt.subplot(1, 4, 1)
            plt.imshow(img)
            plt.title(f"Original (Label= {label})", fontsize=12)
            plt.axis("off")

            # 2. Saliency Map
            plt.subplot(1, 4, 2)
            plt.imshow(sal, cmap="inferno")
            plt.title("Saliency Map", fontsize=12)
            plt.axis("off")

            # 3. SmoothGrad
            plt.subplot(1, 4, 3)
            plt.imshow(smg, cmap="inferno")
            plt.title("SmoothGrad", fontsize=12)
            plt.axis("off")

            # 4. Grad-CAM (Superpuesta)
            plt.subplot(1, 4, 4)
            plt.imshow(img, alpha=0.5)
            plt.imshow(cam, cmap="jet", alpha=0.5)
            plt.title("Grad-CAM Overlay", fontsize=12)
            plt.axis("off")

            # Ajuste para que no se solapen los títulos
            plt.tight_layout()
            plt.savefig(f"{OUTPUT_DIR}/xai_example_{i}_{protocol}.png", bbox_inches='tight', dpi=300)
            plt.close()
   

if __name__ == "__main__":
    main()
