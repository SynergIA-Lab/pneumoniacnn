"""
xai_quantitative.py
===================

Quantitative evaluation of XAI explanations (paper metrics).
"""

import os
import numpy as np
import tensorflow as tf
import pandas as pd
from skimage.metrics import structural_similarity as ssim

MODEL_NAME = "EfficientNetB0"
FOLD = 1
SEED = 42
IMG_SIZE = (224, 224)
DATASET_DIR = "/Users/franciscoantoniogomezvela/Git-wrokspace/GitHub/proyectosTransferencia/pneumoniaCNN/Images"
OUTPUT_DIR = f"Results/{MODEL_NAME}"

np.random.seed(SEED)

def load_img(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IMG_SIZE)
    return (img / 255.0).numpy()

def saliency(model, img):
    img_tf = tf.convert_to_tensor(img[None])
    with tf.GradientTape() as tape:
        tape.watch(img_tf)
        score = model(img_tf)[0][0]
    grads = tape.gradient(score, img_tf)[0]
    sal = tf.reduce_max(tf.abs(grads), axis=-1)
    return sal.numpy()

def deletion(model, img, hmap, steps=50):
    flat = hmap.flatten()
    idxs = np.argsort(-flat)
    img_c = img.copy()
    probs = []

    k = len(flat) // steps
    h, w, _ = img.shape

    for i in range(steps):
        mask = np.zeros(h*w, dtype=bool)
        mask[idxs[i*k:(i+1)*k]] = True
        mask = mask.reshape(h, w)
        img_c[mask] = 0
        probs.append(float(model.predict(img_c[None])[0][0]))

    return np.trapz(probs)

def main():

    model = tf.keras.models.load_model(
        f"{OUTPUT_DIR}/best_fold{FOLD}.keras"
    )

    paths = []
    for cls in ["NORMAL", "PNEUMONIA"]:
        cls_dir = os.path.join(DATASET_DIR, cls)
        paths += [
            os.path.join(cls_dir, p)
            for p in sorted(os.listdir(cls_dir))[:2]
        ]

    rows = []

    for p in paths:
        img = load_img(p)
        hmap = saliency(model, img)
        auc = deletion(model, img, hmap)

        rows.append({
            "image": os.path.basename(p),
            "deletion_auc": auc
        })

    df = pd.DataFrame(rows)
    df.to_csv(f"{OUTPUT_DIR}/xai_metrics.csv", index=False)
    print(df)

if __name__ == "__main__":
    main()
