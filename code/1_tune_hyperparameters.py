"""
tune_hyperparameters.py
=======================

Script to perform Hyperparameter Tuning using KerasTuner.
This script addresses JCR Review comments regarding how hyperparameter
values were chosen. It uses BayesianOptimization (or RandomSearch) to
find the best combination of learning rate, dense units, and dropout rates.
"""

import os
import json
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import (
    ResNet50, MobileNetV2, DenseNet121, EfficientNetB0, ConvNeXtTiny, VGG16
)

try:
    import keras_tuner as kt
except ImportError:
    print("Error: keras_tuner is not installed.")
    print("Please install it running: pip install keras-tuner")
    exit(1)

# ======================================================
# CONFIGURATION
# ======================================================
config_path = os.path.join(os.path.dirname(__file__), "config.json")
with open(config_path, 'r') as f:
    config = json.load(f)

MODEL_NAME   = config["model_name"]
DATASET_DIR  = config["dataset_dir"]
OUTPUT_BASE_DIR = os.path.join(config["output_dir"], "Tuner")
os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)

IMG_SIZE    = tuple(config["img_size"])
BATCH_SIZE  = config["batch_size"]
AUTOTUNE    = tf.data.AUTOTUNE

# ======================================================
# DATA LOADING (SIMPLE 80/20 SPLIT FOR TUNING)
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
    df["label_id"] = df["label"].apply(lambda x: 1 if "pneu" in x.lower() else 0).astype(int)
    return df

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
        ds = ds.shuffle(1024, seed=42)
        ds = ds.map(augment_fn, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)
    return ds

# ======================================================
# BUILD MODEL FOR TUNER
# ======================================================

def build_model(hp):
    """
    Model builder function for KerasTuner.
    """
    architecture = MODEL_NAME
    
    if architecture == "ResNet50":
        base = ResNet50(weights="imagenet", include_top=False, input_shape=(*IMG_SIZE, 3))
    elif architecture == "MobileNetV2":
        base = MobileNetV2(weights="imagenet", include_top=False, input_shape=(*IMG_SIZE, 3))
    elif architecture == "DenseNet121":
        base = DenseNet121(weights="imagenet", include_top=False, input_shape=(*IMG_SIZE, 3))
    elif architecture == "EfficientNetB0":
        base = EfficientNetB0(weights="imagenet", include_top=False, input_shape=(*IMG_SIZE, 3))
    elif architecture == "ConvNeXtTiny":
        base = ConvNeXtTiny(weights="imagenet", include_top=False, input_shape=(*IMG_SIZE, 3))
    elif architecture == "VGG16":
        base = VGG16(weights="imagenet", include_top=False, input_shape=(*IMG_SIZE, 3))
    else:
        raise ValueError("Unknown architecture")
    
    base.trainable = True

    x = layers.GlobalAveragePooling2D()(base.output)
    
    # Tune first dense layer units
    hp_dense_1 = hp.Choice('dense_1_units', values=[256, 512, 1024])
    x = layers.Dense(hp_dense_1, activation="relu")(x)
    
    # Tune first dropout rate
    hp_dropout_1 = hp.Float('dropout_1_rate', min_value=0.2, max_value=0.5, step=0.1)
    x = layers.Dropout(hp_dropout_1)(x)
    
    # Tune second dense layer units
    hp_dense_2 = hp.Choice('dense_2_units', values=[128, 256, 512])
    x = layers.Dense(hp_dense_2, activation="relu")(x)
    
    # Tune second dropout rate
    hp_dropout_2 = hp.Float('dropout_2_rate', min_value=0.2, max_value=0.5, step=0.1)
    x = layers.Dropout(hp_dropout_2)(x)
    
    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = models.Model(inputs=base.input, outputs=outputs)
    
    # Tune learning rate
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-3, 5e-4, 1e-4, 5e-5])
    
    model.compile(
        optimizer=Adam(learning_rate=hp_learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model

# ======================================================
# MAIN TUNING PROCESS
# ======================================================

def main():
    print(f"Starting Hyperparameter Tuning for {MODEL_NAME}...")
    df = build_dataframe(DATASET_DIR)
    
    # Simple Train/Val split (80/20) for tuning
    from sklearn.model_selection import train_test_split
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["label_id"])
    
    train_ds = build_dataset(train_df["filepath"].values, train_df["label_id"].values, train=True)
    val_ds = build_dataset(val_df["filepath"].values, val_df["label_id"].values, train=False)

    # Initialize BayesianOptimization Tuner
    tuner = kt.BayesianOptimization(
        build_model,
        objective='val_loss',
        max_trials=10,        # Number of different configurations to try
        executions_per_trial=1, # Number of times to train each configuration
        directory=OUTPUT_BASE_DIR,
        project_name='pneumonia_hp_tuning',
        overwrite=True
    )

    tuner.search_space_summary()

    # Search for the best hyperparameters
    tuner.search(
        train_ds,
        validation_data=val_ds,
        epochs=10, # Keep it relatively small for tuning
        callbacks=[tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)],
        verbose=1
    )

    print("\n===== TUNING RESULTS =====")
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    
    print(f"Optimal Learning Rate: {best_hps.get('learning_rate')}")
    print(f"Optimal Dense 1 Units: {best_hps.get('dense_1_units')}")
    print(f"Optimal Dropout 1 Rate: {best_hps.get('dropout_1_rate')}")
    print(f"Optimal Dense 2 Units: {best_hps.get('dense_2_units')}")
    print(f"Optimal Dropout 2 Rate: {best_hps.get('dropout_2_rate')}")
    
    # Save a CSV with the summary of all trials
    trial_data = []
    for trial_id, trial in tuner.oracle.trials.items():
        if trial.status == "COMPLETED":
            row = {
                "trial_id": trial_id,
                "score_val_loss": trial.score,
            }
            row.update(trial.hyperparameters.values)
            trial_data.append(row)
            
    df_trials = pd.DataFrame(trial_data)
    df_trials = df_trials.sort_values(by="score_val_loss", ascending=True)
    summary_csv_path = os.path.join(OUTPUT_BASE_DIR, "tuning_results_summary.csv")
    df_trials.to_csv(summary_csv_path, index=False)
    print(f"\nDetailed tuning results saved to: {summary_csv_path}")

    # Update recommendations
    print("\nTo use these parameters, update your config.json with:")
    print(json.dumps({
        "learning_rate": best_hps.get('learning_rate'),
        "dense_1_units": best_hps.get('dense_1_units'),
        "dropout_1_rate": best_hps.get('dropout_1_rate'),
        "dense_2_units": best_hps.get('dense_2_units'),
        "dropout_2_rate": best_hps.get('dropout_2_rate')
    }, indent=4))

if __name__ == "__main__":
    main()
