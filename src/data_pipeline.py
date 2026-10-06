from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.utils.class_weight import compute_class_weight

DATA_DIR = Path("data/processed")
IMG_SIZE = (224, 224)   # MobileNetV2 and ResNet50 expect 224x224
BATCH_SIZE = 32
SEED = 42

def load_datasets():
    """Load train/val/test as tf.data datasets, reading class names from folders."""
    common = dict(
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="categorical",  # one-hot labels
        seed=SEED,
    )
    train_ds = keras.utils.image_dataset_from_directory(
        DATA_DIR / "train", shuffle=True, **common)
    val_ds = keras.utils.image_dataset_from_directory(
        DATA_DIR / "val", shuffle=False, **common)
    test_ds = keras.utils.image_dataset_from_directory(
        DATA_DIR / "test", shuffle=False, **common)  # no shuffle: keeps order for evaluation

    class_names = train_ds.class_names  # alphabetical, e.g. ['cardboard', 'glass', ...]

    # Prefetch = prepare the next batch while the GPU/CPU trains on the current one
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(AUTOTUNE)
    val_ds = val_ds.prefetch(AUTOTUNE)
    test_ds = test_ds.prefetch(AUTOTUNE)
    return train_ds, val_ds, test_ds, class_names

def build_augmentation():
    """Random transformations, applied ONLY during training.
    Keras turns these layers off automatically at prediction time."""
    return keras.Sequential([
        keras.layers.RandomFlip("horizontal"),
        keras.layers.RandomRotation(0.1),       # up to +/- 10% of a full turn (~36 deg)
        keras.layers.RandomZoom(0.15),
        keras.layers.RandomContrast(0.15),
        keras.layers.RandomBrightness(0.15),
    ], name="augmentation")

def get_class_weights(class_names):
    """Rare classes get bigger weights so the model can't ignore them."""
    labels = []
    for idx, name in enumerate(class_names):
        n_images = len(list((DATA_DIR / "train" / name).glob("*.*")))
        labels += [idx] * n_images
    weights = compute_class_weight(
        class_weight="balanced", classes=np.arange(len(class_names)), y=np.array(labels))
    return {i: float(w) for i, w in enumerate(weights)}

if __name__ == "__main__":
    train_ds, val_ds, test_ds, class_names = load_datasets()
    print("Classes:", class_names)
    print("Class weights:", get_class_weights(class_names))

    # Visual check: one image shown augmented 9 different ways
    import matplotlib.pyplot as plt
    aug = build_augmentation()
    images, _ = next(iter(train_ds))
    plt.figure(figsize=(8, 8))
    for i in range(9):
        plt.subplot(3, 3, i + 1)
        plt.imshow(aug(images[:1], training=True)[0].numpy().astype("uint8"))
        plt.axis("off")
    plt.suptitle("Same image, 9 random augmentations")
    plt.savefig("reports/figures/augmentation_examples.png", dpi=150)
    plt.show()