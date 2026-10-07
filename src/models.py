from tensorflow import keras
from tensorflow.keras import layers
from data_pipeline import build_augmentation, IMG_SIZE

INPUT_SHAPE = IMG_SIZE + (3,)   # (224, 224, 3)

def build_baseline(num_classes):
    """Small CNN from scratch: 3 blocks of Conv -> BatchNorm -> ReLU -> Pool."""
    inputs = keras.Input(shape=INPUT_SHAPE)
    x = build_augmentation()(inputs)        # active only during training
    x = layers.Rescaling(1 / 255)(x)        # pixels 0-255 -> 0-1
    for filters in [32, 64, 128]:
        x = layers.Conv2D(filters, 3, padding="same")(x)
        x = layers.Activation("relu")(x)
        x = layers.MaxPooling2D()(x)
    x = layers.GlobalAveragePooling2D()(x)  # one number per feature map
    x = layers.Dropout(0.3)(x)              # randomly switch off 30% of units: less overfitting
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    return keras.Model(inputs, outputs, name="baseline_cnn"), None

def build_transfer(name, num_classes):
    """Pretrained base + new head. Returns (model, base) so we can unfreeze later."""
    if name == "mobilenetv2":
        base = keras.applications.MobileNetV2(
            include_top=False, weights="imagenet", input_shape=INPUT_SHAPE)
    elif name == "resnet50v2":
        base = keras.applications.ResNet50V2(
            include_top=False, weights="imagenet", input_shape=INPUT_SHAPE)
    else:
        raise ValueError(f"Unknown model: {name}")

    base.trainable = False                  # Stage 1: freeze

    inputs = keras.Input(shape=INPUT_SHAPE)
    x = build_augmentation()(inputs)
    x = layers.Rescaling(1 / 127.5, offset=-1)(x)   # pixels 0-255 -> -1..1 (what these nets expect)
    # training=False keeps BatchNorm layers in inference mode, even during fine-tuning.
    # This is important: otherwise small batches can wreck the pretrained statistics.
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    return keras.Model(inputs, outputs, name=name), base

def build_model(name, num_classes):
    if name == "baseline":
        return build_baseline(num_classes)
    return build_transfer(name, num_classes)