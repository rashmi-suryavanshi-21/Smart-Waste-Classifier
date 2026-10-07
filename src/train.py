import argparse
import json
from pathlib import Path
import matplotlib.pyplot as plt
from tensorflow import keras
from data_pipeline import load_datasets, get_class_weights
from models import build_model

MODEL_DIR = Path("models")
REPORT_DIR = Path("reports")
FIG_DIR = Path("reports/figures")
MODEL_DIR.mkdir(exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Stage 2 me base model ki kitni upar wali layers unfreeze karni hain
UNFREEZE_LAYERS = {"mobilenetv2": 60, "resnet50v2": 40}

def get_callbacks():
    return [
        # 4 epoch tak val_loss na sudhre to ruk jao, aur best weights wapas le lo
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=4,
                                      restore_best_weights=True),
        # Progress ruke to learning rate aadha kar do
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2),
    ]

def compile_model(model, lr):
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr),
                  loss="categorical_crossentropy", metrics=["accuracy"])

def plot_history(history, name, stage1_epochs):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, metric in zip(axes, ["accuracy", "loss"]):
        ax.plot(history[metric], label="train")
        ax.plot(history["val_" + metric], label="validation")
        if stage1_epochs and stage1_epochs < len(history[metric]):
            ax.axvline(stage1_epochs - 0.5, color="gray", linestyle="--",
                       label="fine-tuning starts")
        ax.set_title(f"{name}: {metric}")
        ax.set_xlabel("epoch")
        ax.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / f"{name}_training_curves.png", dpi=150)
    plt.close()

def main(name, epochs, ft_epochs, ft_lr):
    train_ds, val_ds, _, class_names = load_datasets()
    class_weights = get_class_weights(class_names)
    model, base = build_model(name, len(class_names))

    # ---- Stage 1: sirf head train hota hai (baseline me poora model) ----
    compile_model(model, lr=1e-3)
    h1 = model.fit(train_ds, validation_data=val_ds, epochs=epochs,
                   class_weight=class_weights, callbacks=get_callbacks())
    history = {k: list(v) for k, v in h1.history.items()}
    stage1_epochs = len(h1.history["loss"])

    # ---- Stage 2: upar ki layers fine-tune (sirf transfer models) ----
    if base is not None:
        base.trainable = True
        for layer in base.layers[:-UNFREEZE_LAYERS[name]]:
            layer.trainable = False          # shuru ki layers frozen rehti hain
        print(f"Fine-tuning learning rate: {ft_lr}")   # check karne ke liye
        compile_model(model, lr=ft_lr)       # trainable badalne ke baad compile zaroori hai
        h2 = model.fit(train_ds, validation_data=val_ds, epochs=ft_epochs,
                       class_weight=class_weights, callbacks=get_callbacks())
        for k, v in h2.history.items():
            history[k] += list(v)

    model.save(MODEL_DIR / f"{name}.keras")
    with open(REPORT_DIR / f"{name}_history.json", "w") as f:
        json.dump(history, f)
    plot_history(history, name, stage1_epochs if base is not None else None)
    print(f"Saved models/{name}.keras")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True,
                        choices=["baseline", "mobilenetv2", "resnet50v2"])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--ft_epochs", type=int, default=10)
    parser.add_argument("--ft_lr", type=float, default=1e-4)
    args = parser.parse_args()
    main(args.model, args.epochs, args.ft_epochs, args.ft_lr)