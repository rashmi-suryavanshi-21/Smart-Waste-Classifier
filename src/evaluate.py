import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow import keras
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
from data_pipeline import load_datasets

FIG_DIR = Path("reports/figures")
REPORT_DIR = Path("reports")

def evaluate(name, split, datasets, class_names):
    ds = datasets[split]
    model = keras.models.load_model(f"models/{name}.keras")

    # Datasets for val/test are not shuffled, so labels and predictions line up
    y_true = np.concatenate([np.argmax(y.numpy(), axis=1) for _, y in ds])
    y_pred = np.argmax(model.predict(ds, verbose=0), axis=1)

    print(f"\n===== {name} on {split} =====")
    print(classification_report(y_true, y_pred, target_names=class_names, digits=3))

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6.5, 5.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"{name} confusion matrix ({split})")
    plt.tight_layout()
    plt.savefig(FIG_DIR / f"{name}_confusion_{split}.png", dpi=150)
    plt.close()

    return {
        "model": name, "split": split,
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "macro_f1": round(f1_score(y_true, y_pred, average="macro"), 4),
        "trash_f1": round(f1_score(y_true, y_pred, average=None)[class_names.index("trash")], 4),
        "params_millions": round(model.count_params() / 1e6, 2),
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--split", choices=["val", "test"], default="val")
    args = parser.parse_args()

    train_ds, val_ds, test_ds, class_names = load_datasets()
    datasets = {"val": val_ds, "test": test_ds}
    rows = [evaluate(m, args.split, datasets, class_names) for m in args.models]

    table = pd.DataFrame(rows)
    print("\n", table.to_string(index=False))
    table.to_csv(REPORT_DIR / f"comparison_{args.split}.csv", index=False)