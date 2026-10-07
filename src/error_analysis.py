from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from tensorflow import keras
from data_pipeline import load_datasets

# Test set ko dobara load karo (shuffle nahi hota, to order stable rehta hai)
_, _, test_ds, class_names = load_datasets()

# Prefetch ke baad file_paths nahi milte, to unhe folder se khud nikalte hain.
# image_dataset_from_directory class folders ko alphabetical order me padhta hai,
# aur har folder ki files ko bhi sorted order me. Hum wahi order dohrate hain.
file_paths = []
for cls in class_names:
    cls_files = sorted((Path("data/processed/test") / cls).glob("*.*"))
    file_paths += [str(p) for p in cls_files]

model = keras.models.load_model("models/mobilenetv2.keras")
probs = model.predict(test_ds, verbose=0)

y_true = np.concatenate([np.argmax(y.numpy(), axis=1) for _, y in test_ds])
y_pred = np.argmax(probs, axis=1)

# Safety check: file ka folder naam asli label se match hona chahiye.
# (Ye ab yahan hai, kyunki file_paths aur y_true dono ban chuke hain.)
folder_labels = [Path(p).parent.name for p in file_paths]
assert folder_labels == [class_names[i] for i in y_true], "Order mismatch! file_paths galat order me hain"

# Sabhi predictions ek table me (ye Phase 4 me database me jayegi)
df = pd.DataFrame({
    "file_path": file_paths,
    "actual": [class_names[i] for i in y_true],
    "predicted": [class_names[i] for i in y_pred],
    "confidence": probs.max(axis=1).round(4),
})
df["correct"] = df["actual"] == df["predicted"]
df.to_csv("reports/test_predictions.csv", index=False)
print(f"Wrong predictions: {(~df['correct']).sum()} / {len(df)}")

# Galat photos ka grid: asli class glass, model ne bola metal
wrong = df[(df["actual"] == "glass") & (df["predicted"] == "metal")].head(12)
fig, axes = plt.subplots(3, 4, figsize=(12, 9))
for ax, (_, row) in zip(axes.flat, wrong.iterrows()):
    ax.imshow(Image.open(row["file_path"]))
    ax.set_title(f"conf {row['confidence']:.2f}", fontsize=10)
    ax.axis("off")
for ax in axes.flat[len(wrong):]:
    ax.axis("off")
plt.suptitle("Actual: glass, Predicted: metal")
plt.tight_layout()
plt.savefig("reports/figures/glass_as_metal.png", dpi=150)
plt.show()