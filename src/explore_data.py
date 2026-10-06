from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from PIL import Image

RAW_DIR = Path("data/raw")
FIG_DIR = Path("reports/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Collect every image path with its label (the label = the folder name)
rows = []
for class_dir in sorted(RAW_DIR.iterdir()):
    if class_dir.is_dir():
        for img_path in class_dir.glob("*.*"):
            rows.append({"path": str(img_path), "label": class_dir.name})

df = pd.DataFrame(rows)
print(f"Total images: {len(df)}")
print(df["label"].value_counts())

# --- Plot 1: class distribution ---
counts = df["label"].value_counts()
plt.figure(figsize=(8, 5))
ax = sns.barplot(x=counts.index, y=counts.values, palette="viridis")
ax.bar_label(ax.containers[0])  # show the number on each bar
plt.title("Class Distribution (TrashNet)")
plt.ylabel("Number of images")
plt.xlabel("Waste class")
plt.tight_layout()
plt.savefig(FIG_DIR / "class_distribution.png", dpi=150)
plt.show()

# --- Plot 2: 4 sample images per class ---
classes = sorted(df["label"].unique())
fig, axes = plt.subplots(len(classes), 4, figsize=(10, 2.2 * len(classes)))
for i, cls in enumerate(classes):
    samples = df[df["label"] == cls].sample(4, random_state=42)
    for j, path in enumerate(samples["path"]):
        axes[i, j].imshow(Image.open(path))
        axes[i, j].axis("off")
        if j == 0:
            axes[i, j].set_title(cls, loc="left", fontsize=11)
plt.tight_layout()
plt.savefig(FIG_DIR / "sample_images.png", dpi=150)
plt.show()

# --- Check image sizes (are they consistent?) ---
sizes = [Image.open(p).size for p in df["path"].sample(200, random_state=1)]
print("Unique sizes in a 200-image sample:", set(sizes))