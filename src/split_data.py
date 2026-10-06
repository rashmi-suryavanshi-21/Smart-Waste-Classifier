from pathlib import Path
import shutil
from sklearn.model_selection import train_test_split

RAW_DIR = Path("data/raw")
OUT_DIR = Path("data/processed")
SEED = 42  # fixed seed = same split every time (reproducibility)

# 1. Gather all paths and labels
paths, labels = [], []
for class_dir in sorted(RAW_DIR.iterdir()):
    if class_dir.is_dir():
        for p in class_dir.glob("*.*"):
            paths.append(p)
            labels.append(class_dir.name)

# 2. First split: 70% train, 30% temporary holdout
train_p, temp_p, train_l, temp_l = train_test_split(
    paths, labels, test_size=0.30, stratify=labels, random_state=SEED
)
# 3. Split the holdout in half: 15% val, 15% test
val_p, test_p, val_l, test_l = train_test_split(
    temp_p, temp_l, test_size=0.50, stratify=temp_l, random_state=SEED
)

# 4. Copy files into data/processed/<split>/<class>/
def copy_files(split_name, split_paths, split_labels):
    for path, label in zip(split_paths, split_labels):
        dest = OUT_DIR / split_name / label
        dest.mkdir(parents=True, exist_ok=True)
        shutil.copy(path, dest / path.name)

if OUT_DIR.exists():
    shutil.rmtree(OUT_DIR)  # start clean so reruns don't mix old files

copy_files("train", train_p, train_l)
copy_files("val", val_p, val_l)
copy_files("test", test_p, test_l)

print(f"Train: {len(train_p)} | Val: {len(val_p)} | Test: {len(test_p)}")