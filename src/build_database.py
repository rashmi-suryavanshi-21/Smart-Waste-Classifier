import sqlite3
from pathlib import Path
import pandas as pd

DB_PATH = Path("database/waste.db")
SCHEMA = Path("database/schema.sql")
CSV = Path("reports/test_predictions.csv")

# Recyclable: cardboard, glass, metal, paper, plastic = 1. Trash = 0.
CLASSES = [
    ("cardboard", 1), ("glass", 1), ("metal", 1),
    ("paper", 1), ("plastic", 1), ("trash", 0),
]

if DB_PATH.exists():
    DB_PATH.unlink()          # purani database hata do, dobara fresh banegi

conn = sqlite3.connect(DB_PATH)
conn.execute("PRAGMA foreign_keys = ON")
conn.executescript(SCHEMA.read_text())     # tables banao

# 1) classes table bharo
class_id = {}
for i, (name, recyclable) in enumerate(CLASSES, start=1):
    conn.execute("INSERT INTO waste_classes VALUES (?, ?, ?)", (i, name, recyclable))
    class_id[name] = i

# 2) images aur predictions bharo
df = pd.read_csv(CSV)
for n, row in enumerate(df.itertuples(index=False), start=1):
    conn.execute(
        "INSERT INTO images VALUES (?, ?, ?, ?)",
        (n, Path(row.file_path).name, class_id[row.actual], "test"),
    )
    conn.execute(
        "INSERT INTO predictions VALUES (?, ?, ?, ?, ?, ?)",
        (n, n, class_id[row.predicted], float(row.confidence),
         int(row.correct), "mobilenetv2"),
    )

conn.commit()

# Check: kitni rows gayi
for table in ["waste_classes", "images", "predictions"]:
    count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    print(table, count)
conn.close()