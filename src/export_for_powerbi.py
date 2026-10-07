import sqlite3
import pandas as pd

QUERY = """
SELECT
    i.image_id,
    i.file_name,
    a.class_name  AS actual_class,
    pr.class_name AS predicted_class,
    p.confidence,
    p.is_correct,
    a.is_recyclable,
    p.model_name
FROM images i
JOIN predictions p    ON p.image_id = i.image_id
JOIN waste_classes a  ON a.class_id = i.actual_class_id
JOIN waste_classes pr ON pr.class_id = p.predicted_class_id
ORDER BY i.image_id;
"""

conn = sqlite3.connect("database/waste.db")
df = pd.read_sql_query(QUERY, conn)
conn.close()

# Confidence band bhi saath me, taaki Power BI me seedha use ho
df["confidence_band"] = pd.cut(
    df["confidence"], bins=[0, 0.5, 0.8, 1.0],
    labels=["low", "medium", "high"], include_lowest=True)

df.to_csv("reports/powerbi_export.csv", index=False)
print(df.shape)
print(df.head())