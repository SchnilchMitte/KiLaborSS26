import zipfile
from pathlib import Path
from io import BytesIO
import pandas as pd


outer_zip_path = Path("data/train.zip")
inner_label_zip_path = "train/labelTxt-v1.0/labelTxt.zip"

rows = []

with zipfile.ZipFile(outer_zip_path, "r") as outer_z:
    with outer_z.open(inner_label_zip_path) as inner_zip_file:
        inner_zip_bytes = BytesIO(inner_zip_file.read())

    with zipfile.ZipFile(inner_zip_bytes, "r") as label_z:
        label_files = [
            f for f in label_z.namelist()
            if f.endswith(".txt")
        ]

        print("Anzahl Label-Dateien:", len(label_files))
        print(label_files[:5])

        for label_file in label_files:
            image_id = Path(label_file).stem

            with label_z.open(label_file) as f:
                for raw_line in f:
                    line = raw_line.decode("utf-8").strip()
                    if not line:
                        continue

                    parts = line.split()

                    if parts[0] in ["imagesource:", "gsd:"]:
                        continue

                    if len(parts) < 10:
                        continue

                    x1, y1, x2, y2, x3, y3, x4, y4 = map(float, parts[:8])

                    rows.append({
                        "image_id": image_id,
                        "category": parts[8],
                        "difficult": int(parts[9]),
                        "x1": x1, "y1": y1,
                        "x2": x2, "y2": y2,
                        "x3": x3, "y3": y3,
                        "x4": x4, "y4": y4,
                        "label_file": label_file,
                    })

df = pd.DataFrame(rows)

print(df.head())

print(df.shape)
print(df["category"].value_counts())