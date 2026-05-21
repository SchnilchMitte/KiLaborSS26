import sys
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
from DOTA import DOTA
import numpy as np
from matplotlib.patches import Polygon
basepath = "DOTA/train"

dota = DOTA(basepath)


img_ids = dota.getImgIds()
print("Anzahl Bilder:", len(img_ids))
num_annotations = sum(len(anns) for anns in dota.ImgToAnns.values())
print("Anzahl Annotationen:", num_annotations)
print("Erste Bild-ID:", img_ids[0])

for img_id, anns in dota.ImgToAnns.items():
    print(img_id, len(anns))

img_id = img_ids[0]
anns = dota.loadAnns(imgId=img_id)

print("Annotationen:")
for ann in anns[:5]:
    print(ann)


def show_image_with_annotations(
    dota,
    img_id=None,
    figsize=(12, 12)
):


    catNms = []

    img_path = Path(dota.imagepath) / f"{img_id}.png"

    img = cv2.imread(str(img_path))

    if img is None:
        raise FileNotFoundError(f"Bild konnte nicht geladen werden: {img_path}")

    # OpenCV = BGR, matplotlib erwartet RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    anns = dota.loadAnns(catNms=catNms, imgId=img_id)


    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(img)
    ax.axis("off")
    ax.set_title(f"{img_id} | {len(anns)} Annotationen")

    for ann in anns:
        poly = np.array(ann["poly"], dtype=np.float32)

        polygon = Polygon(
            poly,
            closed=True,
            fill=True,
            edgecolor="red",
            facecolor="red",
            alpha=0.25,
            linewidth=2
        )

        ax.add_patch(polygon)
        x0, y0 = poly[0]
        ax.plot(x0, y0, "bo", markersize=4)

        ax.text(
                x0,
                y0,
                ann["name"],
                color="yellow",
                fontsize=8,
                bbox=dict(facecolor="black", alpha=0.6, edgecolor="none")
            )

    plt.tight_layout()
    plt.show()

    return anns


show_image_with_annotations(dota, img_id="P0005")