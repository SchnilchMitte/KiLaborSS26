import os
import cv2
import random
import numpy as np
import matplotlib.pyplot as plt


# ----------------------------
# CONFIG
# ----------------------------

# Adjust this depending on current working directory

SPLIT = "train"


# Optional: match your DOTA classes
CLASS_NAMES = {
    0: "plane",
    1: "ship",
    2: "storage tank",
    3: "baseball diamond",
    4: "tennis court",
    5: "basketball court",
    6: "ground track field",
    7: "harbor",
    8: "bridge",
    9: "large vehicle",
    10: "small vehicle",
    11: "helicopter",
    12: "roundabout",
    13: "soccer ball field",
    14: "swimming pool"
}


# ----------------------------
# HELPERS
# ----------------------------

def load_yolo_obb(label_path):
    labels = []
    if not os.path.exists(label_path):
        return labels

    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 9:
                continue

            cls = int(parts[0])
            coords = list(map(float, parts[1:9]))

            pts = np.array(coords).reshape(4, 2)
            labels.append((cls, pts))

    return labels


def draw_labels(img, labels):
    h, w = img.shape[:2]

    for cls, pts in labels:
        pts_px = (pts.copy() * np.array([w, h])).astype(int)

        color = (0, 255, 0)

        cv2.polylines(img, [pts_px], isClosed=True, color=color, thickness=2)

        # label text
        x, y = pts_px[0]
        name = CLASS_NAMES.get(cls, str(cls))
        cv2.putText(
            img,
            name,
            (x, y - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA
        )

    return img


def show_grid(images, titles, cols=3):
    if len(images) == 0 or len(titles) == 0:
        raise ValueError("No images")
    rows = (len(images) + cols - 1) // cols
    plt.figure(figsize=(15, 5 * rows))

    for i, (img, title) in enumerate(zip(images, titles)):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        plt.subplot(rows, cols, i + 1)
        plt.imshow(img_rgb)
        plt.title(title)
        plt.axis("off")

    plt.tight_layout()
    plt.show()


# ----------------------------
# MAIN VISUALIZATION
# ----------------------------

def visualize_random(img_root, lbl_root, n_samples=10):
    
    img_dir = os.path.join(img_root, SPLIT)
    lbl_dir = os.path.join(lbl_root, SPLIT)
    print(img_dir)
    images = [f for f in os.listdir(img_dir) if f.endswith(".jpg")]

    sample_count = min(n_samples, len(images))
    assert sample_count > 0
    samples = random.sample(images, sample_count)
    assert len(samples) > 0

    vis_imgs = []
    titles = []

    for img_file in samples:
        img_path = os.path.join(img_dir, img_file)
        lbl_path = os.path.join(lbl_dir, img_file.replace(".jpg", ".txt"))

        img = cv2.imread(img_path)
        if img is None:
            continue

        labels = load_yolo_obb(lbl_path)

        img_vis = draw_labels(img.copy(), labels)

        vis_imgs.append(img_vis)
        titles.append(img_file)

    show_grid(vis_imgs, titles)


if __name__ == "__main__":
    visualize_random(
        "project/dota128_sliced/images",
        "project/dota128_sliced/labels",
        12
    )
