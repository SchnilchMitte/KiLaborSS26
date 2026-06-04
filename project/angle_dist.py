import matplotlib.pyplot as plt
from DOTA import DOTA
import dota_utils as util
from collections import defaultdict
import numpy as np


def plot_angle_distribution(basepath):
    dota = DOTA(basepath)

    # Store angles per class: { 'class_name': [angle1, angle2, ...] }
    class_angles = defaultdict(list)

    print(f"Processing annotations from: {basepath}")

    for img_id, annotations in dota.ImgToAnns.items():
        for ann in annotations:
            class_name = ann['name']

            # polygonToRotRectangle expects a flat list: [x1, y1, x2, y2, x3, y3, x4, y4]
            # parse_dota_poly returns: [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
            poly = ann['poly']
            flat_poly = []
            for pt in poly:
                flat_poly.extend([pt[0], pt[1]])

            # Returns [cx, cy, w, h, theta]
            res = util.polygonToRotRectangle(flat_poly)
            angle = res[4]

            class_angles[class_name].append(angle)

    if not class_angles:
        print("No annotations found.")
        return

    # Plotting
    classes = sorted(class_angles.keys())
    num_classes = len(classes)
    cols = 3
    rows = (num_classes + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(15, rows * 3), constrained_layout=True)
    axes = axes.flatten()

    for i, cls in enumerate(classes):
        ax = axes[i]
        angles = class_angles[cls]

        # Convert radians to degrees for easier reading
        angles_deg = np.degrees(angles)

        ax.hist(angles_deg, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
        ax.set_title(f"{cls} (n={len(angles)})")
        ax.set_xlim(-90, 90)  # Standardize view for rotation
        ax.set_xlabel("Angle (Degrees)")
        ax.set_ylabel("Frequency")

    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.suptitle(f"Distribution of Bounding Box Angles in DOTA Dataset", fontsize=16)
    plt.show()

    # Save the plot
    plot_filename = "angle_distribution.png"
    plt.savefig(plot_filename)
    print(f"\nPlot saved as '{plot_filename}'")


if __name__ == "__main__":
    DATASET_PATH = "DOTA/train"
    plot_angle_distribution(DATASET_PATH)
