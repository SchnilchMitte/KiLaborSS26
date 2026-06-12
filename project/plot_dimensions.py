import matplotlib.pyplot as plt
from DOTA import DOTA
import dota_utils as util
from collections import defaultdict


def plot_dimension_distribution(basepath):
    dota = DOTA(basepath)

    # Store min_dimensions per class:
    # { 'class_name': { 'orig': [], '1024': [], '512': [], '256': [] } }
    class_dims = defaultdict(lambda: defaultdict(list))

    print(f"Processing annotations from: {basepath}")

    # Target longest sides
    target_scales = [1024, 512, 256]

    for img_id, annotations in dota.ImgToAnns.items():
        # --- IMPORTANT: You need the original image dimensions to calculate scale ---
        # Assuming dota object or a lookup table provides (width, height) for img_id
        # If not available, you may need to extract this from your metadata/images
        try:
            img_w, img_h, _ = dota.loadImgs(img_id, ext='jpg')[0].shape
        except AttributeError as e:
            print(e)
            continue
            # Fallback/Placeholder: In a real scenario, ensure img_w/h is accessible
            # This is a dummy value; replace with actual logic to get image resolution
            img_w, img_h = 2000, 1000

        max_orig_side = max(img_w, img_h)

        for ann in annotations:
            class_name = ann['name']
            poly = ann['poly']
            flat_poly = []
            for pt in poly:
                flat_poly.extend([pt[0], pt[1]])

            # Returns [cx, cy, w, h, theta]
            res = util.polygonToRotRectangle(flat_poly)
            w, h = res[2], res[3]

            # 1. Original min dimension
            min_dim_orig = min(w, h)
            class_dims[class_name]['orig'].append(min_dim_orig)

            # 2. Calculate scaled min dimensions
            for target in target_scales:
                scale_factor = target / max_orig_side
                scaled_min_dim = min_dim_orig * scale_factor
                class_dims[class_name][str(target)].append(scaled_min_dim)

    if not class_dims:
        print("No annotations found.")
        return

    # Plotting
    classes = sorted(class_dims.keys())
    num_classes = len(classes)
    cols = 3
    rows = (num_classes + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(18, rows * 4), constrained_layout=True)
    axes = axes.flatten()

    # Colors for the different scales
    colors = {'orig': 'black', '1024': 'blue', '512': 'green', '256': 'red'}
    alphas = {'orig': 0.4, '1024': 0.5, '512': 0.6, '256': 0.7}

    for i, cls in enumerate(classes):
        ax = axes[i]

        # We plot all scales on the same axis to compare distributions
        # Use log scale for X-axis because dimensions can vary wildly (e.g., 10px to 1000px)
        for scale_key in ['orig', '1024', '512', '256']:
            data = class_dims[cls][scale_key]
            if not data:
                continue

            ax.hist(data, bins=30, color=colors[scale_key],
                    alpha=alphas[scale_key], label=f"{scale_key.replace('orig', 'Original')}")

        ax.set_title(f"{cls} (n={len(class_dims[cls]['orig'])})")
        ax.set_xscale('log')  # Log scale is much better for dimension distributions
        ax.set_xlabel("Min Dimension (pixels) - Log Scale")
        ax.set_ylabel("Frequency")
        ax.legend(fontsize='small')
        ax.grid(True, which="both", ls="-", alpha=0.2)

    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.suptitle(f"Distribution of Bounding Box Min(Width, Height) at Different Scales", fontsize=16)

    # Save the plot
    plot_filename = "dim_distribution.png"
    plt.savefig(plot_filename)
    print(f"\nPlot saved as '{plot_filename}'")
    plt.show()


if __name__ == "__main__":
    # Replace with your actual dataset path
    DATASET_PATH = "DOTAv1.5_train"
    plot_dimension_distribution(DATASET_PATH)
