import matplotlib.pyplot as plt
import os
from DOTA import DOTA


def analyze_dota_distribution(basepath):
    if not os.path.exists(basepath):
        print(f"Error: The path '{basepath}' does not exist.")
        return

    print(f"Loading dataset from: {basepath}...")
    dota = DOTA(basepath)

    # The DOTA class stores category counts in catToImgs.
    # Because 'catToImgs[cat].append(imgid)' is called for every object,
    # the length of the list for a category equals the total number of instances.

    # Extract class names and their respective counts
    class_counts = {}
    for category, img_ids in dota.catToImgs.items():
        class_counts[category] = len(img_ids)

    if not class_counts:
        print("No annotations found in the specified directory.")
        return

    # Sort the classes by count (descending)
    sorted_counts = dict(sorted(class_counts.items(), key=lambda item: item[1], reverse=True))

    # Print
    print("\n--- Label Distribution ---")
    print(f"{'Category':<20} | {'Count':<10}")
    print("-" * 35)
    for category, count in sorted_counts.items():
        print(f"{category:<20} | {count:<10}")
    print("-" * 35)
    print(f"Total Annotations: {sum(sorted_counts.values())}")
    print(f"Total Unique Classes: {len(sorted_counts)}")

    # Plotting
    plt.figure(figsize=(12, 7))
    categories = list(sorted_counts.keys())
    counts = list(sorted_counts.values())

    bars = plt.bar(categories, counts, color='skyblue', edgecolor='navy')

    # Add labels on top of each bar
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval + 0.5, int(yval),
                 ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.xlabel('Class Name', fontsize=12)
    plt.ylabel('Number of Instances', fontsize=12)
    plt.title('DOTA Dataset: Class Distribution', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()

    # Save and show the plot
    plot_filename = "label_distribution.png"
    plt.savefig(plot_filename)
    print(f"\nPlot saved as '{plot_filename}'")
    plt.show()


if __name__ == "__main__":
    DATASET_PATH = "DOTA/train"
    analyze_dota_distribution(DATASET_PATH)
