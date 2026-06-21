from pathlib import Path
from typing import Union

import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import numpy as np
from enum import Enum

PROJECT_ROOT = Path(__file__).resolve().parents[2]
class BaselineRun(Enum):
    BASELINE = PROJECT_ROOT / "runs" / "obb" / "train_baseline"
    BASELINE_DOTA8 = PROJECT_ROOT / "runs" / "obb" / "train_baseline_dota8"
    BASELINE_DOTA128 = PROJECT_ROOT / "runs" / "obb" / "train_baseline_dota128"

RunPath = Union[str, Path, BaselineRun]

def resolve_run_dir(run: RunPath) -> Path:
    """
    Converts the input into a Path object.

    Examples:
    BaselineRun.BASELINE -> Path("runs/obb/train_baseline")
    "runs/obb/train_augmented" -> Path("runs/obb/train_augmented")
    Path("runs/obb/train_augmented") -> Path("runs/obb/train_augmented")
    """
    if isinstance(run, BaselineRun):
        return run.value

    return Path(run)


# ---------------------------------------------------------------------
# Load results.csv and calculate F1
# ---------------------------------------------------------------------

def load_results(run_dir: RunPath) -> pd.DataFrame:
    run_dir = resolve_run_dir(run_dir)
    csv_path = run_dir / "results.csv"

    if not csv_path.exists():
        raise FileNotFoundError(f"No results.csv found in: {csv_path}")

    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]

    p_col = "metrics/precision(B)"
    r_col = "metrics/recall(B)"

    if p_col in df.columns and r_col in df.columns:
        p = df[p_col]
        r = df[r_col]

        df["metrics/F1(B)"] = np.where(
            (p + r) > 0,
            2 * p * r / (p + r),
            0
        )

    return df


def get_best_row(
    df: pd.DataFrame,
    metric: str = "metrics/mAP50-95(B)"
) -> pd.Series:
    if metric not in df.columns:
        raise ValueError(f"Metric '{metric}' not found in results.csv.")

    idx = df[metric].idxmax()
    return df.loc[idx]


# ---------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------

def make_summary_table(
    baseline_df: pd.DataFrame,
    experiment_df: pd.DataFrame,
    baseline_name: str = "baseline",
    new_name: str = "new",
    best_metric: str = "metrics/mAP50-95(B)"
) -> pd.DataFrame:

    baseline_best = get_best_row(baseline_df, best_metric)
    new_best = get_best_row(experiment_df, best_metric)

    metrics = [
        "metrics/mAP50-95(B)",
        "metrics/mAP50(B)",
        "metrics/precision(B)",
        "metrics/recall(B)",
        "metrics/F1(B)",
        "val/box_loss",
        "val/cls_loss",
        "val/dfl_loss",
        "val/angle_loss",
    ]

    rows = []

    for metric in metrics:
        if metric not in baseline_best.index or metric not in new_best.index:
            continue

        base_val = baseline_best[metric]
        new_val = new_best[metric]
        delta = new_val - base_val

        rows.append({
            "metric": metric,
            baseline_name: base_val,
            new_name: new_val,
            "delta": delta,
            "delta_%": (delta / base_val * 100) if base_val != 0 else np.nan
        })

    summary = pd.DataFrame(rows)

    numeric_cols = summary.select_dtypes(include=[np.number]).columns
    summary[numeric_cols] = summary[numeric_cols].round(5)

    return summary


# ---------------------------------------------------------------------
# Save CSV-based metric curves
# ---------------------------------------------------------------------

def plot_metric_curves_save(
    baseline_df: pd.DataFrame,
    experiment_df: pd.DataFrame,
    out_dir: str | Path,
    baseline_name: str = "baseline",
    new_name: str = "new"
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics = [
        "metrics/mAP50-95(B)",
        "metrics/mAP50(B)",
        "metrics/precision(B)",
        "metrics/recall(B)",
        "metrics/F1(B)",
        "val/box_loss",
        "val/cls_loss",
        "val/dfl_loss",
        "val/angle_loss",
        "train/box_loss",
        "train/cls_loss",
        "train/dfl_loss",
        "train/angle_loss",
    ]

    for metric in metrics:
        if metric not in baseline_df.columns or metric not in experiment_df.columns:
            continue

        plt.figure(figsize=(10, 5))

        plt.plot(
            baseline_df["epoch"],
            baseline_df[metric],
            label=baseline_name,
            linewidth=2
        )

        plt.plot(
            experiment_df["epoch"],
            experiment_df[metric],
            label=new_name,
            linewidth=2
        )

        plt.title(metric)
        plt.xlabel("Epoch")
        plt.ylabel(metric)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()

        safe_name = (
            metric
            .replace("/", "_")
            .replace("(", "")
            .replace(")", "")
            .replace("-", "_")
        )

        plt.savefig(out_dir / f"{safe_name}.png", dpi=160)
        plt.close()


def plot_bar_comparison_save(
    summary: pd.DataFrame,
    out_dir: str | Path,
    baseline_name: str = "baseline",
    new_name: str = "new"
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    higher_is_better = [
        "metrics/mAP50-95(B)",
        "metrics/mAP50(B)",
        "metrics/precision(B)",
        "metrics/recall(B)",
        "metrics/F1(B)",
    ]

    plot_df = summary[summary["metric"].isin(higher_is_better)].copy()

    x = np.arange(len(plot_df))
    width = 0.35

    plt.figure(figsize=(12, 6))
    plt.bar(x - width / 2, plot_df[baseline_name], width, label=baseline_name)
    plt.bar(x + width / 2, plot_df[new_name], width, label=new_name)

    plt.xticks(x, plot_df["metric"], rotation=35, ha="right")
    plt.ylabel("Score")
    plt.title("YOLO OBB Comparison: Main Metrics")
    plt.grid(axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plt.savefig(out_dir / "metric_bar_comparison.png", dpi=160)
    plt.close()


# ---------------------------------------------------------------------
#YOLO-generated images side by side
# ---------------------------------------------------------------------

def create_image_side_by_side(
    baseline_dir: RunPath,
    experiment_dir: RunPath,
    image_name: str,
    baseline_label: str = "baseline",
    new_label: str = "new"
):
    baseline_dir = resolve_run_dir(baseline_dir)
    experiment_dir = resolve_run_dir(experiment_dir)

    img1_path = baseline_dir / image_name
    img2_path = experiment_dir / image_name

    if not img1_path.exists() or not img2_path.exists():
        return None

    img1 = Image.open(img1_path).convert("RGB")
    img2 = Image.open(img2_path).convert("RGB")

    target_height = min(img1.height, img2.height)

    def resize_to_height(img, height):
        ratio = height / img.height
        width = int(img.width * ratio)
        return img.resize((width, height))

    img1 = resize_to_height(img1, target_height)
    img2 = resize_to_height(img2, target_height)

    label_height = 40
    gap = 20

    canvas_width = img1.width + img2.width + gap
    canvas_height = target_height + label_height

    canvas = Image.new("RGB", (canvas_width, canvas_height), "white")
    draw = ImageDraw.Draw(canvas)

    draw.text((10, 10), baseline_label, fill="black")
    draw.text((img1.width + gap + 10, 10), new_label, fill="black")

    canvas.paste(img1, (0, label_height))
    canvas.paste(img2, (img1.width + gap, label_height))

    return canvas


def make_image_side_by_side_save(
    baseline_dir: RunPath,
    experiment_dir: RunPath,
    out_dir: str | Path,
    image_name: str,
    baseline_label: str = "baseline",
    new_label: str = "new"
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    canvas = create_image_side_by_side(
        baseline_dir=baseline_dir,
        experiment_dir=experiment_dir,
        image_name=image_name,
        baseline_label=baseline_label,
        new_label=new_label
    )

    if canvas is None:
        return

    out_path = out_dir / f"side_by_side_{image_name}"
    canvas.save(out_path)


# ---------------------------------------------------------------------
# Show results without saving
# ---------------------------------------------------------------------

def show_summary_table(summary: pd.DataFrame):
    """
    Displays the summary nicely in Jupyter/Notebook.
    Falls back to plain text in normal Python environments.
    """
    try:
        from IPython.display import display
        display(summary)
    except ImportError:
        print(summary.to_string(index=False))


def show_pil_image(img, title: str | None = None):
    """
    Displays a PIL image in normal IDEs like PyCharm using matplotlib.
    """
    if img is None:
        print("No image to display.")
        return

    plt.figure(figsize=(16, 8))
    plt.imshow(img)
    plt.axis("off")

    if title is not None:
        plt.title(title)

    plt.tight_layout()
    plt.show()


def plot_metric_curves_show(
    baseline_df: pd.DataFrame,
    experiment_df: pd.DataFrame,
    baseline_name: str = "baseline",
    new_name: str = "new"
):
    metrics = [
        "metrics/mAP50-95(B)",
        "metrics/mAP50(B)",
        "metrics/precision(B)",
        "metrics/recall(B)",
        "metrics/F1(B)",
        "val/box_loss",
        "val/cls_loss",
        "val/dfl_loss",
        "val/angle_loss",
        "train/box_loss",
        "train/cls_loss",
        "train/dfl_loss",
        "train/angle_loss",
    ]

    for metric in metrics:
        if metric not in baseline_df.columns or metric not in experiment_df.columns:
            continue

        plt.figure(figsize=(10, 5))

        plt.plot(
            baseline_df["epoch"],
            baseline_df[metric],
            label=baseline_name,
            linewidth=2
        )

        plt.plot(
            experiment_df["epoch"],
            experiment_df[metric],
            label=new_name,
            linewidth=2
        )

        plt.title(metric)
        plt.xlabel("Epoch")
        plt.ylabel(metric)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()


def plot_bar_comparison_show(
    summary: pd.DataFrame,
    baseline_name: str = "baseline",
    new_name: str = "new"
):
    higher_is_better = [
        "metrics/mAP50-95(B)",
        "metrics/mAP50(B)",
        "metrics/precision(B)",
        "metrics/recall(B)",
        "metrics/F1(B)",
    ]

    plot_df = summary[summary["metric"].isin(higher_is_better)].copy()

    x = np.arange(len(plot_df))
    width = 0.35

    plt.figure(figsize=(12, 6))
    plt.bar(x - width / 2, plot_df[baseline_name], width, label=baseline_name)
    plt.bar(x + width / 2, plot_df[new_name], width, label=new_name)

    plt.xticks(x, plot_df["metric"], rotation=35, ha="right")
    plt.ylabel("Score")
    plt.title("YOLO OBB Comparison: Main Metrics")
    plt.grid(axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------
# save comparison report
# ---------------------------------------------------------------------

def compare_yolo_runs_save(
    baseline: RunPath,
    experiment_dir: RunPath,
    out_dir: str | Path = "comparison_report",
    baseline_name: str = "baseline",
    new_name: str = "new",
    best_metric: str = "metrics/mAP50-95(B)"
) -> pd.DataFrame:

    baseline_dir = resolve_run_dir(baseline)
    experiment_dir = resolve_run_dir(experiment_dir)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    baseline_df = load_results(baseline_dir)
    experiment_df = load_results(experiment_dir)

    summary = make_summary_table(
        baseline_df=baseline_df,
        experiment_df=experiment_df,
        baseline_name=baseline_name,
        new_name=new_name,
        best_metric=best_metric
    )

    summary.to_csv(out_dir / "summary_comparison.csv", index=False)

    plot_metric_curves_save(
        baseline_df=baseline_df,
        experiment_df=experiment_df,
        out_dir=out_dir,
        baseline_name=baseline_name,
        new_name=new_name
    )

    plot_bar_comparison_save(
        summary=summary,
        out_dir=out_dir,
        baseline_name=baseline_name,
        new_name=new_name
    )

    yolo_images = [
        "BoxF1_curve.png",
        "BoxP_curve.png",
        "BoxR_curve.png",
        "BoxPR_curve.png",
        "confusion_matrix.png",
        "confusion_matrix_normalized.png",
        "results.png",
        "labels.jpg",
    ]

    for img_name in yolo_images:
        make_image_side_by_side_save(
            baseline_dir=baseline_dir,
            experiment_dir=experiment_dir,
            out_dir=out_dir,
            image_name=img_name,
            baseline_label=baseline_name,
            new_label=new_name
        )

    print("\nComparison done.")
    print(f"Report saved in: {out_dir}")
    print(f"\nBest-epoch comparison based on: {best_metric}")
    print(summary.to_string(index=False))

    return summary


# ---------------------------------------------------------------------
# show comparison directly
# ---------------------------------------------------------------------

def compare_yolo_runs_show(
    baseline: RunPath,
    experiment_dir: RunPath,
    baseline_name: str = "baseline",
    new_name: str = "new",
    best_metric: str = "metrics/mAP50-95(B)",
    show_yolo_images: bool = True,
    show_curves: bool = True
) -> pd.DataFrame:

    baseline_dir = resolve_run_dir(baseline)
    experiment_dir = resolve_run_dir(experiment_dir)

    baseline_df = load_results(baseline_dir)
    experiment_df = load_results(experiment_dir)

    summary = make_summary_table(
        baseline_df=baseline_df,
        experiment_df=experiment_df,
        baseline_name=baseline_name,
        new_name=new_name,
        best_metric=best_metric
    )

    print(f"\nBest-epoch comparison based on: {best_metric}")
    show_summary_table(summary)

    plot_bar_comparison_show(
        summary=summary,
        baseline_name=baseline_name,
        new_name=new_name
    )

    if show_curves:
        plot_metric_curves_show(
            baseline_df=baseline_df,
            experiment_df=experiment_df,
            baseline_name=baseline_name,
            new_name=new_name
        )

    if show_yolo_images:
        yolo_images = [
            "BoxF1_curve.png",
            "BoxP_curve.png",
            "BoxR_curve.png",
            "BoxPR_curve.png",
            "confusion_matrix.png",
            "confusion_matrix_normalized.png",
            "results.png",
            "labels.jpg",
        ]

        for img_name in yolo_images:
            img = create_image_side_by_side(
                baseline_dir=baseline_dir,
                experiment_dir=experiment_dir,
                image_name=img_name,
                baseline_label=baseline_name,
                new_label=new_name
            )

            if img is not None:
                print(f"\n{img_name}")
                show_pil_image(img)

    return summary