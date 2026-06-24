from pathlib import Path

from compare_yolo_runs import (
    BaselineRun,
    compare_yolo_runs_show,
    compare_yolo_runs_save,
    load_results,
    make_summary_table,
    plot_bar_comparison_show,
    plot_metric_curves_show,
    create_image_side_by_side,
    show_pil_image,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

#BASELINE = BaselineRun.BASELINE_DOTA128
BASELINE = PROJECT_ROOT / "runs" / "obb" / "train_baseline"
EXPERIMENT_DIR = PROJECT_ROOT / "runs" / "obb" / "train-4"

BASELINE_NAME = "image 1024"
EXPERIMENT_NAME = "image1280"


def example_show_everything():
    """
    Shows the comparison directly in the IDE.
    Does not save anything - I think it makes sense to use this for testing.
    And then for the final result you want to use for the presentation use the
    example_save_report() method.
    """
    summary = compare_yolo_runs_show(
        baseline=BASELINE,
        experiment_dir=EXPERIMENT_DIR,
        baseline_name=BASELINE_NAME,
        new_name=EXPERIMENT_NAME,
        best_metric="metrics/mAP50-95(B)",
        show_yolo_images=True,
        show_curves=True,
    )

    return summary


def example_show_only_main_metrics():
    """
    Shows only the summary table and the main bar chart.
    Useful if you do not want many plots opening.
    """
    summary = compare_yolo_runs_show(
        baseline=BASELINE,
        experiment_dir=EXPERIMENT_DIR,
        baseline_name=BASELINE_NAME,
        new_name=EXPERIMENT_NAME,
        best_metric="metrics/mAP50-95(B)",
        show_yolo_images=False,
        show_curves=False,
    )

    return summary


def example_save_report():
    """
    Saves the full comparison report to disk.
    Creates CSV summary, metric plots and side-by-side YOLO images.
    """
    comparison_dir = PROJECT_ROOT / "runs" / "obb" / "comparisons" / "baseline_vs_augmented"

    summary = compare_yolo_runs_save(
        baseline=BASELINE,
        experiment_dir=EXPERIMENT_DIR,
        out_dir=comparison_dir,
        baseline_name=BASELINE_NAME,
        new_name=EXPERIMENT_NAME,
        best_metric="metrics/mAP50-95(B)",
    )

    return summary


def example_compare_with_different_best_metric():
    """
    Uses another metric to choose the best epoch.
    Default is metrics/mAP50-95(B), but you can also use mAP50, F1, recall, etc.
    """
    summary = compare_yolo_runs_show(
        baseline=BASELINE,
        experiment_dir=EXPERIMENT_DIR,
        baseline_name=BASELINE_NAME,
        new_name=EXPERIMENT_NAME,
        best_metric="metrics/mAP50(B)",
        show_yolo_images=False,
        show_curves=True,
    )

    return summary


def example_manual_usage():
    """
    Shows how to use the lower-level functions manually.
    Useful if you want more control.
    """
    baseline_df = load_results(BASELINE)
    experiment_df = load_results(EXPERIMENT_DIR)

    summary = make_summary_table(
        baseline_df=baseline_df,
        experiment_df=experiment_df,
        baseline_name=BASELINE_NAME,
        new_name=EXPERIMENT_NAME,
        best_metric="metrics/mAP50-95(B)",
    )

    print(summary.to_string(index=False))

    plot_bar_comparison_show(
        summary=summary,
        baseline_name=BASELINE_NAME,
        new_name=EXPERIMENT_NAME,
    )

    plot_metric_curves_show(
        baseline_df=baseline_df,
        experiment_df=experiment_df,
        baseline_name=BASELINE_NAME,
        new_name=EXPERIMENT_NAME,
    )

    return summary


def example_show_single_yolo_image():
    """
    Shows only one YOLO-generated image side by side.
    For example: BoxF1_curve.png, BoxPR_curve.png, results.png, confusion_matrix.png.
    """
    img = create_image_side_by_side(
        baseline_dir=BASELINE,
        experiment_dir=EXPERIMENT_DIR,
        image_name="BoxF1_curve.png",
        baseline_label=BASELINE_NAME,
        new_label=EXPERIMENT_NAME,
    )

    show_pil_image(img)


if __name__ == "__main__":
    # Choose one example here:

    # 1. Full direct visual comparison
    example_show_everything()

    # 2. Only summary + main bar chart
    #example_show_only_main_metrics()

    # 3. Save complete comparison report
    #example_save_report()

    # 4. Use another metric to select the best epoch
    #example_compare_with_different_best_metric()

    # 5. Manual usage of individual functions
    # example_manual_usage()

    # 6. Show only one YOLO-generated comparison image
    #example_show_single_yolo_image()