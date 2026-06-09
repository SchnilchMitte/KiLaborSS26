import cv2
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Polygon as MplPolygon

def visualize_obb_vs_segmentation(
    img_rgb,
    anns,
    masks,
    alpha=0.45,
    figsize=(14, 14),
    show_obb=True,
    show_outline=True,
):
    overlay = img_rgb.copy().astype(np.float32)
    rng = np.random.default_rng(123)

    # Erst alle Masken in overlay eintragen
    for item in masks:
        mask = item["mask"].astype(bool)

        if mask.sum() == 0:
            continue

        color = rng.integers(40, 255, size=3)
        overlay[mask] = color

    blended = cv2.addWeighted(
        img_rgb.astype(np.uint8),
        1 - alpha,
        overlay.astype(np.uint8),
        alpha,
        0
    )

    # Jetzt erst plotten
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(blended)
    ax.axis("off")

    for ann, item in zip(anns, masks):
        mask = item["mask"].astype(bool)

        if mask.sum() == 0:
            continue

        # OBB = weiße Linie
        if show_obb:
            poly = np.array(ann["poly"], dtype=np.float32)

            obb_patch = MplPolygon(
                poly,
                closed=True,
                fill=False,
                edgecolor="white",
                linewidth=1.5
            )
            ax.add_patch(obb_patch)

        # Segmentierungskontur = rote Linie
        if show_outline:
            contours, _ = cv2.findContours(
                mask.astype(np.uint8),
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )

            for cnt in contours:
                cnt = cnt.squeeze()

                if cnt.ndim == 2 and len(cnt) >= 3:
                    ax.plot(
                        cnt[:, 0],
                        cnt[:, 1],
                        color="red",
                        linewidth=2
                    )

        # Label
        ys, xs = np.where(mask)

        if len(xs) > 0:
            cx = int(xs.mean())
            cy = int(ys.mean())

            ax.text(
                cx,
                cy,
                item["category"],
                color="yellow",
                fontsize=8,
                bbox=dict(
                    facecolor="black",
                    alpha=0.7,
                    edgecolor="none"
                )
            )

    ax.set_title("White = original OBB | Red = generated segmentation contour | Color = mask area")
    plt.tight_layout()
    plt.show()




def visualize_obb_segmentation_result(
    img_rgb,
    anns,
    masks,
    alpha=0.45,
    figsize=(14, 14),
    show_obb=True,
    show_contour=True,
    show_label=True,
    show_score=True,
    min_score=None,
):
    """
    Visualisiert OBBs und erzeugte Segmentierungsmasken.

    img_rgb: RGB-Bild
    anns: DOTA-Annotationen
    masks: Liste aus segmenter.segment_many(...)
           erwartet pro item:
           {
               "category": ...,
               "mask": ...,
               "score": ...,
               "metrics": ...
           }
    min_score: optional; nur Masken ab diesem Score anzeigen
    """

    base = img_rgb.copy().astype(np.uint8)
    overlay = base.copy().astype(np.float32)

    rng = np.random.default_rng(42)

    for ann, item in zip(anns, masks):
        score = item.get("score", 0.0)

        if min_score is not None and score < min_score:
            continue

        mask = item["mask"].astype(bool)

        if mask.sum() == 0:
            continue

        color = rng.integers(40, 255, size=3)
        overlay[mask] = color

    blended = cv2.addWeighted(
        base,
        1 - alpha,
        overlay.astype(np.uint8),
        alpha,
        0
    )

    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(blended)
    ax.axis("off")

    for ann, item in zip(anns, masks):
        score = item.get("score", 0.0)

        if min_score is not None and score < min_score:
            continue

        mask = item["mask"].astype(np.uint8)

        if mask.sum() == 0:
            continue

        poly = np.array(ann["poly"], dtype=np.float32).reshape(-1, 2)

        if show_obb:
            obb_patch = MplPolygon(
                poly,
                closed=True,
                fill=False,
                edgecolor="white",
                linewidth=1.5
            )
            ax.add_patch(obb_patch)

        if show_contour:
            contours, _ = cv2.findContours(
                mask,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )

            for cnt in contours:
                cnt = cnt.squeeze()

                if cnt.ndim == 2 and len(cnt) >= 3:
                    ax.plot(
                        cnt[:, 0],
                        cnt[:, 1],
                        color="red",
                        linewidth=2
                    )

        if show_label:
            ys, xs = np.where(mask > 0)

            if len(xs) > 0:
                cx = int(xs.mean())
                cy = int(ys.mean())

                label = item.get("category", ann.get("name", "object"))

                if show_score:
                    label += f"\nscore={score:.2f}"

                ax.text(
                    cx,
                    cy,
                    label,
                    color="yellow",
                    fontsize=8,
                    ha="center",
                    va="center",
                    bbox=dict(
                        facecolor="black",
                        alpha=0.7,
                        edgecolor="none"
                    )
                )

    ax.set_title("White = OBB | Red = mask contour | Color = generated mask")
    plt.tight_layout()
    plt.show()