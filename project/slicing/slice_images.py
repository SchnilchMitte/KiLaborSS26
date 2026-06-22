import numpy as np
from typing import Optional

type DOTALabel = tuple[int, int, int, int, int, int, int, int, str, int]
type AnchorPoint = "center" | "top-left"


def _bbox_from_label(label: DOTALabel) -> tuple[int, int, int, int]:
    xs = label[0:8:2]
    ys = label[1:8:2]
    return min(xs), min(ys), max(xs), max(ys)


def _intersects(box: tuple[int, int, int, int],
                obj: tuple[int, int, int, int]) -> bool:
    x0, y0, x1, y1 = box
    ox0, oy0, ox1, oy1 = obj
    return not (ox1 <= x0 or ox0 >= x1 or oy1 <= y0 or oy0 >= y1)


def _inside(box: tuple[int, int, int, int],
            obj: tuple[int, int, int, int]) -> bool:
    x0, y0, x1, y1 = box
    ox0, oy0, ox1, oy1 = obj
    return ox0 >= x0 and oy0 >= y0 and ox1 <= x1 and oy1 <= y1


def compute_slice_boxes(
    image_shape: tuple[int, int],
    slice_size: tuple[int, int],
    anchor: AnchorPoint,
) -> list[tuple[int, int, int, int]]:
    """Returns (x0, y0, x1, y1) for every slice."""
    H, W = image_shape
    sh, sw = slice_size

    boxes = []

    if anchor == "top-left":
        x_starts = list(range(0, W, sw))
        y_starts = list(range(0, H, sh))

    else:  # center
        x_remainder = W % sw
        y_remainder = H % sh
        x_start_offset = x_remainder // 2
        y_start_offset = y_remainder // 2

        x_starts = list(range(x_start_offset, W, sw))
        y_starts = list(range(y_start_offset, H, sh))

    for y0 in y_starts:
        for x0 in x_starts:
            x1 = min(x0 + sw, W)
            y1 = min(y0 + sh, H)
            boxes.append((x0, y0, x1, y1))

    return boxes


def adjust_label(
    label: DOTALabel,
    slice_box: tuple[int, int, int, int],
    only_full: bool = False
) -> Optional[DOTALabel]:
    """
    Return label in slice coords if partially contained, else None.
    If only_full is True, return label only if fully contained.
    """
    x0, y0, x1, y1 = slice_box
    obj_box = _bbox_from_label(label)

    if only_full:
        if not _inside(slice_box, obj_box):
            return None
    else:
        if not _intersects(slice_box, obj_box):
            return None

    # shift coordinates into slice-local space
    shifted = []
    for i in range(0, 8, 2):
        shifted.append(int(label[i] - x0))
        shifted.append(int(label[i + 1] - y0))

    return (
        shifted[0], shifted[1],
        shifted[2], shifted[3],
        shifted[4], shifted[5],
        shifted[6], shifted[7],
        label[8],
        label[9],
    )


def slice_image(
    image: np.array,
    slice_size: tuple,
    labels: list[DOTALabel] = None,
    anchor: AnchorPoint = "center",
) -> list[tuple[np.array, list[DOTALabel]]]:
    """Slices image and optionally adjusts labels."""
    H, W = image.shape[:2]
    sh, sw = slice_size

    boxes = compute_slice_boxes((H, W), (sh, sw), anchor)

    results = []

    for box in boxes:
        x0, y0, x1, y1 = box
        crop = image[y0:y1, x0:x1]

        adjusted = []
        if labels:
            for lbl in labels:
                new_lbl = adjust_label(lbl, box, only_full=False)
                if new_lbl is not None:
                    adjusted.append(new_lbl)

        results.append((crop, adjusted))

    return results

def str_label_to_tuple(label: list[str]) -> DOTALabel:
    x1, y1, x2, y2, x3, y3, x4, y4, class_label, diff = label
    return int(x1), int(y1), int(x2), int(y2), int(x3), int(y3), int(x4), int(y4), class_label, int(diff)

