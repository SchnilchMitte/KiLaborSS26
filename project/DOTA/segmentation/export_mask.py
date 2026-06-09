import cv2
import numpy as np
from pathlib import Path

from project.DOTA.segmentation.segmentation import grabcut_from_obb


def mask_to_polygon(
    mask,
    epsilon_ratio=0.002,
    min_area=10,
    keep_largest=True
):


    mask_uint8 = (mask > 0).astype(np.uint8) * 255

    contours, _ = cv2.findContours(
        mask_uint8,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    polygons = []

    if len(contours) == 0:
        return polygons

    if keep_largest:
        contours = [max(contours, key=cv2.contourArea)]

    for contour in contours:
        area = cv2.contourArea(contour)

        if area < min_area:
            continue

        perimeter = cv2.arcLength(contour, True)
        epsilon = epsilon_ratio * perimeter

        approx = cv2.approxPolyDP(contour, epsilon, True)

        polygon = approx.reshape(-1, 2)

        if len(polygon) < 3:
            continue

        polygons.append(polygon.astype(int))

    return polygons

def save_mask_txt_for_image(
    img_id,
    anns,
    masks,
    out_dir="DOTA/train/labelMaskTxt",
    epsilon_ratio=0.002,
    min_area=10
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / f"{img_id}.txt"

    lines = []

    for ann, mask_item in zip(anns, masks):
        category = ann["name"]
        difficult = ann.get("difficult", 0)
        mask = mask_item["mask"]

        polygons = mask_to_polygon(
            mask,
            epsilon_ratio=epsilon_ratio,
            min_area=min_area,
            keep_largest=True
        )

        for poly in polygons:
            coords = []

            for x, y in poly:
                coords.append(str(int(x)))
                coords.append(str(int(y)))

            line = " ".join(coords + [category, str(difficult)])
            lines.append(line)

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"Saved: {out_path}")
    print(f"Amount Mask-Polygone: {len(lines)}")


def export_all_grabcut_masks_as_txt(
    dota,
    out_dir="DOTA/train/labelMaskTxt",
    margin=0.25,
    iters=5,
    restrict_to_obb=True,
    epsilon_ratio=0.002,
    min_area=10
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    img_ids = dota.getImgIds()

    for idx, img_id in enumerate(img_ids):
        print(f"[{idx + 1}/{len(img_ids)}] processing {img_id}")

        img_path = Path(dota.imagepath) / f"{img_id}.png"
        img_bgr = cv2.imread(str(img_path))

        if img_bgr is None:
            print(f"Couldnt cload pic: {img_path}")
            continue

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        anns = dota.loadAnns(imgId=img_id)

        lines = []

        for ann_idx, ann in enumerate(anns):
            category = ann["name"]
            difficult = ann.get("difficult", 0)

            try:
                mask = grabcut_from_obb(
                    img_rgb=img_rgb,
                    poly=ann["poly"],
                    margin=margin,
                    iters=iters,
                    restrict_to_obb=restrict_to_obb
                )
            except Exception as e:
                print(f"Error for {img_id}, Annotation {ann_idx}: {e}")
                continue

            polygons = mask_to_polygon(
                mask,
                epsilon_ratio=epsilon_ratio,
                min_area=min_area,
                keep_largest=True
            )

            for poly in polygons:
                coords = []

                for x, y in poly:
                    coords.append(str(int(x)))
                    coords.append(str(int(y)))

                line = " ".join(coords + [category, str(difficult)])
                lines.append(line)

        out_path = out_dir / f"{img_id}.txt"

        with open(out_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

    print("Done")
    print(f"Mask Txt saved in: {out_dir}")