import os
import cv2
from tqdm import tqdm
from slice_images import slice_image, str_label_to_tuple, yolo_str_label_to_tuple

IMG_ROOT = "DOTAv1/images"
LBL_ROOT = "DOTAv1/labels"
OUT_ROOT = "DOTAv1_sliced"

SPLITS = ["train", "val", "test"]

# Spelling for labels is from DOTA file.
# Numbering same as YOLO yaml file. (yaml replaces dashes with spaces)
DEFAULT_CLASS_MAP = {
    "plane": 0,
    "ship": 1,
    "storage-tank": 2,
    "baseball-diamond": 3,
    "tennis-court": 4,
    "basketball-court": 5,
    "ground-track-field": 6,
    "harbor": 7,
    "bridge": 8,
    "large-vehicle": 9,
    "small-vehicle": 10,
    "helicopter": 11,
    "roundabout": 12,
    "soccer-ball-field": 13,
    "swimming-pool": 14
}

def reverse_dict(d: dict[str, int]) -> dict:
    rev = dict()
    for k, i in d.items():
        rev[i] = k
    return rev

# ----------------------------
# LABEL LOADING
# ----------------------------

def load_labels(label_path):
    if not os.path.exists(label_path):
        return []

    with open(label_path, "r") as f:
        lines = f.readlines()[2:]

    labels = [l.split() for l in lines]
    return labels

# ----------------------------
# CONVERSION TO YOLO OBB
# ----------------------------

def to_yolo_obb(label, img_w, img_h, class_map):
    """
    label format:
    x1 y1 x2 y2 x3 y3 x4 y4 class_name difficulty
    """

    coords = list(map(float, label[:8]))
    class_name = label[8]

    cls_id = class_map[class_name]

    # normalize points
    norm_coords = []
    for i in range(0, 8, 2):
        x = coords[i] / img_w
        y = coords[i + 1] / img_h
        norm_coords.extend([x, y])

    return [cls_id] + norm_coords


# ----------------------------
# SAVE FUNCTIONS
# ----------------------------

def save_txt(path, rows):
    with open(path, "w") as f:
        for r in rows:
            f.write(" ".join(map(str, r)) + "\n")


# ----------------------------
# PIPELINE
# ----------------------------

def process_split(split, class_map,
                  src_img_dir=IMG_ROOT,
                  src_lbl_dir=LBL_ROOT,
                  out_dir=OUT_ROOT):
    reversed_class_map = reverse_dict(class_map)
    
    img_dir = os.path.join(src_img_dir, split)
    lbl_dir = os.path.join(src_lbl_dir, f"{split}_original")

    out_img_dir = os.path.join(out_dir, "images", split)
    out_lbl_dota = os.path.join(out_dir, "labels", f"{split}_original")
    out_lbl_yolo = os.path.join(out_dir, "labels", split)

    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_lbl_dota, exist_ok=True)
    os.makedirs(out_lbl_yolo, exist_ok=True)

    if not os.path.exists(img_dir):
        print(f"Image directory {img_dir} does not exist. Skipping '{split}'.")
        return
    images = [f for f in os.listdir(img_dir) if f.endswith(".jpg")]

    for img_file in tqdm(images, desc=f"Processing {split}"):
        img_path = os.path.join(img_dir, img_file)
        label_path = os.path.join(lbl_dir, img_file.replace(".jpg", ".txt"))

        base = os.path.splitext(img_file)[0]

        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Failed to read image {img_path}")

        try:
            labels = load_labels(label_path)
            if len(labels) == 0:
                raise ValueError('Missing labels.')
            labels = [str_label_to_tuple(l) for l in labels]
        except Exception as e:
            # print(f"Error processing labels for {img_file}: {e}")
            # print("Test fallback")
            label_path = label_path.replace("_original", "")
            labels = load_labels(label_path)
            w, h, _ = img.shape
            
            labels = [yolo_str_label_to_tuple(l, (w, h), reversed_class_map) for l in labels]
            if len(labels) == 0:
                pass
                # print("[WARNING] No labels")

        slices = slice_image(
            img,
            slice_size=(1024, 1024),
            labels=labels,
            anchor="top-left"
        )

        for i, (img_slice, slice_labels) in enumerate(slices):
            name = f"{base}_{i+1}"

            h, w = img_slice.shape[:2]

            # save image
            img_out = os.path.join(out_img_dir, name + ".jpg")
            cv2.imwrite(img_out, img_slice)

            # save original labels (sliced)
            dota_out = os.path.join(out_lbl_dota, name + ".txt")
            save_txt(dota_out, slice_labels)

            # save YOLO OBB labels
            yolo_labels = []
            for l in slice_labels:
                yolo_labels.append(to_yolo_obb(l, w, h, class_map))

            yolo_out = os.path.join(out_lbl_yolo, name + ".txt")
            save_txt(yolo_out, yolo_labels)


if __name__ == "__main__":
    fwd = os.path.dirname(os.path.abspath(__file__)) # filedir
    basedir = os.path.abspath(os.path.join(fwd, "..", "dota8")) # 'filedir/../dota8'
    
    print(basedir)
    for split in SPLITS:
        process_split(split,
                    DEFAULT_CLASS_MAP,
                    src_img_dir=f"{basedir}/images",
                    src_lbl_dir=f"{basedir}/labels",
                    out_dir=f"{basedir}_sliced")
