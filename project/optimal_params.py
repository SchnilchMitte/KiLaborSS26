from ultralytics import YOLO
from pprint import pprint

model = YOLO("yolo11n-obb.pt")

print(model.ckpt.keys())
pprint(model.ckpt.get("train_args", None))