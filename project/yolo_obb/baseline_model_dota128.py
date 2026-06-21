from multiprocessing import freeze_support
from ultralytics import YOLO

if __name__ == '__main__':
    print("Starting Base Model DOTA128")
    freeze_support()
    model = YOLO('yolov8n-obb.yaml')
    results = model.train(data="DOTA128.yaml",
                          epochs=100,
                          imgsz=640,
                          batch=3,
                          pretrained=False,
                          seed=42,)

