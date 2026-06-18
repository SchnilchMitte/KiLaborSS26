from multiprocessing import freeze_support
from ultralytics import YOLO


if __name__ == '__main__':
    print("Starting Base Model DOTA8")
    freeze_support()
    print("Loading YOLO model...")
    model = YOLO('yolov8n-obb.yaml')
    results = model.train(data="DOTA8.yaml",
                          epochs=100,
                          imgsz=1024,
                          batch=3,
                          pretrained=False,
                          seed=42,)

