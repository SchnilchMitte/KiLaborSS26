from multiprocessing import freeze_support
from ultralytics import YOLO

if __name__ == '__main__':
    freeze_support()
    model = YOLO('yolov8n-obb.yaml')
    results = model.train(data="DOTAv1.yaml",
                          time=.5,
                          imgsz=1024,
                          batch=3,
                          pretrained=False,
                          seed=42,)
