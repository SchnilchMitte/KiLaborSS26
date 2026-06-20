from multiprocessing import freeze_support
from ultralytics import YOLO

if __name__ == '__main__':
    freeze_support()
    model = YOLO('yolov8n-obb.yaml')
    results = model.train(data="DOTAv1.5.yaml",
                          time=.5,
                          imgsz=1024,
                          batch=-1,
                          pretrained=False,
                          seed=42,)
