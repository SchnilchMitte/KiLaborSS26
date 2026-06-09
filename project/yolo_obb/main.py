from ultralytics import YOLO

model = YOLO('yolov8n-obb.yaml')
results = model.train(data="DOTAv1.yaml", epochs=20, imgsz=1024//2)
