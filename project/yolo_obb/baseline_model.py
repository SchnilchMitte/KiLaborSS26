from multiprocessing import freeze_support
from ultralytics import YOLO



def resume_from_last_checkpoint(path):
    print("Resuming Base Model DOTA128")
    freeze_support()

    model = YOLO(path)

    results = model.train(
        resume=True,
        workers=0,
        cache=False,
    )

def start_from_beginning():
    print("Starting base model DOTA128")
    freeze_support()
    model = YOLO('yolov8n-obb.yaml')
    results = model.train(data="DOTAv1.yaml",
                          epochs=100,
                          imgsz=1024,
                          batch=3,
                          pretrained=False,
                          degrees=180,
                          seed=42,)


if __name__ == '__main__':
    #resume_from_last_checkpoint("../../runs/obb/train-3/weights/last.pt")
    start_from_beginning()