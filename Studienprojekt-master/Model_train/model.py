from ultralytics import YOLO
from multiprocessing import freeze_support

def main():

    model = YOLO("yolov8s-seg.pt ")

    model.train(
       data="data.yaml",
        epochs=100,
        lr0=0.001,# CUDA
        workers=4,
         lrf=0.01,
        optimizer="AdamW",
        momentum=0.937,
        weight_decay=0.0005,
        imgsz=640,
        batch=32,
        device=0,
        mosaic=1.0,
        mixup=0.05,
        cutmix=0.05,
        translate=0.1,
        scale=0.05,
        hsv_s=0.1,
        hsv_v=0.1,
        patience= 20
    )

    metrics = model.val(device=0)
    print(metrics)

if __name__ == "__main__":
    freeze_support()   # 🔥 wichtig auf Windows
    main()
