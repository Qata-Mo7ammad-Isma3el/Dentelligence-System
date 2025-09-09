from ultralytics import YOLO

model = YOLO("/content/drive/MyDrive/configs/yolo/dentex_yolov8x.yaml", task="detect")
model.load("/content/drive/MyDrive/runs/detect/train/weights/best.pt")
model.train(data="/content/drive/MyDrive/configs/yolo/dentex_disease_all_dataset.yaml", epochs=100)


# from ultralytics import YOLO

# model = YOLO("/content/drive/MyDrive/configs/yolo/dentex_yolov8x.yaml", task="detect")
# model.load("/content/drive/MyDrive/checkpoints/yolov8x.pt")
# model.train(data="/content/drive/MyDrive/configs/yolo/dentex_disease_dataset.yaml", epochs=100)

