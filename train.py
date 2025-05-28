from ultralytics import YOLO

# Load a YOLOv8n model config (starts training from scratch)
model = YOLO("yolo11n")  # or use 'yolov8n.pt' for transfer learning

# Train on the generated dataset
model.train(
    data="yolo_dataset/data.yaml",
    epochs=10,
    imgsz=640,
    batch=16
)

model.save("yolo11n_malaria_fine_tuned.pt")
