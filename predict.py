from pathlib import Path
import pandas as pd
from ultralytics import YOLO

def run_inference(
    model_path: str,
    image_folder: str,
    conf_threshold: float = 0.25
) -> pd.DataFrame:
    """
    Run inference on a folder of images using a trained YOLO model.

    Returns a DataFrame with columns:
        ['Image_ID', 'class', 'confidence', 'ymin', 'xmin', 'ymax', 'xmax']
    """
    # Load model
    model = YOLO(model_path)

    # Prepare image paths
    image_folder = Path(image_folder)
    img_paths = sorted([p for p in image_folder.iterdir() if p.suffix.lower() in ['.jpg', '.png', '.jpeg']])

    records = []

    # Run prediction
    results = model.predict(
        source=[str(p) for p in img_paths],
        conf=conf_threshold,
        save=False  # don't save annotated images
    )

    # Parse detections
    for res in results:
        img_path = Path(res.path)
        image_id = img_path.name

        if res.boxes is None or len(res.boxes) == 0:
            records.append({
                'Image_ID': image_id,
                'class': 'NEG',
                'confidence': 0.0,
                'ymin': 0.0,
                'xmin': 0.0,
                'ymax': 0.0,
                'xmax': 0.0,
            })
            continue

        boxes = res.boxes.xyxy.cpu().numpy()  # [[x1, y1, x2, y2], ...]
        scores = res.boxes.conf.cpu().numpy()  # [score1, score2, ...]
        cls_ids = res.boxes.cls.cpu().numpy().astype(int)  # [cls1, cls2, ...]

        # Map class IDs to names
        names = model.names

        for (x1, y1, x2, y2), score, cid in zip(boxes, scores, cls_ids):
            class_name = names[cid]
            records.append({
                'Image_ID': image_id,
                'class': class_name,
                'confidence': float(score),
                'ymin': float(y1),
                'xmin': float(x1),
                'ymax': float(y2),
                'xmax': float(x2),
            })

    # Build DataFrame
    df = pd.DataFrame.from_records(records,
        columns=['Image_ID', 'class', 'confidence', 'ymin', 'xmin', 'ymax', 'xmax']
    )
    return df


if __name__ == '__main__':
    model_path = "yolo11n_malaria_fine_tuned.pt"
    image_folder = "data/test_images"
    output = "submission.csv"

    df = run_inference(model_path, image_folder, conf_threshold=0.01)

    df.to_csv(output, index=False)
    print(f"Saved predictions to {output}")
