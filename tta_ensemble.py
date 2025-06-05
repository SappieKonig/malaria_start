"""Very small script for flip-based test time augmentation.

It loads one or more YOLO models, predicts every image in a folder four times
(no flip, horizontal, vertical and both), undoes the flips and writes all
detections to a CSV file. This is meant purely as a teaching example and keeps
the implementation intentionally simple.
"""

from pathlib import Path
from typing import List, Optional
from PIL import Image, ImageOps
import pandas as pd
from ultralytics import YOLO


def _predict_single(model: YOLO, img_path: Path, flip: Optional[str], conf: float) -> List[dict]:
    img = Image.open(img_path).convert("RGB")
    w, h = img.size
    if flip == "h":
        img = ImageOps.mirror(img)
    elif flip == "v":
        img = ImageOps.flip(img)
    elif flip == "hv":
        img = ImageOps.flip(ImageOps.mirror(img))

    res = model.predict(img, conf=conf, save=False)[0]

    records = []
    if res.boxes is None or len(res.boxes) == 0:
        records.append({
            "Image_ID": img_path.name,
            "class": "NEG",
            "confidence": 0.0,
            "ymin": 0.0,
            "xmin": 0.0,
            "ymax": 0.0,
            "xmax": 0.0,
        })
        return records

    boxes = res.boxes.xyxy.cpu().numpy()
    scores = res.boxes.conf.cpu().numpy()
    cls_ids = res.boxes.cls.cpu().numpy().astype(int)
    for (x1, y1, x2, y2), score, cid in zip(boxes, scores, cls_ids):
        if flip in ("h", "hv"):
            x1, x2 = w - x2, w - x1
        if flip in ("v", "hv"):
            y1, y2 = h - y2, h - y1
        records.append({
            "Image_ID": img_path.name,
            "class": model.names[cid],
            "confidence": float(score),
            "ymin": float(y1),
            "xmin": float(x1),
            "ymax": float(y2),
            "xmax": float(x2),
        })
    return records


def run_tta_ensemble(model_paths: List[str], image_folder: str, conf_threshold: float = 0.25) -> pd.DataFrame:
    """Run a small flip-based ensemble on a folder of images.

    Parameters
    ----------
    model_paths : list[str]
        Paths to the YOLO model weights to use.
    image_folder : str
        Directory containing the images to predict.
    conf_threshold : float, default 0.25
        Minimum confidence score for detections.

    Returns
    -------
    pandas.DataFrame
        All detections from every model and TTA variant.
    """

    image_folder = Path(image_folder)
    img_paths = sorted([p for p in image_folder.iterdir() if p.suffix.lower() in [".jpg", ".png", ".jpeg"]])
    records = []
    for model_path in model_paths:
        model = YOLO(model_path)
        for img_path in img_paths:
            for flip in [None, "h", "v", "hv"]:
                records.extend(_predict_single(model, img_path, flip, conf_threshold))
    df = pd.DataFrame.from_records(
        records,
        columns=["Image_ID", "class", "confidence", "ymin", "xmin", "ymax", "xmax"],
    )
    return df


if __name__ == "__main__":
    models = ["yolo11n_malaria_fine_tuned.pt"]
    image_folder = "data/test_images"
    output = "tta_submission.csv"

    df = run_tta_ensemble(models, image_folder)
    df.to_csv(output, index=False)
    print(f"Saved predictions to {output}")
