import os
import pandas as pd

from predict import run_inference
from metric import evaluate_lacuna_malaria_metric


if __name__ == '__main__':
    model_path = "yolo11n_malaria_fine_tuned.pt"
    image_folder = "yolo_dataset/images/val"
    output = "predictions.csv"

    df = run_inference(model_path, image_folder, conf_threshold=0.01)

    train_df = pd.read_csv("data/train.csv")
    val_ids = [os.path.basename(p) for p in os.listdir("yolo_dataset/images/val")]
    val_df = train_df[train_df['Image_ID'].isin(val_ids)]
    print(val_ids)
    print(len(val_ids))

    mAP = evaluate_lacuna_malaria_metric(val_df, df)
    print(f"mAP: {mAP}")

    df.to_csv(output, index=False)
    print(f"Saved predictions to {output}")