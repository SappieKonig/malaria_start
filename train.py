from ultralytics import YOLO
from omegaconf import DictConfig
import hydra
import wandb


@hydra.main(config_path="configs", config_name="base_config")
def train_model(config: DictConfig):
    model = YOLO("yolo11n")  # or use 'yolov8n.pt' for transfer learning

    wandb.init()

    # Train on the generated dataset
    results = model.train(
        data="yolo_dataset/data.yaml",
        epochs=config.training.num_epochs,
        imgsz=config.training.imgsz,
        batch=config.training.batch_size,
        lr0=config.training.learning_rate
    )
    # wandb.log({"val_accuracy": float(results.results_dict['metrics/mAP50(B)'])})


if __name__ == "__main__":
    train_model()
    
