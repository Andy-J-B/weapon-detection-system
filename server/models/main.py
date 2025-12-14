# This is the file that holds code that is run to train the YOLOv5 weapon detection model
from ultralytics import YOLO

# Load a model
if __name__ == "__main__":
    model = YOLO("yolov5su.pt")
    data_path = r"/Users/Andy_1/dev/code/programs/GitHub/weapon-detection-system/server/models/dataset.yaml"

    # Train the model
    train_results = model.train(
        data=data_path,  # path to dataset YAML
        # Core Parameters
        epochs=150,  # 100-150 epochs for convergence
        batch=8,  # Batch size: 8 is the recommended max for stable CPU training
        imgsz=640,  # Input size: 640x640 (standard)
        device="cpu",  # Hardware: Set to "cpu" as per your existing code
        # Optimization and Regularization
        optimizer="AdamW",  # Suggested optimizer for stability on small datasets
        lr0=0.001,  # Learning Rate: Lower base LR for CPU training (0.001) to prevent divergence
        weight_decay=0.0005,  # Regularization to prevent overfitting
        # Early Stopping and Checkpointing
        patience=10,  # Early stopping: stop if val-loss doesn't improve for 10 epochs
        save_period=5,  # Checkpointing: Save a model every 5 epochs
        # Augmentation (Set to True/Defaults, Flips are added explicitly)
        augment=True,  # Enables Mosaic, MixUp, and other standard augments
        flipud=0.5,  # Augmentation: Vertical flip (50% chance) for robustness
        fliplr=0.5,  # Augmentation: Horizontal flip (50% chance) for robustness
        # Project Management
        project="server/models/runs/train",  # Main directory for results
        name="weapon_v5s_640_cpu",  # Specific run name for organization
    )

# yolo export model=best.pt format=onnx opset=12
