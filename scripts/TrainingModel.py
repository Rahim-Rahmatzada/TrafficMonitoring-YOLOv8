import torch
from ultralytics import YOLO
import os


def train_yolo_model():
    if torch.cuda.is_available():
        print("CUDA is available. Using GPU for training.")
        device = 'cuda'
    else:
        print("CUDA is not available. Using CPU for training.")
        device = 'cpu'

    # Get the current script's directory
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Construct the path to data.yaml
    data_yaml_path = os.path.join(current_dir, '..', 'model_data', 'data.yaml')

    # Print the full path for debugging
    print(f"Full path to data.yaml: {os.path.abspath(data_yaml_path)}")

    if not os.path.exists(data_yaml_path):
        raise FileNotFoundError(f"data.yaml not found at {data_yaml_path}")

    # Load a pretrained YOLOv8n model
    model = YOLO('yolov8n.pt')

    # Train the model using our custom dataset
    results = model.train(
        data=data_yaml_path,
        epochs=100,
        imgsz=640,
        batch=32,
        name='yolov8n_vehicle_detection',
        device=device
    )

    # Validate the model
    results = model.val(device=device)

    print("Training and validation complete.")


if __name__ == "__main__":
    train_yolo_model()