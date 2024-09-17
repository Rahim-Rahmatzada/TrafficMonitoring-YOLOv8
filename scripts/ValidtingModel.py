import torch
from ultralytics import YOLO
import os
import numpy as np


def validate_yolo_model():
    if torch.cuda.is_available():
        print("CUDA is available. Using GPU for validation.")
        device = 'cuda'
    else:
        print("CUDA is not available. Using CPU for validation.")
        device = 'cpu'

    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_yaml_path = os.path.join(current_dir, '..', 'model_data', 'data.yaml')
    weights_path = os.path.join(current_dir, 'runs', 'detect', 'yolov8n_vehicle_detection6', 'weights', 'best.pt')

    print(f"Full path to data.yaml: {os.path.abspath(data_yaml_path)}")
    print(f"Full path to weights: {os.path.abspath(weights_path)}")

    if not os.path.exists(data_yaml_path):
        raise FileNotFoundError(f"data.yaml not found at {data_yaml_path}")
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Weights file not found at {weights_path}")

    model = YOLO(weights_path)
    results = model.val(data=data_yaml_path, device=device)

    print("\nValidation Results:")
    print(f"mAP50: {results.box.map50:.4f}")
    print(f"mAP50-95: {results.box.map:.4f}")
    print(f"Precision: {np.mean(results.box.p):.4f}")  # Average precision across classes
    print(f"Recall: {np.mean(results.box.r):.4f}")  # Average recall across classes

    # Validation Results:
    # mAP50: 0.9726
    # mAP50-95: 0.7337
    # Precision: 0.9174
    # Recall: 0.9402

if __name__ == "__main__":
    validate_yolo_model()