import cv2
import numpy as np
from ultralytics import YOLO
import os

def resize_frame(frame, target_size):
    return cv2.resize(frame, (target_size, target_size), interpolation=cv2.INTER_AREA)

def main():
    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Load the YOLO model
    model_path = os.path.join(script_dir, 'runs', 'detect', 'yolov8n_vehicle_detection6', 'weights', 'best.pt')
    model = YOLO(model_path)

    # Open the video file
    video_path = os.path.join(script_dir, '..', 'model_data', 'testing_videos', 'Cars Moving On Road Stock Footage.mp4')
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video file.")
        return

    window_size = 700
    car_count = 0
    previous_detections = set()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = resize_frame(frame, window_size)
        results = model(frame)
        annotated_frame = results[0].plot()

        current_detections = set()
        for box in results[0].boxes.data:
            x1, y1, x2, y2, conf, class_id = box.cpu().numpy()
            if conf > 0.5:  # Confidence threshold
                center = ((x1 + x2) / 2, (y1 + y2) / 2)
                current_detections.add(center)

        # Count new cars
        new_cars = len(current_detections - previous_detections)
        # Count cars that left
        left_cars = len(previous_detections - current_detections)

        car_count += new_cars - left_cars
        previous_detections = current_detections

        # Draw car count on the frame
        cv2.putText(annotated_frame, f"Cars on screen: {car_count}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        cv2.imshow("Car Counter", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()