import cv2
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import os

class DetectionTracking:
    def __init__(self,model_path):
        self.model = YOLO(model_path)
        self.tracker = DeepSort(max_age=50, n_init=5)  # Adjust these parameters as needed

    def process_frame(self, frame):
        CONFIDENCE_THRESHOLD = 0.7
        detections = self.model(frame, classes=[0])[0]

        deepsort_detections = []
        for data in detections.boxes.data.tolist():
            confidence = data[4]
            if float(confidence) < CONFIDENCE_THRESHOLD:
                continue
            bbox = [int(data[0]), int(data[1]), int(data[2]) - int(data[0]), int(data[3]) - int(data[1])]
            deepsort_detections.append((bbox, confidence, int(data[5])))

        tracks = self.tracker.update_tracks(deepsort_detections, frame=frame)
        # Extract IDs and bounding boxes in "xyxy" format
        ordered_data = [(track.track_id, track.to_tlbr()) for track in tracks]
        return ordered_data
    def annotate_frame(self, frame, ordered_data):
        for track_id, bbox in ordered_data:
            t,l,b,r = map(int, bbox)
            color = (0, 255, 0)  # Green bounding box
            cv2.rectangle(frame, (t,l), (b,r), color, 2)
            cv2.putText(frame, f"ID: {track_id}", (t-10, l), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        return frame