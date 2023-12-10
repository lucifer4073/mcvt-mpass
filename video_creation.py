import cv2
from ultralytics import YOLO
import os

# Define the video path and output path
video_path = 'c4_v3_s1_5fps.mp4'
output_path = os.path.join("generated_data\\tbr",'c4_wreid.mp4')

# Load the YOLO model
model = YOLO('models\yolov8n (1).pt')  # Replace 'yolov5s.pt' with the path to your YOLO model

# Open the video capture
cap = cv2.VideoCapture(video_path)

# Get the video's width, height, and frame rate
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
frame_rate = cap.get(5)

# Define the codec and create a VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use appropriate codec (e.g., 'XVID' for AVI)
out = cv2.VideoWriter(output_path, fourcc, frame_rate, (frame_width, frame_height))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Perform object tracking using YOLO
    results = model.track(frame,classes=[0],persist=True)

    # Get the processed frame with annotated objects
    annotated_frame = results[0].plot()

    # Write the processed frame to the output video
    out.write(annotated_frame)

# Release the video capture and the VideoWriter
cap.release()
out.release()

cv2.destroyAllWindows()

print(f"Video tracking and saving completed. The processed video is saved as {output_path}")
