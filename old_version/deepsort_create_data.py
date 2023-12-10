from ultralytics import YOLO
import cv2,os
import numpy as np  
from utils.ROI_drawing import select_roi_from_video
from utils.properties import check_roi,create_directory
from utils.make_video import VideoEncoder
from utils.New_tracker import DetectionTracking
# load yolov8 model
model = YOLO('models\yolov8n (1).pt')

def calculate_bbox_center(x, y, width, height):
    center_x = x + (width / 2)
    center_y = y + (height / 2)
    return center_x, center_y
# load video
video_path = 'c3_v1_s1_5fps.mp4'
cap = cv2.VideoCapture(video_path)
dir_name = "generated_data\c3_v1_dsort"
create_directory(dir_name)
# Create an OpenCV window
cv2.namedWindow('frame', cv2.WINDOW_NORMAL)  # This line sets the window to be resizable

ret = True

os.makedirs(dir_name, exist_ok=True)  # Ensure the directory exists
roi = np.array(select_roi_from_video(video_path)) 
# read frames
frame_num = 0
cam_name=video_path[:2]
annotated_frame_list=[]
tracker=DetectionTracking(model_path='models\yolov8n (1).pt')
while ret:
    ret, frame = cap.read()
    if ret:
        
        data=tracker.process_frame(frame)
        for track_id,bbox in data:
            
            # Extract bounding boxes
            t,l,b,r=bbox
            center_x,center_y=(l+r)/2,(t+b)/2
            if not check_roi(center_x,center_y,roi):
                continue
            
            file_name = os.path.join(dir_name, f"{cam_name}_{track_id}_{frame_num}.jpg")  # Include file extension

        #     # Crop the object using the bounding box coordinates
            ultralytics_crop_object = frame[int(t):int(b), int(l):int(r)]

        #     # Save the cropped object as an image
            cv2.imwrite(file_name, ultralytics_crop_object)

        #     # Visualize the results on the frame
        annotated_frame = tracker.annotate_frame(frame,data)
        annotated_frame_list.append(annotated_frame)
        # Display the annotated frame
        cv2.imshow("YOLOv8 Tracking", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        frame_num += 1
print("Tracking Completed")
cap.release()
cv2.destroyAllWindows()
encoder=VideoEncoder()
print("Encoding video")
encoder.encode_frames(frame_list=annotated_frame_list,out_path="generated_data\\tbr")
print("Encoding completed")
# Release video capture and destroy the OpenCV window
