from ultralytics import YOLO
import cv2,os
import numpy as np  
from utils.ROI_drawing import select_roi_from_video
from utils.properties import check_roi,create_directory
from utils.make_video import VideoEncoder
from utils.Image_manager import *
# load yolov8 model
model = YOLO('models\yolov8n (1).pt')

def save_images(image_queue_manager, save_path,cid):
    for tracking_id, priority_queue in image_queue_manager.queue_dict.items():
        if priority_queue:
            # Create a directory for each tracking ID
            tracking_id_dir = os.path.join(save_path, f"{tracking_id}")
            os.makedirs(tracking_id_dir, exist_ok=True)

            # Save images in the format camid_trackid_index
            for index, (confidence, image_array) in enumerate(priority_queue.get_top_pairs()):
                image_filename = f"{tracking_id}_{cid}_{index}.jpg"
                image_path = os.path.join(tracking_id_dir, image_filename)

                # Assuming image_array is a valid image data (e.g., numpy array)
                # Save the image to the specified path
                # You may need to adjust this part based on your actual image data and saving method
                save_image(image_array, image_path)

def save_image(image_array, image_path):
    # Assuming image_array is a valid image data (e.g., numpy array)
    # Save the image to the specified path
    # You may need to adjust this part based on your actual image data and saving method
    # For example, if image_array is a NumPy array representing an image, you can use a library like OpenCV:
    import cv2
    cv2.imwrite(image_path, image_array)
# load video
video_path = 'c4_v3_s1_5fps.mp4'
cap = cv2.VideoCapture(video_path)
dir_name = os.path.join("generated_data",video_path[:5])
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
img_queue=ImageQueueManager()
while ret:
    ret, frame = cap.read()
    if ret:
        # Perform object tracking
        results = model.track(frame,classes=[0],conf=0.1,persist=True)
        # Get the boxes and track IDs
        boxes = results[0].boxes.xyxy.cpu()
        annotated_frame = results[0].plot()
        cv2.imshow("YOLOv8 Tracking", annotated_frame)
        try:
            track_ids = results[0].boxes.id.int().cpu().tolist()
            conf_values=results[0].boxes.conf.float().cpu().tolist()
        except AttributeError:
            print("no object detected, skipping frame")
            continue
        
        for idx in range(len(track_ids)):
            
            # Extract bounding boxes
            x1, y1, x2, y2 = boxes.numpy()[idx]
            center_x,center_y=(x1+x2)/2,(y1+y2)/2
            if not check_roi(center_x,center_y,roi):
                continue
            confidence=conf_values[idx]
        #     # Crop the object using the bounding box coordinates
            cropped_image = frame[int(y1):int(y2), int(x1):int(x2)]
            image_pair=(cropped_image,confidence)
            img_queue.add_image(track_ids[idx],image_pair)
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        frame_num += 1
savepath="generated_data\c4_v3"
save_images(img_queue,savepath,cid="c4")
print("Tracking Completed")
cap.release()
cv2.destroyAllWindows()
# Release video capture and destroy the OpenCV window
