import threading
import cv2
import numpy as np
import sys
sys.path.append('.')
from ultralytics import YOLO
from utils.roi_extract import *
from utils.ROI_drawing import *
from utils.properties import check_roi, create_directory
from utils.Image_manager import *
from utils.REID.sort_ids import *
from utils.plotting_funcs import *



def run_tracker_in_thread(filename, model, file_index,roi,save_path,cam_id,reidentify=False, gallery=None, reid_model=''):
    """
    Runs a video file or webcam stream concurrently with the YOLOv8 model using threading.

    This function captures video frames from a given file or camera source and utilizes the YOLOv8 model for object
    tracking. The function runs in its own thread for concurrent processing.

    Args:
        filename (str): The path to the video file or the identifier for the webcam/external camera source.
        model (obj): The YOLOv8 model object.
        file_index (int): An index to uniquely identify the file being processed, used for display purposes.

    Note:
        Press 'q' to quit the video display window.
    """
    video = cv2.VideoCapture(filename)  # Read the video file
    _,frame=video.read()
    os.makedirs(save_path, exist_ok=True)
    frame_num = 0
    img_queue = ImageQueueManager()

    while True:
        ret, frame = video.read()  # Read the video frames
        
        # Exit the loop if no more frames in either video
        # Track objects in frames if available
        results = model.track(frame, persist=True,classes=[0],conf=0.2)
        boxes = results[0].boxes.xyxy.cpu()
        track_ids = results[0].boxes.id
        if track_ids is None:
            cv2.imshow(f"Tracking_Stream_{file_index}", frame)
            continue

        track_ids = track_ids.int().cpu().tolist()
        conf_values = results[0].boxes.conf.float().cpu().tolist()
        fnum=0
        for idx in range(len(track_ids)):
            x1, y1, x2, y2 = boxes.numpy()[idx]
            center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
            if not check_roi(center_x, center_y, roi):
                        continue

            confidence = conf_values[idx]
            cropped_image = frame[int(y1):int(y2), int(x1):int(x2)]
            image_pair = (cropped_image, confidence)
            
            if reidentify:
                    query_image_path = os.path.join("temporary_cache", "temp_img.jpg")
                    save_image(cropped_image, query_image_path)
                    closest_id, _ = find_matching_id_with_distances(reid_model, query_image_path, gallery)
                    track_ids[idx] = closest_id
                    delete_image(query_image_path)
            
            img_queue.add_image(track_ids[idx], image_pair)
            fnum+=1

        frame=plot_bbox(frame,roi)
        res_plotted = plot_bboxes_with_ids(frame,boxes,track_ids)
        cv2.imshow(f"Tracking_Stream_{file_index}", res_plotted)
        save_images(img_queue, save_path, cam_id)

        key = cv2.waitKey(1)
        if key == ord('q'):
            break

    # Release video sources
    video.release()

reid_model=load_pretrained_weights("pretrained_models\\resnet50_alignedreid.pth")
#save paths
s1=os.path.join("temporary_cache","c3")
s2=os.path.join("temporary_cache","c4")
# Load the models
model1 = YOLO('models\yolov8n (1).pt')
model2 = YOLO('models\yolov8n (1).pt')

#ids

cam_id1='c3'
cam_id2='c4'
# Define the video files for the trackers
video_file1 = "c3_v1_s1_5fps.mp4"  # Path to video file, 0 for webcam
video_file2 = "c4_v3_s1_5fps.mp4"  # Path to video file, 0 for webcam, 1 for external camera

#select roi
roi_1=select_roi_from_video(video_file1)
roi_2=select_roi_from_video(video_file2)

# Create the tracker threads
tracker_thread1 = threading.Thread(target=run_tracker_in_thread, args=(video_file1, model1, 1,roi_1,s1,cam_id1), daemon=True)
tracker_thread2 = threading.Thread(target=run_tracker_in_thread, args=(video_file2, model2, 2,roi_2,s2,cam_id2), daemon=True)

# Start the tracker threads
tracker_thread1.start()
tracker_thread2.start()

# Wait for the tracker threads to finish
tracker_thread1.join()
tracker_thread2.join()

# Clean up and close windows
cv2.destroyAllWindows()

#,True,s1,reid_model

