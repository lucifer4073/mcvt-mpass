import threading
import cv2
import os
import numpy as np
from ultralytics import YOLO
from utils.ROI_drawing import select_roi_from_video
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
            
            if not reidentify:
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
    cv2.destroyAllWindows()


def run_multithreaded_tracking(video_paths, save_paths, cam_ids, reidentify, gallery, reid_model=''):
    threads = []
    ROI=[]
    for vid_path in video_paths:
        roi=select_roi_from_video(vid_path)
        ROI.append(roi)  
    for i, video_path in enumerate(video_paths):
        thread = threading.Thread(target=run_tracker_in_thread, args=(vid_path,), daemon=True)
        threads.append(thread)

    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()


         
if __name__ == "__main__":
    # Example Usage:
    video_files = ["path/to/video1.mp4", "path/to/video2.mp4"]
    save_paths = ["generated_data\cam1", "generated_data\cam2"]
    cam_ids = ["cam1", "cam2"]

    run_multithreaded_tracking(video_files, save_paths, cam_ids, reidentify=True, gallery=your_gallery_data, reid_model=your_reid_model)

"""
SOME MAJOR FUNCTION CHANGES MADE HERE

.

.
.

USE THE GIVEN UPDATED FUNCTION

"""

