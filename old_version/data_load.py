import numpy as np
import threading
from ultralytics import YOLO
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import torch
import sys
from tqdm import tqdm
sys.path.append('.')
from utils.Detector import Detector
from utils.properties import *
from utils.Image_manager import *
from utils.fps_conversion import convert_to_5fps
from utils.ROI_drawing import select_roi_from_video
import os
import re




def create_gallery(video_path, roi, save_path, model, cid):
    video = cv2.VideoCapture(video_path)  # Read the video file
    _, frame = video.read()
    os.makedirs(save_path, exist_ok=True)
    img_queue = ImageQueueManager()

    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    # Use tqdm to track progress
    for frame_idx in tqdm(range(total_frames), desc="Processing Frames", unit="frame"):
        ret, frame = video.read()
        if not ret:
            break

        roi_frame = frame[roi[1]:roi[3], roi[0]:roi[2]]
        results = model.track(roi_frame, persist=True, classes=[0], conf=0.2)
        boxes = results[0].boxes.xyxy.cpu()
        track_ids = results[0].boxes.id

        if track_ids is None:
            continue

        track_ids = track_ids.int().cpu().tolist()
        conf_values = results[0].boxes.conf.float().cpu().tolist()

        for idx in range(len(track_ids)):
            x1, y1, x2, y2 = boxes.numpy()[idx]
            confidence = conf_values[idx]
            cropped_image = roi_frame[int(y1):int(y2), int(x1):int(x2)]
            image_pair = (cropped_image, confidence)
            img_queue.add_image(track_ids[idx], image_pair)   
    save_images(img_queue,save_path,cid)

def thread_gallery_data(video_paths,save_paths,model,cam_ids):
    roi=[]
    for vid in video_paths:
        roi_temp=select_roi_from_video(vid)
        roi.append(roi_temp)
    gal_threads=[]
    for i in range(len(video_paths)):
        t = threading.Thread(target=create_gallery, args=(video_paths[i],
                                                          roi[i],save_paths[i],
                                                          model,cam_ids[i]))
        gal_threads.append(t)
    for thread in gal_threads:
        thread.start()
    for thread in gal_threads:
        thread.join()
    cv2.destroyAllWindows()

# video_paths=["D:\\visitor_dwell_time\\visitor-dwell-time\old_version\sample_data\c4.mp4"]

# save_paths=["D:\\visitor_dwell_time\\visitor-dwell-time\gallery\c4_v3"]

# model=YOLO("D:\\visitor_dwell_time\\visitor-dwell-time\models\yolov8n (1).pt")
# cam_ids=['c4']

# thread_gallery_data(video_paths,save_paths,model,cam_ids)



        


