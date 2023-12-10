from ultralytics import YOLO
import cv2
import os
import numpy as np
from utils.ROI_drawing import select_roi_from_video
from utils.properties import check_roi, create_directory
from utils.make_video import VideoEncoder
from utils.Image_manager import ImageQueueManager
from utils.Threads.threading import *
from utils.REID.alignedreid_utils import *

video_paths=["c3_v1_s1_5fps.mp4","c4_v3_s1_5fps.mp4"]
save_paths=["gallery\c3_v1","gallery\c4_v3"]

reidentify=[False,True]
gallery=[None,save_paths[0]]
camids=[3,4]
reid_model=load_pretrained_weights("pretrained_models\\resnet50_alignedreid.pth")

run_multithreaded_tracking(video_paths,save_paths,camids,reidentify,gallery,reid_model)