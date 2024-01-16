from ultralytics import YOLO
import cv2,os,time
import numpy as np  
from PIL import Image
import matplotlib.pyplot as plt
import torch
from threading_testing import run_tracker_in_thread,process_video_with_csv
import sys
sys.path.append('.')
from utils.roi_extract import *
from utils.ROI_drawing import *
from utils.properties import check_roi, create_directory
from utils.Image_manager import *
from utils.REID.sort_ids import *
from utils.plotting_funcs import *


cam_paths = [
    "old_version\sample_data\merged_videos\c10_merged.mp4",
    "old_version\sample_data\merged_videos\c4_merged.mp4",
    "old_version\sample_data\merged_videos\c3_merged.mp4"
]

gallery_paths = [
    "gallery\c10_2",
    "gallery\c4_2",
    "gallery\c3_2"
]

camids = ["c10", "c3", "c4"]

tracker_1 = YOLO('models\\yolov8n (1).pt')
tracker_2= YOLO('models\\yolov8n (1).pt')
tracker_3= YOLO('models\\yolov8n (1).pt')
reid_model = load_pretrained_weights("pretrained_models\\resnet50_alignedreid.pth")

# # Manually specify ROIs for each video
# rois = [
#     select_roi_from_video("old_version\sample_data\merged_videos\c10_merged.mp4"),
#     select_roi_from_video("old_version\sample_data\merged_videos\c4_merged.mp4"),
#     select_roi_from_video("old_version\sample_data\merged_videos\c3_merged.mp4")
# ]

# # Run tracker for the first video
# run_tracker_in_thread(cam_paths[0], tracker_1, 1, rois[0], gallery_paths[0], camids[0])
# cv2.destroyAllWindows()
# time.sleep(2)
# # Run tracker for the second video
# run_tracker_in_thread(cam_paths[1], tracker_2, 1, rois[1], gallery_paths[1], camids[1])
# cv2.destroyAllWindows()
# time.sleep(2)
# # Run tracker for the third video
# # run_tracker_in_thread(cam_paths[2], tracker_3, 1, rois[2], gallery_paths[2], camids[2])
# # cv2.destroyAllWindows()
# # time.sleep(2)
# new_ids_c4=assign_new_ids(reid_model,gallery_paths[1],gallery_paths[2])
# new_ids_c10=assign_new_ids(reid_model,gallery_paths[0],gallery_paths[2])

# print(new_ids_c4,new_ids_c10)
c3_ids=load_list_from_file("D:\\visitor_dwell_time\\visitor-dwell-time\gallery\c3_2\\tracked_ids.pkl")
id_dict=dict(zip(c3_ids,c3_ids))
process_video_with_csv(video_path="old_version\sample_data\merged_videos\c3_merged.mp4", 
                       csv_path="gallery\c3_2\\tracking_results_c3.csv", 
                       id_dict=id_dict, 
                       output_path="gallery\cam3.mp4")