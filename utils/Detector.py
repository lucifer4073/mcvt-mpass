import os
import tqdm
import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
from tqdm import tqdm
from utils.properties import check_roi
from utils.properties import calculate_bbox_center
class Detector:
    """
    A class for detecting and tracking objects in a video.

    Args:
        video_path (str): The path to the video file.
        ROI (numpy.ndarray): The Region of Interest (ROI) as a NumPy array [XMIN, YMIN, XMAX, YMAX].
        model (ultralytics.YOLO): The YOLO model for object detection and tracking.
        num_frames (int, optional): Number of frames to process, default is 100.
        fps (int, optional): Frames per second, default is 5.
        height (int, optional): Height of the frames, default is 650.
        width (int, optional): Width of the frames, default is 800.
        save_frames (bool, optional): Whether to save annotated frames, default is False.
        out_path (str, optional): Output directory for saving annotated frames, required if save_frames is True.

    Methods:
        detect:
            Processes the video frames, detects and tracks objects, and returns annotated frames with DataFrames.
    """

    def __init__(self, video_path, ROI, model, num_frames=100, fps=5, height=650, width=800, save_frames=False, out_path=""):
        """
        Initializes a Detector instance.

        Args:
            video_path (str): The path to the video file.
            ROI (numpy.ndarray): The Region of Interest (ROI) as a NumPy array [XMIN, YMIN, XMAX, YMAX].
            model (ultralytics.YOLO): The YOLO model for object detection and tracking.
            num_frames (int, optional): Number of frames to process, default is 100.
            fps (int, optional): Frames per second, default is 5.
            height (int, optional): Height of the frames, default is 650.
            width (int, optional): Width of the frames, default is 800.
            save_frames (bool, optional): Whether to save annotated frames, default is False.
            out_path (str, optional): Output directory for saving annotated frames, required if save_frames is True.
        """
        self.video_path = video_path
        self.ROI = ROI
        self.model = model
        self.height = height
        self.width = width
        self.fps = fps
        self.num_frames = num_frames
        self.save = save_frames
        self.out_path = out_path
    def detect_and_track_video(self):
        """
        Processes video frames, detects and tracks objects, and returns annotated frames with DataFrames.

        This method iterates through the video frames, performs object detection and tracking,
        and annotates the frames with bounding boxes. It also saves annotated frames if specified.

        Returns:
            list: A list of frame and DataFrame pairs.
        """
        video = cv2.VideoCapture(self.video_path)
        frame_list = []  # type -> frame : (track values)
        cnt = 0
        for i in tqdm(range(self.num_frames), ncols=200):
            _, frame = video.read()
            result = self.model.track(frame,classes=[0],persist=True)

            if self.save:
                os.makedirs(self.out_path, exist_ok=True)
                frame_filename = os.path.join(self.out_path, f"frame_{cnt:04d}.jpg")
                cv2.imwrite(frame_filename, frame)
                cnt += 1

            annotated_frame = result[0].plot()
            XMIN, YMIN, XMAX, YMAX = self.ROI.astype(int)
            cv2.rectangle(annotated_frame, (XMIN, YMIN), (XMAX, YMAX), (0, 255, 0), 5)
            try:
                 df = pd.DataFrame(result[0].cpu().numpy().boxes.data,
                                columns=['xmin', 'ymin', 'xmax', 'ymax', 'id', 'conf'])
            except:
                print("Skipping frame")
                continue

            curr = (annotated_frame, df)
            frame_list.append(curr)

        return frame_list

    # def detect_and_crop (self):
    #     images_by_id = dict()
    #     ids_per_frame = []
    #     track_cnt=dict()
    #     annotated_frame_list=[]
    #     video = cv2.VideoCapture(self.video_path)
    #     for i in tqdm(range(self.num_frames)):
    #         _, frame = video.read()
    #         annotated_frame=result[0].plot()
    #         annotated_frame_list.append(annotated_frame)
    #         tmp_ids=set()
    #         frame_cnt=0
    #         result = self.model.track(frame,classes=[0],persist=True)
    #         boxes=result[0].boxes.xyxy.cpu()
    #         b2=result[0].boxes.xywh.cpu()
    #         x,y,w,h=b2
    #         center_x,center_y=calculate_bbox_center(x,y,w,h)
    #         try:
    #             track_ids = result[0].boxes.id.int().cpu().tolist()
    #         except AttributeError:
    #             print("no object detected, skipping frame")
    #         for box, track_id in zip(boxes, track_ids):
    #             if not check_roi(center_x,center_y,self.roi):
    #                 continue
    #             x1, y1, x2, y2 = box
    #             ultralytics_crop_object = frame[int(y1):int(y2), int(x1):int(x2)]
    #             if track_id not in track_cnt:
    #                 track_cnt[track_id]=1
    #                 images_by_id[track_id]=[ultralytics_crop_object]
    #             else:
    #                 images_by_id[track_id].append(ultralytics_crop_object)
    #             tmp_ids.add(track_id)
            
            

            

