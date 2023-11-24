# Display image and videos
import numpy as np
from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
import torch
from utils.Detector import Detector
from utils.properties import *
from utils.detect_time import Timer
from utils.make_video import VideoEncoder
from utils.Dataset import FrameExtractor
from utils.fps_conversion import convert_to_5fps
from utils.ROI_drawing import select_roi_from_video
import os
from reid.REID import REID
from tqdm import tqdm
import operator
# Check if CUDA is available for GPU acceleration
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load YOLO models
yolov8=YOLO("models\yolov8n (1).pt")

# Input video path
path = "Shortit.mp4"
# Initialize properties of the video
prop = Properties(path)
roi = np.array(select_roi_from_video(path))  # Select a Region of Interest (ROI)
org_height, org_width, org_fps, org_num_frames = prop.height(), prop.width(), prop.fps(), prop.num_frames()
prop.display_properties()
num_frames = org_num_frames  # Number of frames to process
def cv2_addBox(track_id, frame, x1, y1, x2, y2, line_thickness, text_thickness, text_scale):
    color = get_color(abs(track_id))
    cv2.rectangle(frame, (x1, y1), (x2, y2), color=color, thickness=line_thickness)
    cv2.putText(
        frame, str(track_id), (x1, y1 + 30), cv2.FONT_HERSHEY_PLAIN, text_scale, (0, 0, 255), thickness=text_thickness)
def get_color(idx):
    idx = idx * 3
    color = ((37 * idx) % 255, (17 * idx) % 255, (29 * idx) % 255)
    return color

def get_FrameLabels(frame):
    text_scale = max(1, frame.shape[1] / 1600.)
    text_thickness = 1 if text_scale > 1.1 else 1
    line_thickness = max(1, int(frame.shape[1] / 500.))
    return text_scale, text_thickness, line_thickness
track_cnt = dict()
images_by_id = dict()
ids_per_frame = []
# Initialize the detector to detect and track objects in the video
annotated_frame_list=[]
original_frame_list=[]
video = cv2.VideoCapture(path)
# print(num_frames)

for i in tqdm(range(int(num_frames))):
    _, frame = video.read()
    original_frame_list.append(frame)
    result = yolov8.track(frame,classes=[0],persist=True)
    annotated_frame=result[0].plot()
    annotated_frame_list.append(annotated_frame)
    tmp_ids=set()
    frame_cnt=0
    boxes=result[0].boxes.xyxy.cpu()
    try:
        track_ids = result[0].boxes.id.int().cpu().tolist()
    except AttributeError:
        print("no object detected, skipping frame")
        continue
    for idx in range(len(track_ids)):
        x1, y1, x2, y2 = boxes[idx]

        center_x,center_y=(x1+x2)/2,(y1+y2)/2
        if not check_roi(center_x,center_y,roi):
            continue
        area=int(((x2-x1)*(y2-y1)).item())
        ultralytics_crop_object = frame[int(y1):int(y2), int(x1):int(x2)]
        if track_ids[idx] not in track_cnt:
            track_cnt[track_ids[idx]]=[[i,int(x1.item()),
            int(y1.item()),int(x2.item()),int(y2.item()),area]]
            images_by_id[track_ids[idx]]=[ultralytics_crop_object]
        else:
            track_cnt[track_ids[idx]].append([i,int(x1.item()),
            int(y1.item()),int(x2.item()),int(y2.item()),area])
            images_by_id[track_ids[idx]].append(ultralytics_crop_object)
        tmp_ids.add(track_ids[idx])
    ids_per_frame.append(tmp_ids)

reid = REID(model_path="models\\resnet_50_pret.pth")
threshold = 320
exist_ids = set()
final_fuse_id = dict()
print(f'Total IDs = {len(images_by_id)}')
feats = dict()
for i in images_by_id:
    print(f'ID number {i} -> Number of frames {len(images_by_id[i])}')
    feats[i] = reid._features(images_by_id[i])  # reid._features(images_by_id[i][:min(len(images_by_id[i]),100)])

for f in ids_per_frame:
        if f:
            if len(exist_ids) == 0:
                for i in f:
                    final_fuse_id[i] = [i]
                exist_ids = exist_ids or f
            else:
                new_ids = f - exist_ids
                for nid in new_ids:
                    dis = []
                    if len(images_by_id[nid]) < 10:
                        exist_ids.add(nid)
                        continue
                    unpickable = []
                    for i in f:
                        for key, item in final_fuse_id.items():
                            if i in item:
                                unpickable += final_fuse_id[key]
                    print('exist_ids {} unpickable {}'.format(exist_ids, unpickable))
                    for oid in (exist_ids - set(unpickable)) & set(final_fuse_id.keys()):
                        tmp = np.mean(reid.compute_distance(feats[nid], feats[oid]))
                        print('nid {}, oid {}, tmp {}'.format(nid, oid, tmp))
                        dis.append([oid, tmp])
                    exist_ids.add(nid)
                    if not dis:
                        final_fuse_id[nid] = [nid]
                        continue
                    dis.sort(key=operator.itemgetter(1))
                    if dis[0][1] < threshold:
                        combined_id = dis[0][0]
                        images_by_id[combined_id] += images_by_id[nid]
                        final_fuse_id[combined_id].append(nid)
                    else:
                        final_fuse_id[nid] = [nid]

print(f'Total IDs = {len(images_by_id)}')
print(f"Final fs= {final_fuse_id}")
print(f"exist_ids= {exist_ids}")
print(f"tracke_cnt={track_cnt}")
VIDEO_CODEC = "MP4V"
output_path=os.path.join("generated_data\\tbr","vid-reid_2.mp4")
if os.path.exists(output_path):
     delete_directory(output_path)
fourcc = cv2.VideoWriter_fourcc(*VIDEO_CODEC)
out = cv2.VideoWriter(output_path, fourcc, org_fps, (int(org_width), int(org_height))) 
for frame in range(len(original_frame_list)):
        frame2 = original_frame_list[frame]
        for idx in final_fuse_id:
            for i in final_fuse_id[idx]:
                for f in track_cnt[i]:
                    # print('frame {} f0 {}'.format(frame,f[0]))
                    # print(track_cnt[i])
                    # print(f,frame)
                    if frame == f[0]:
                        text_scale, text_thickness, line_thickness = get_FrameLabels(frame2)
                        cv2_addBox(idx, frame2, f[1], f[2], f[3], f[4], line_thickness, text_thickness, text_scale)
        out.write(frame2)
out.release()
  








# # Save detected frames and DataFrames
# is_save=False
# if is_save:
#     save_data(data_list=frame_list, image_dir_path=image_dir_path, df_dir_path=df_dir_path)

# #[IMP] Put time code here
