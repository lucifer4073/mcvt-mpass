import threading
import cv2
import numpy as np
import sys,time

from ultralytics import YOLO
from roi_extract import *
from ROI_drawing import *
from properties import check_roi, create_directory
from Image_manager import *
from REID.sort_ids import *
from plotting_funcs import *

from tqdm import trange

def run_tracker_without_reid(video_path,model,roi,save_path,cam_id,output_path):
    video=cv2.VideoCapture(video_path)
    os.makedirs(save_path, exist_ok=True)
    ids_in_roi = set() 
    # Get video properties
    frame_width = int(video.get(3))
    frame_height = int(video.get(4))
    fps = int(video.get(5))
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    print("Started processing...")

    #time counter
    t, inc = 0, round(1 / fps, 2)
    storage = dict()

    #img queue manager
    img_queue = ImageQueueManager()
    for idx_tp in trange(total_frames,desc="Tracking_video",unit='frames'):
        ret,frame=video.read()
        if not ret: 
            break
        results = model.track(frame, persist=True, classes=[0], conf=0.2)
        boxes = results[0].boxes.xyxy.cpu()
        track_ids = results[0].boxes.id
        if track_ids is None:
            out.write(frame)
            continue
        track_ids = track_ids.int().cpu().tolist()
        conf_values = results[0].boxes.conf.float().cpu().tolist()
        for idx in range(len(track_ids)):
            x1, y1, x2, y2 = boxes.numpy()[idx]
            center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
            if check_roi(center_x, center_y, roi):
                ids_in_roi.add(track_ids[idx])
                if storage.get(track_ids[idx],'invalid')=='invalid':
                    storage[track_ids[idx]]=t
                else:
                    time_in=t-storage[track_ids[idx]]
                    cv2.putText(frame, text="Time in:" + str(time_in), org=(int(x1), int(y2) - 15),
                                fontFace=cv2.FONT_HERSHEY_TRIPLEX,
                                fontScale=1, color=(255, 0, 0), thickness=2)
            confidence = conf_values[idx]
            cropped_image = frame[int(y1):int(y2), int(x1):int(x2)]
            image_pair = (cropped_image, confidence)

            img_queue.add_image(track_ids[idx], image_pair)
        frame = plot_bbox(frame, roi)
        frame = plot_bboxes_with_ids(frame, boxes, track_ids)
        t+=inc
        save_images(img_queue, save_path, cam_id)
        # Write the processed frame to output video file
        out.write(frame)

        # Break the loop if 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # Release resources
    video.release()
    out.release()
    print(f"Processed video saved at: {output_path}")
    return ids_in_roi

        
        
    pass
def run_tracker_with_reid(video_path,model,roi,save_path,cam_id,output_path,ids_list):
    video=cv2.VideoCapture(video_path)
    os.makedirs(save_path, exist_ok=True)

    # Get video properties
    frame_width = int(video.get(3))
    frame_height = int(video.get(4))
    fps = int(video.get(5))
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    print("Started processing...")
    ids_in_roi = set() 
    #time counter
    t, inc = 0, round(1 / fps, 2)
    storage = dict()

    #store new ids
    new_ids=dict()
    current_pt=0
    #img queue manager
    other_id=20
    img_queue = ImageQueueManager()
    for idx_tp in trange(total_frames,desc="Tracking_video",unit='frames'):
        ret,frame=video.read()
        if not ret: 
            break
        results = model.track(frame, persist=True, classes=[0], conf=0.2)
        boxes = results[0].boxes.xyxy.cpu()
        track_ids = results[0].boxes.id
        if track_ids is None:
            out.write(frame)
            continue
        track_ids = track_ids.int().cpu().tolist()
        conf_values = results[0].boxes.conf.float().cpu().tolist()
        for idx in range(len(track_ids)):
            x1, y1, x2, y2 = boxes.numpy()[idx]
            center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
            if check_roi(center_x, center_y, roi):
                if (new_ids.get(track_ids[idx],0)):
                    track_ids[idx]=new_ids[track_ids[idx]]
                else:
                    if (current_pt>=len(ids_list)):
                        current_pt-=1
                        new_ids[track_ids[idx]]=other_id
                        track_ids[idx]=other_id
                        # other_id+=1
                        print("Number of people in rois differ")
                    else:
                        new_ids[track_ids[idx]]=ids_list[current_pt]
                        track_ids[idx]=ids_list[current_pt]
                        current_pt+=1
                ids_in_roi.add(track_ids[idx])
            else:
                track_ids[idx]=new_ids.get(track_ids[idx],other_id)
                # other_id+=1

                if storage.get(track_ids[idx],'invalid')=='invalid':
                    storage[track_ids[idx]]=t
                else:
                    time_in=t-storage[track_ids[idx]]
                    cv2.putText(frame, text="Time in:" + str(time_in), org=(int(x1), int(y2) - 15),
                                fontFace=cv2.FONT_HERSHEY_TRIPLEX,
                                fontScale=1, color=(255, 0, 0), thickness=2)

            confidence = conf_values[idx]
            cropped_image = frame[int(y1):int(y2), int(x1):int(x2)]
            image_pair = (cropped_image, confidence)

            img_queue.add_image(track_ids[idx], image_pair)
        t+=inc
        frame = plot_bbox(frame, roi)
        frame = plot_bboxes_with_ids(frame, boxes, track_ids)
        
        save_images(img_queue, save_path, cam_id)
        # Write the processed frame to output video file
        out.write(frame)
        # cv2.imshow("Video stream",frame)
        # print(new_ids)
        # try:
        #     print(ids_list[current_pt])
        # except:
        #     print("Exceded")

        # Break the loop if 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # Release resources
    video.release()
    out.release()
    print(f"Processed video saved at: {output_path}")
    return ids_in_roi
def main():
    # Load the reid model
    reid_model = load_pretrained_weights("pretrained_models\\resnet50_alignedreid.pth")

    # Save paths
    s1 = "gallery\\c3_2"
    s2 = "gallery\\c4_2"
    s3 = "gallery\\c10_2"

    # Load the models
    model1 = YOLO('models\yolov8n (1).pt')
    model2 = YOLO('models\yolov8n (1).pt')
    model3 = YOLO('models\yolov8n (1).pt')

    # Camera IDs
    cam_id1 = 'c3'
    cam_id2 = 'c4'
    cam_id3 = 'c10'

    # Define the video files for the trackers
    video_file1 = "old_version\sample_data\merged_videos\c3_merged.mp4"
    video_file2 = "old_version\sample_data\merged_videos\c4_test.mp4"
    video_file3 = "old_version\sample_data\merged_videos\c10_merged.mp4"

    # Select ROI
    roi_1 = select_roi_from_video(video_file1)
    roi_2 = select_roi_from_video(video_file2)
    roi_3 = select_roi_from_video(video_file3)
    ids_in_roi=run_tracker_without_reid(video_file1,model1,roi_1,
                             s1,'c3',"gallery\cam3.mp4")
    # print(ids_in_roi)
    save_list_to_file(list(ids_in_roi),"gallery\ids.pkl")
    ids_in_roi = load_list_from_file("gallery\ids.pkl")
    print(ids_in_roi)
    # time.sleep(3)
    run_tracker_with_reid(video_file3,model3,roi_3,s3,cam_id3,"gallery\cam10.mp4",list(ids_in_roi))
    run_tracker_with_reid(video_file2, model2, roi_2, s2, cam_id2, "gallery\cam4.mp4", list(ids_in_roi))
    # print(ids_in_roi)

if __name__ == "__main__":
    main()
# if __name__=="__main__":
#     # Load the reid model
#     reid_model = load_pretrained_weights("pretrained_models\\resnet50_alignedreid.pth")

#     # Save paths
#     s1 = "gallery\\c3_2"
#     s2 = "gallery\\c4_2"
#     s3 = "gallery\\c10_2"

#     # Load the models
#     model1 = YOLO('models\yolov8n (1).pt')
#     model2 = YOLO('models\yolov8n (1).pt')
#     model3 = YOLO('models\yolov8n (1).pt')

#     # Camera IDs
#     cam_id1 = 'c3'
#     cam_id2 = 'c4'
#     cam_id3 = 'c10'

#     # Define the video files for the trackers
#     video_file1 = "old_version\sample_data\merged_videos\c3_merged.mp4"
#     video_file2 = "old_version\sample_data\merged_videos\c4_test.mp4"
#     video_file3 = "old_version\sample_data\merged_videos\c10_merged.mp4"

#     # Select ROI
#     # roi_1 = select_roi_from_video(video_file1)
#     roi_2 = select_roi_from_video(video_file2)
#     # roi_3 = select_roi_from_video(video_file3)
#     # ids_in_roi=run_tracker_without_reid(video_file1,model1,roi_1,
#     #                          s1,'c3',"gallery\cam3.mp4")
#     # print(ids_in_roi)
#     # save_list_to_file(list(ids_in_roi),"gallery\ids.pkl")
#     ids_in_roi=load_list_from_file("gallery\ids.pkl")
#     print(ids_in_roi)
#     # time.sleep(3)
#     # run_tracker_with_reid(video_file3,model3,roi_3,s3,cam_id3,"gallery\cam10.mp4",list(ids_in_roi))
#     ids_in_roi=run_tracker_with_reid(video_file2,model2,roi_2,s2,cam_id2,"gallery\cam4.mp4",list(ids_in_roi))
#     # print(ids_in_roi)