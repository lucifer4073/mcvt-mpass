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

def process_video_and_save(video_path, save_path, cam_id, reidentify=False, gallery=None, reid_model=''):
    try:
        model = YOLO('models\yolov8n (1).pt')
        cap = cv2.VideoCapture(video_path)
        create_directory(save_path)
        ret = True
        os.makedirs(save_path, exist_ok=True)
        roi = np.array(select_roi_from_video(video_path))
        frame_num = 0
        img_queue = ImageQueueManager()

        while ret:
            ret, frame = cap.read()
            if ret:
                # Perform object tracking
                results = model.track(frame, classes=[0], conf=0.1, persist=True)
                boxes = results[0].boxes.xyxy.cpu()

                try:
                    track_ids = results[0].boxes.id.int().cpu().tolist()
                    conf_values = results[0].boxes.conf.float().cpu().tolist()
                except AttributeError:
                    print("No object detected, skipping frame")
                    continue

                for idx in range(len(track_ids)):
                    x1, y1, x2, y2 = boxes.numpy()[idx]
                    center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2

                    if not check_roi(center_x, center_y, roi):
                        continue

                    confidence = conf_values[idx]
                    cropped_image = frame[int(y1):int(y2), int(x1):int(x2)]
                    image_pair = (cropped_image, confidence)

                    if reidentify:
                        query_image_path = os.path.join("temporary_cache", "temp_img")
                        save_image(cropped_image, query_image_path)
                        closest_id, _ = find_matching_id_with_distances(reid_model, query_image_path, gallery)
                        track_ids[idx] = closest_id
                        delete_image(query_image_path)

                    img_queue.add_image(track_ids[idx], image_pair)

                # Display annotated frame
                plot_bboxes_with_ids(frame, boxes.numpy(), track_ids, display=True)
                # if cv2.waitKey(1) & 0xFF == ord("q"):
                #     break

                frame_num += 1

        save_images(img_queue, save_path, cid=cam_id)
        print(f"Tracking Completed for Camera {cam_id}")

    except Exception as e:
        print(f"An error occurred: {e}")

    finally:
        cap.release()
        cv2.destroyAllWindows()

def run_tracker_in_thread(video_path, save_path, cam_id, reidentify=False, gallery=[], reid_model=''):
    process_video_and_save(video_path, save_path, cam_id, reidentify, gallery, reid_model)

def run_multithreaded_tracking(video_paths, save_paths, cam_ids, reidentify, gallery, reid_model=''):
    threads = []
    for i, video_path in enumerate(video_paths):
        thread = threading.Thread(target=run_tracker_in_thread, args=(video_path, save_paths[i], cam_ids[i], reidentify[i], gallery[i], reid_model), daemon=True)
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
