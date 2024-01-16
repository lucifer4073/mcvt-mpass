import threading
import cv2
import numpy as np
import sys,time
sys.path.append('.')
from ultralytics import YOLO
from utils.roi_extract import *
from utils.ROI_drawing import *
from utils.properties import check_roi, create_directory
from utils.Image_manager import *
from utils.REID.sort_ids import *
from utils.plotting_funcs import *

from tqdm import trange

def process_video_with_csv(video_path, csv_path, id_dict, output_path):
    # Read CSV file into a Pandas DataFrame
    tracking_df = pd.read_csv(csv_path)

    # Open video capture
    video = cv2.VideoCapture(video_path)

    # Get video properties
    frame_width = int(video.get(3))
    frame_height = int(video.get(4))
    fps = int(video.get(5))
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    # Initialize start times for each person
    start_times = {track_id: None for track_id in tracking_df['track_id'].unique()}

    for _ in trange(total_frames, desc='Processing Video', unit='frames'):
        ret, frame = video.read()

        if not ret:
            break  # Break the loop if no more frames

        for index, row in tracking_df.iterrows():
            track_id = row['track_id']
            if (track_id == 0):
                out.write(frame)
                continue
            x1, y1, x2, y2 = row['x1'], row['y1'], row['x2'], row['y2']

            new_id = id_dict.get(track_id, track_id)
            frame = draw_box_with_id(frame, (x1, y1, x2, y2), new_id)

            # Update time label
            if start_times[track_id] is None:
                start_times[track_id] = time.time()

            elapsed_time = time.time() - start_times[track_id]
            time_label = f"{int(elapsed_time)}s"
            cv2.putText(frame, time_label, (int(x1), int(y1) - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        out.write(frame)

    # Release resources
    video.release()
    out.release()
    print(f"Processed video saved at: {output_path}")
            



def run_tracker_in_thread(filename, model, file_index, roi, save_path, cam_id):
    """
    Runs a video file or webcam stream concurrently with the YOLOv8 model using threading.

    This function captures video frames from a given file or camera source and utilizes the YOLOv8 model for object
    tracking. The function runs in its own thread for concurrent processing.

    Args:
        filename (str): The path to the video file or the identifier for the webcam/external camera source.
        model (obj): The YOLOv8 model object.
        file_index (int): An index to uniquely identify the file being processed, used for display purposes.
        roi (tuple): Region of Interest coordinates (x1, y1, x2, y2).
        save_path (str): Path to save the results.
        cam_id (int): Camera identifier.

    Note:
        Press 'q' to quit the video display window.
    """

    video = cv2.VideoCapture(filename)  # Read the video file
    _, frame = video.read()
    os.makedirs(save_path, exist_ok=True)
    frame_num = 0
    ids_in_roi = set()  # Initialize set to store IDs in ROI

    # Create DataFrame for tracking results
    columns = ['track_id', 'x1', 'y1', 'x2', 'y2', 'frame_num']
    tracking_df = pd.DataFrame(columns=columns)

    img_queue = ImageQueueManager()
    while True:
        ret, frame = video.read()  # Read the video frames

        # Exit the loop if no more frames in either video
        if not ret:
            break

        # Track objects in frames if available
        results = model.track(frame, persist=True, classes=[0], conf=0.2)
        boxes = results[0].boxes.xyxy.cpu()
        track_ids = results[0].boxes.id
        if track_ids is None:
            cv2.imshow(f"Tracking_Stream_{file_index}", frame)
            tracking_df = tracking_df._append(
            {
                'track_id': 0,
                'x1': 1,
                'y1': 1,
                'x2': 1,
                'y2': 1,
                'frame_num': frame_num
            }, ignore_index=True)
            frame_num+=1
            continue

        track_ids = track_ids.int().cpu().tolist()
        conf_values = results[0].boxes.conf.float().cpu().tolist()
        fnum = 0
        for idx in range(len(track_ids)):
            x1, y1, x2, y2 = boxes.numpy()[idx]
            center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
            if check_roi(center_x, center_y, roi):
                ids_in_roi.add(track_ids[idx])
            # Append tracking results to DataFrame
            tracking_df = tracking_df._append({
                'track_id': track_ids[idx],
                'x1': x1,
                'y1': y1,
                'x2': x2,
                'y2': y2,
                'frame_num': frame_num
            }, ignore_index=True)

            confidence = conf_values[idx]
            cropped_image = frame[int(y1):int(y2), int(x1):int(x2)]
            image_pair = (cropped_image, confidence)

            img_queue.add_image(track_ids[idx], image_pair)
            fnum += 1

        frame = plot_bbox(frame, roi)
        res_plotted = plot_bboxes_with_ids(frame, boxes, track_ids)
        cv2.imshow(f"Tracking_Stream_{file_index}", res_plotted)

        save_images(img_queue, save_path, cam_id)

        # Increment frame number
        frame_num += 1

        # Break the loop if 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Save DataFrame to CSV file
    csv_file_path = os.path.join(save_path, f"tracking_results_{cam_id}.csv")
    ids_path=os.path.join(save_path,f"tracked_ids.pkl")
    tracking_df.to_csv(csv_file_path, index=False)
    save_list_to_file(list(ids_in_roi),ids_path)
    # Release video capture object and destroy OpenCV windows
    video.release()
    cv2.destroyAllWindows()


if __name__=="__main__":
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
    video_file2 = "old_version\sample_data\merged_videos\c4_merged.mp4"
    video_file3 = "old_version\sample_data\merged_videos\c10_merged.mp4"

    # Select ROI
    roi_1 = select_roi_from_video(video_file1)
    roi_2 = select_roi_from_video(video_file2)
    roi_3 = select_roi_from_video(video_file3)

    #Check_ids in ROI

    ids_1=set()
    ids_2=set()
    ids_3=set()



    # # Create the tracker threads
    tracker_thread1 = threading.Thread(target=run_tracker_in_thread, args=(video_file1, model1, 1, roi_1, s1, cam_id1), daemon=True)
    # tracker_thread2 = threading.Thread(target=run_tracker_in_thread, args=(video_file2, model2, 2, roi_2, s2, cam_id2), daemon=True)
    tracker_thread3 = threading.Thread(target=run_tracker_in_thread, args=(video_file3, model3, 3, roi_3, s3, cam_id3), daemon=True)

    #tracker threads for reid

    # tracker_thread1 = threading.Thread(target=process_video_newids, args=(video_file1, roi_1, model1, id_dict_1, "gallery\cam3.mp4"), daemon=True)
    # tracker_thread2 = threading.Thread(target=process_video_newids, args=(video_file2, roi_2, model2, id_dict_2, "gallery\cam4.mp4"), daemon=True)
    # tracker_thread3 = threading.Thread(target=process_video_newids, args=(video_file3, roi_3, model3, id_dict_3, "gallery\cam10.mp4"), daemon=True)

    # # Start the tracker threads
    tracker_thread1.start()
    # tracker_thread2.start()
    tracker_thread3.start()

    # # Wait for the tracker threads to finish
    tracker_thread1.join()
    # tracker_thread2.join()
    tracker_thread3.join()

    # # Clean up and close windows
    cv2.destroyAllWindows()
    #,True,s1,reid_model

    # s1 = "gallery\\c10_2"

    # # Load the model
    # model1 = YOLO('models\\yolov8n (1).pt')

    # # Camera ID
    # cam_id1 = 'c10'

    # # Define the video file for the tracker
    # video_file1 = "old_version\\sample_data\\c10_tallman.mp4"  # Path to video file, 0 for webcam

    # # Select ROI
    # roi_1 = select_roi_from_video(video_file1)

    # # Run the tracker for a single camera
    # run_tracker_in_thread(video_file1, model1, 1, roi_1, s1, cam_id1)

    # # Clean up and close windows
    # cv2.destroyAllWindows()



