import torch
import cv2,os,re,sys
import numpy as np
from torchvision import transforms
from PIL import Image
sys.path.append('.')
from utils.REID.alignedreid_utils import *

def extract_ids(file_name):
    # Define the regex pattern to match the file name format
    pattern = re.compile(r'(\d+)_(\w+)_(\d+)')

    # Use the pattern to search for matches in the file name
    match = pattern.search(file_name)

    if match:
        # Extract camera_id, person_id, and frame_num from the matched groups
        camera_id = match.group(1)
        person_id = match.group(2)
        frame_num = match.group(3)

        return camera_id, person_id, frame_num
    else:
        # Return None if the file name doesn't match the expected format
        return None
def get_all_file_paths(directory):
    file_paths = []
    
    # Walk through the directory and its subdirectories
    for foldername, subfolders, filenames in os.walk(directory):
        for filename in filenames:
            # Construct the full file path
            file_path = os.path.join(foldername, filename)
            file_paths.append(file_path)

    return file_paths
def find_matching_id_with_distances(model, query_image_path, gallery_path):
    # query_cam_id, query_person_id, _ = extract_ids(os.path.basename(query_image_path))
    gallery_image_paths=get_all_file_paths(gallery_path)
    gallery_cam_ids, gallery_person_ids, _ = zip(*[extract_ids(os.path.basename(image_path)) 
                                                for image_path in gallery_image_paths])
    
    distances = total_distance_between_images(model, query_image_path, gallery_image_paths)
    # Convert distances to numpy array
    distances_np = np.array(distances.detach().numpy())
    # Find the index of the closest matching ID
    min_distance_index = np.argmin(distances_np)
    # Get the closest matching ID and its distance
    closest_id=gallery_person_ids[min_distance_index]
    min_dist=distances_np[min_distance_index]

    return closest_id,min_dist


def process_video_newids(video_path, roi, tracking_model, id_dict):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # Get video properties
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fps = int(cap.get(5))

    # Define the codec and create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_path = video_path.replace('.mp4', '_processed.mp4')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Run tracker on the current frame
        detections = tracking_model.detect(frame)

        # Process each detection
        for det in detections:
            x, y, w, h, conf, class_id = det
            center_x = x + w // 2
            center_y = y + h // 2

            # Check if the person is in ROI
            if center_x >= roi[0] and center_x <= roi[2] and center_y >= roi[1] and center_y <= roi[3]:
                # Change the old ID to the new ID for the detected frame
                old_id = id_dict.get(class_id, 20)  # Default to 20 if ID not found
                id_dict[class_id] = old_id + 1  # Increment the ID for the next frame

            else:
                # Person is outside ROI, give the ID as 20
                old_id = 20

            # Draw bounding box and ID on the frame
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f'ID: {old_id}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # Write the frame to the output video
        out.write(frame)

    # Release the video capture and writer objects
    cap.release()
    out.release()

    print(f"Processed video saved at: {output_path}")
# gallery_path="temporary_cache\c3"
# gallery_image_paths=get_all_file_paths(gallery_path)
# print(gallery_image_paths)
# gallery_cam_ids, gallery_person_ids, _ = zip(*[extract_ids(os.path.basename(image_path)) 
#                                                 for image_path in gallery_image_paths])
# print(gallery_cam_ids)
# print(gallery_person_ids)