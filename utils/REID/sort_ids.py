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


# gallery_path="temporary_cache\c3"
# gallery_image_paths=get_all_file_paths(gallery_path)
# print(gallery_image_paths)
# gallery_cam_ids, gallery_person_ids, _ = zip(*[extract_ids(os.path.basename(image_path)) 
#                                                 for image_path in gallery_image_paths])
# print(gallery_cam_ids)
# print(gallery_person_ids)