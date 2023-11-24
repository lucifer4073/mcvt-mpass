from ultralytics import YOLO
import cv2,os
import numpy as np  
from data_load import get_ids_images
from PIL import Image
import matplotlib.pyplot as plt
import torch
from Model import Runner

# load yolov8 model
def save_images_with_format(image_array, track_ids, output_directory):
    """
    Save images with a specified format in the provided directory.

    Parameters:
    - image_array: numpy.ndarray
        Array of images to be saved.
    - track_ids: numpy.ndarray
        Array of tracking IDs corresponding to each image.
    - output_directory: str
        Directory where the images will be saved.
    """
    # Create the output directory if it doesn't exist
    os.makedirs(output_directory, exist_ok=True)

    # Iterate over each image and track ID
    for index, (image, track_id) in enumerate(zip(image_array, track_ids)):
        # Convert the NumPy array to PIL Image
        pil_image = Image.fromarray(np.uint8(image))

        # Save the image with the specified format
        file_name = f"{track_id}_{index}.jpg"
        file_path = os.path.join(output_directory, file_name)
        pil_image.save(file_path)

        print(f"Image saved: {file_path}")
#All paths
model_path="models\yolov8n (1).pt"
video_path="c4_v3_s1_5fps.mp4"
query_folder="generated_data\c4_v3"
gallery_folder="generated_data\c3_v1"

qarray,qids,_,_=get_ids_images(query_folder)
garray,gids,_,_=get_ids_images(gallery_folder)

model=Runner()
# qfeats=model._features(qarray)
# gfeats=model._features(garray)
#save generated query features and gallery features
# np.save(os.path.join("generated_data","query_features.npy"),qfeats)
# np.save(os.path.join("generated_data","gallery_features.npy"),gfeats)
#load saved query and gallery features


# dmat=model.compute_distance(qfeats,gfeats)
# 
# torch.save(dmat, "generated_data/distance_mat.pth")

dmat=torch.load("generated_data\distance_mat.pth")
indices=np.argmin(dmat,axis=1)
new_ids=gids[indices]

save_images_with_format(qarray,new_ids,"generated_data\c4_v3_reid")


# print("printing min distances")
# print(np.min(dmat,axis=1))
# print("-----------")
# print(gids[indices]==qids)
# print('_______')
# print(dmat)
# fig, ax = plt.subplots()
# heatmap=ax.imshow(dmat,cmap='viridis')

# fig.savefig("heatmap_1.jpg")


