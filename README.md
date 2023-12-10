# Multi Camera Visitor Tracking Sytem

## Objective

We aim to develop a system for measuring visitor wait times in specific scenarios. This system will help assess engagement levels, identify inefficiencies, and monitor interactions between customers and staff.

## Description

- Creating a visual representation of a designated area in front of the counter or showcase, referred to as a Region of Interest (ROI), and tracking the duration of people's presence within this area. The aim is to enhance and refine this process for optimal efficiency.


#### Test Case Description

- Input: Video (mp4) + ROI coordinates.
- ROI is in the format (xmin,ymin,xmax,ymax)
- Output: Video (mp4)

## Requirements

To install the required Python packages for this project, create a virtual environment (optional but recommended) and run the following command:

```shell
pip install -r requirements.txt
```
### Dataset

Given below is the trimmed link for the two datasets

Video-1: 

[link] <https://iitgnacin-my.sharepoint.com/:v:/g/personal/21110040_iitgn_ac_in/ETrg69Ow0Z5GmAqi5eGszYoBnihlQo4Io8aSXW3RQSskuA?nav=eyJyZWZlcnJhbEluZm8iOnsicmVmZXJyYWxBcHAiOiJPbmVEcml2ZUZvckJ1c2luZXNzIiwicmVmZXJyYWxBcHBQbGF0Zm9ybSI6IldlYiIsInJlZmVycmFsTW9kZSI6InZpZXciLCJyZWZlcnJhbFZpZXciOiJNeUZpbGVzTGlua0NvcHkifX0&e=QdHfqs>

Video-2:

[link] <https://iitgnacin-my.sharepoint.com/:v:/g/personal/21110040_iitgn_ac_in/EYH25J5NqQ5IgioUL-c5_40B8g5o8cj9v2cVifhDCJtvYQ?nav=eyJyZWZlcnJhbEluZm8iOnsicmVmZXJyYWxBcHAiOiJPbmVEcml2ZUZvckJ1c2luZXNzIiwicmVmZXJyYWxBcHBQbGF0Zm9ybSI6IldlYiIsInJlZmVycmFsTW9kZSI6InZpZXciLCJyZWZlcnJhbFZpZXciOiJNeUZpbGVzTGlua0NvcHkifX0&e=mmp0iI>

## Project Structure

### Main script

`main.py` : Takes a 5 fps video input, runs tracker and detection and creates a gallery dataset for each camera id

### Utility modules

### 1. `compute_distance_matrix.py`

This file likely contains code related to computing distance matrices. It might be used for calculating distances between points or entities in a dataset.

### 2. `Dataset.py`

`Dataset.py`  a module that handles the management and processing of datasets. It contain classes and functions for loading, preprocessing, and manipulating datasets.

### 3. `Detector.py`

This file contains code related to object detection. It includes functions and classes for detecting objects within images or videos.

### 4. `detect_time.py`

`detect_time.py` involves time-related functionalities in the context of detection. It may be used for tracking the time of detection events or durations.

### 5. `fps_conversion.py`

This file is responsible for converting frames per second (fps) values. It contains functions for converting between different fps representations.

### 6. `Image_manager.py`

`Image_manager.py` is involved in managing images. It may include functions and classes for loading images in queue data structure.

### 7. `make_video.py`

This file related to video creation. It contain functions for assembling a series of images or frames into a video file.

### 8. `New_tracker.py`

Contains implementation of deepsort tracker

### 9. `plotting_funcs.py`

Used in plotting the results

### 10. `properties.py`

This file contain configurations or properties used throughout the utility. It could include constants, settings, or parameters used by various modules.

### 11. `REID Utils`

This folder contains utility functions for the person Re-Identification (REID) module.

### 12. `ROI_drawing.py`

`ROI_drawing.py` contains code for drawing Regions of Interest (ROIs).

### 12. `Threads.py`

Consist of application of multithreading process in a multi camera system.

# Models Used in the Process

## YOLOv8
- **Description:** YOLOv8 (You Only Look Once, version 8) is an object detection model known for its real-time object detection capabilities.
- **Link to Model:** [YOLOv8 Model](https://iitgnacin-my.sharepoint.com/:u:/g/personal/21110040_iitgn_ac_in/Ee2coWMsUT9IhfiQ9f__VW8BCL59N_vkMO94oM-NB-dhfg?e=iVpQ5X)

## AlignedReID ResNet-50
- **Description:** AlignedReID ResNet-50 is a model specifically designed for person re-identification tasks, providing robust features for matching individuals across different camera views.
- **Link to Weights:** [AlignedReID ResNet-50 Weights](https://iitgnacin-my.sharepoint.com/:u:/g/personal/21110040_iitgn_ac_in/EYos21S06O5OgEUzgitYKFMBq8pYnf476k85sZBFZsjz7A?e=QbHECs)

---


## Usage

### Command Line Interface (CLI)

To run the Multi Camera Visitor Tracking System via the command line interface (CLI), follow these steps:

1. **Clone the Repository:** Clone the repo to local machine

``` shell
git clone https://github.com/lucifer4073/mcvt-mpass.git
```
2. **Run system**: Run the Multi Camera Visitor Tracking System using the following command:

```shell
python main.py
```
## USING state-of-the-art (SOTA) models

1. **Object Detection**: Consider adopting models like EfficientDet or Detectron2's RetinaNet. These models offer advanced architecture designs and have shown superior accuracy and efficiency compared to YOLO in recent benchmarks.

2. **Tracking**: Instead of relying solely on traditional tracking algorithms, incorporate deep learning-based trackers such as FairMOT or Tracktor, which leverage the temporal information for better tracking, especially in crowded scenes.

3. **Feature Fusion**: Explore models that integrate both object detection and tracking in an end-to-end manner. This can potentially improve the system's robustness by allowing better incorporation of temporal information during detection and tracking stages.
