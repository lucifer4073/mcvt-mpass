import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
def plot_heatmap(matrix):
    plt.imshow(matrix, cmap='viridis', interpolation='nearest')
    plt.colorbar()
    plt.show()


def display_frame(frame, window_title='Frame'):
    # Convert the frame to RGB (OpenCV uses BGR by default)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Display the frame
    cv2.imshow(window_title, frame_rgb)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def plot_bbox(frame,bbox):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    x_min, y_min, x_max, y_max = bbox
    # Draw the bounding box
    cv2.rectangle(frame_rgb, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (255, 0, 0), 2)
    
    return frame_rgb

def draw_box_with_id(frame, box, track_id):
    x1, y1, x2, y2 = map(int, box)

    # Draw the box
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Display the track ID
    cv2.putText(frame, f'Track ID: {track_id}', (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return frame
def plot_bboxes_with_ids(frame, boxes, box_ids,display=False,wtitle="Frame"):
    # Convert the frame to RGB (OpenCV uses BGR by default)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Loop through each bounding box and ID
    for bbox, bbox_id in zip(boxes, box_ids):
        # Extract coordinates
        x_min, y_min, x_max, y_max = bbox

        # Draw the bounding box
        cv2.rectangle(frame_rgb, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (255, 0, 0), 2)

        # Display the ID
        cv2.putText(frame_rgb, f'ID: {bbox_id}', (int(x_min), int(y_min) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2, cv2.LINE_AA)

    # Use the display_frame function to display the frame
    if display:
        display_frame(frame_rgb,wtitle)
    
    return frame_rgb
    


if __name__ == "__main__":
    # Example usage
    frame = np.random.randint(0, 256, size=(200, 300, 3), dtype=np.uint8)  # Example frame (replace with your actual frame)
    bbox = (50, 30, 200, 150)  # Example bounding box in (x_min, y_min, x_max, y_max) format
    bbox_id = 1  # Example ID
    
    # Plot the bounding box on the frame with ID
    plot_bboxes_with_ids(frame, bbox, bbox_id,display=True)