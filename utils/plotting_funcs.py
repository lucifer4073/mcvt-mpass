import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
def plot_heatmap(matrix):
    plt.imshow(matrix, cmap='viridis', interpolation='nearest')
    plt.colorbar()
    plt.show()

def plot_bbox_on_frame(frame, bbox, bbox_id):
    # Extract coordinates
    x_min, y_min, x_max, y_max = bbox

    # Convert the frame to RGB (OpenCV uses BGR by default)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Draw the bounding box
    cv2.rectangle(frame_rgb, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (255, 0, 0), 2)

    # Display the ID
    cv2.putText(frame_rgb, f'ID: {bbox_id}', (int(x_min), int(y_min) - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2, cv2.LINE_AA)

    # Display the frame
    cv2.imshow('Frame with Bounding Box', frame_rgb)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# if __name__ == "__main__":
#     # Example usage
#     m = 5  # Number of rows
#     n = 8  # Number of columns

#     # Create a random m x n matrix for demonstration
#     matrix = np.random.rand(m, n)

#     # Plot the heatmap
#     plot_heatmap(matrix)

if __name__ == "__main__":
    # Example usage
    frame = np.random.randint(0, 256, size=(200, 300, 3), dtype=np.uint8)  # Example frame (replace with your actual frame)
    bbox = (50, 30, 200, 150)  # Example bounding box in (x_min, y_min, x_max, y_max) format
    bbox_id = 1  # Example ID
    
    # Plot the bounding box on the frame with ID
    plot_bbox_on_frame(frame, bbox, bbox_id)