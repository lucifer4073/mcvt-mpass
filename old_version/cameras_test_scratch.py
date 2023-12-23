import cv2

def get_first_frame(video_path):
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Check if the video file was successfully opened
    if not cap.isOpened():
        print(f"Error: Unable to open video source {video_path}")
        return None

    # Read the first frame from the video
    ret, frame = cap.read()

    # Check if the frame was successfully read
    if not ret:
        print(f"Error: Unable to read video frame from {video_path}")
        cap.release()
        return None

    # Display the first frame
    cv2.imshow("First Frame", frame)
    cv2.waitKey(0)  # Wait until a key is pressed
    cv2.destroyAllWindows()

    # Release the video source
    cap.release()

    return frame

# Example usage
video_path = "c3_v1_s1_5fps.mp4"
first_frame = get_first_frame(video_path)
