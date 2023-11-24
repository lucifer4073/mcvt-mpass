import re

def extract_ids(file_name):
    # Define the regex pattern to match the file name format
    pattern = re.compile(r'(\d+)_(\d+)_(\d+)\.')

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

# if __name__ == "__main__":
#     # Example usage:
#     file_name = "123_456_789.jpg"
#     result = extract_ids(file_name)

#     if result is not None:
#         camera_id, person_id, frame_num = result
#         print("Camera ID:", camera_id)
#         print("Person ID:", person_id)
#         print("Frame Number:", frame_num)
#     else:
#         print("Invalid file name format.")
