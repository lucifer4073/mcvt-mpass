import heapq
import os,cv2
import pandas as pd
import pickle
class PriorityQueue:
    def __init__(self):
        self.queue = []

    def push(self, pair):
        image_array, confidence= pair
        heapq.heappush(self.queue, (-confidence, image_array))

        # If the size exceeds 10, pop the pair with the least confidence
        if len(self.queue) > 2:
            heapq.heappop(self.queue)

    def get_top_pairs(self):
        # Return the pairs with the highest confidence
        return [(-confidence, image_array) for confidence, image_array in self.queue]

class ImageQueueManager:
    def __init__(self):
        self.queue_dict = {}

    def add_image(self, tracking_id, image_tri):
        if tracking_id not in self.queue_dict:
            self.queue_dict[tracking_id] = PriorityQueue()

        self.queue_dict[tracking_id].push(image_tri)

    def get_priority_queue(self, tracking_id):
        return self.queue_dict.get(tracking_id, None)
    
def save_images(image_queue_manager, save_path, cam_id):

    for tracking_id, priority_queue in image_queue_manager.queue_dict.items():
        if priority_queue:
            # tracking_id_dir = os.path.join(save_path, f"{tracking_id}")
            # os.makedirs(tracking_id_dir, exist_ok=True)

            for index, (confidence, image_array) in enumerate(priority_queue.get_top_pairs()):
                image_filename = f"{tracking_id}_{cam_id}_{index}.jpg"

                image_path = os.path.join(save_path, image_filename)
                save_image(image_array, image_path)

def save_image(image_array, image_path):
    cv2.imwrite(image_path, image_array)

def get_all_files_in_subfolders(parent_path):
    file_paths = []

    # Walk through all subdirectories and get file paths
    for dirpath, _, filenames in os.walk(parent_path):
        for filename in filenames:
            file_path = os.path.join(dirpath, filename)
            file_paths.append(file_path)

    return file_paths

def get_all_file_paths(parent_path_list):
    file_paths=[]
    for parent_path in parent_path_list:
        file_paths.extend(get_all_files_in_subfolders(parent_path))
    return file_paths


def delete_image(image_path):
    """
    Delete an image at the specified path.

    Args:
    - image_path: Path of the image to be deleted.
    """
    try:
        os.remove(image_path)
        print(f"Image at '{image_path}' deleted successfully.")
    except FileNotFoundError:
        print(f"Image at '{image_path}' not found.")
    except Exception as e:
        print(f"Error deleting image: {e}")
def save_list_to_file(lst, filename):
    with open(filename, 'wb') as file:
        pickle.dump(lst, file)

def load_list_from_file(filename):
    with open(filename, 'rb') as file:
        loaded_list = pickle.load(file)
    return loaded_list
if __name__ == "__main__":
    # Example usage:
    queue_manager = ImageQueueManager()

    # Add images to different queues
    queue_manager.add_image("tracking_id_1", ([1, 2, 3], 0.9))
    queue_manager.add_image("tracking_id_2", ([4, 5, 6], 0.8))
    queue_manager.add_image("tracking_id_1", ([7, 8, 9], 0.95))
    queue_manager.add_image("tracking_id_3", ([10, 11, 12], 0.85))

    # Get priority queues for specific tracking IDs
    queue_1 = queue_manager.get_priority_queue("tracking_id_1")
    queue_2 = queue_manager.get_priority_queue("tracking_id_2")
    queue_3 = queue_manager.get_priority_queue("tracking_id_3")

    # Display the top pairs for each tracking ID
    if queue_1:
        print("Top pairs for tracking_id_1:")
        for pair in queue_1.get_top_pairs():
            print(pair)

    if queue_2:
        print("Top pairs for tracking_id_2:")
        for pair in queue_2.get_top_pairs():
            print(pair)

    if queue_3:
        print("Top pairs for tracking_id_3:")
        for pair in queue_3.get_top_pairs():
            print(pair)
