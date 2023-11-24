import heapq

class PriorityQueue:
    def __init__(self):
        self.queue = []

    def push(self, pair):
        image_array, confidence = pair
        heapq.heappush(self.queue, (-confidence, image_array))

        # If the size exceeds 10, pop the pair with the least confidence
        if len(self.queue) > 10:
            heapq.heappop(self.queue)

    def get_top_pairs(self):
        # Return the pairs with the highest confidence
        return [(-confidence, image_array) for confidence, image_array in self.queue]

class ImageQueueManager:
    def __init__(self):
        self.queue_dict = {}

    def add_image(self, tracking_id, image_pair):
        if tracking_id not in self.queue_dict:
            self.queue_dict[tracking_id] = PriorityQueue()

        self.queue_dict[tracking_id].push(image_pair)

    def get_priority_queue(self, tracking_id):
        return self.queue_dict.get(tracking_id, None)

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
