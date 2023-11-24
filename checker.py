from utils.New_tracker import DetectionTracking
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
model_path="models\yolov8n (1).pt"
obj=DetectionTracking(model_path)
img=np.array(Image.open("RLCAFTCONF-C2_107274.jpeg"))

res=obj.process_frame(img)
print(res)
new_frame=obj.annotate_frame(img,res)
plt.imshow(new_frame)
plt.show()