"""PREDICT"""

from ultralytics import YOLO
import cv2

from ultralytics import YOLO
import cv2

# Load your custom trained model
model = YOLO("C:/Users/yavuz/OneDrive - uludag.edu.tr/2-YOLOv8/Snail/best.pt")
# Image file to run predictions on
image_path = "C:/Users/yavuz/OneDrive - uludag.edu.tr/2-YOLOv8/Snail/images/Picture1.jpg"
# Run predictions with a confidence threshold of 0.65
results = model.predict(image_path, conf=0.65)

# Visualize the results
res_plotted = results[0].plot()
# Create a named window where the size can be changed
cv2.namedWindow("Predictions", cv2.WINDOW_NORMAL)
# Display the image with predictions
cv2.imshow("Predictions", res_plotted)
# Save the image with predictions
# cv2.imwrite("C:/Users/yavuz/OneDrive - uludag.edu.tr/2-YOLOv8/output", res_plotted)
cv2.waitKey(0)  # waits until a key is pressed
cv2.destroyAllWindows()  # destroys the window showing image

"""LIST LABELS"""

import yaml

with open("C:/Users/yavuz/OneDrive - uludag.edu.tr/2-YOLOv8/Snail/dataset.yaml", 'r') as stream:
    data_loaded = yaml.safe_load(stream)

print(data_loaded['names'])
