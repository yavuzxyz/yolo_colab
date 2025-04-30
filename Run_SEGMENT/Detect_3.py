from ultralytics import YOLO
import cv2
from collections import Counter

# Load your custom trained model
model = YOLO("best.pt")
   
# Image file to run predictions on
image_path = "c2.jpeg"
# Run predictions with a confidence threshold of 0.90
results_list = model.predict(image_path, conf=0.65)

# Store all detected classesq
detected_classes = []

# Parse the results
for res in results_list:
    for box in res.boxes.data:
        # Each box is a detection with the following format: [x1, y1, x2, y2, confidence, class]
        x_center, y_center, width, height, conf, class_id = box
        x1 = x_center - (width / 2)
        y1 = y_center - (height / 2)
        x2 = x_center + (width / 2)
        y2 = y_center + (height / 2)
        class_name = model.names[int(class_id)]  # Get the class name
        print(f"Detected {class_name} with confidence {conf:.2f}")
        detected_classes.append(class_name)

# Visualize the results
res_plotted = results_list[0].plot()
# Create a named window where the size can be changed
cv2.namedWindow("Detection", cv2.WINDOW_NORMAL)
# Display the image with predictions
cv2.imshow("Detection", res_plotted)

import os
# Current working directory
working_directory = os.getcwd()
# Define the save path within the working directory
save_path = os.path.join(working_directory, "Detection.jpg")
# Save the image with predictions
cv2.imwrite(save_path, res_plotted)


cv2.waitKey(0)  # waits until a key is pressed
cv2.destroyAllWindows()  # destroys the window showing image