from ultralytics import YOLO
import cv2

# Load your custom trained model
model = YOLO("best.pt")
# Image file to run predictions on
image_path = "c2.jpeg"
# Run predictions with a confidence threshold of 0.55
results = model.predict(image_path, conf=0.50)

# Visualize the results
res_plotted = results[0].plot()

# Create a named window where the size can be changed
cv2.namedWindow("Predictions", cv2.WINDOW_NORMAL)

# Display the image with predictions
cv2.imshow("Predictions", res_plotted)

# Save the image with predictions
cv2.imwrite("output12.jpg", res_plotted)

cv2.waitKey(0)  # waits until a key is pressed
cv2.destroyAllWindows()  # destroys the window showing image