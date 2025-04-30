import cv2
import torch
import numpy as np
import ultralytics
from ultralytics import YOLO

# Load your custom trained model
model = YOLO("C:/Users/yavuz/OneDrive - uludag.edu.tr/2-YOLOv8/Snail_segment/segment.pt")

# Image file to run predictions on
image_path = "C:/Users/yavuz/OneDrive - uludag.edu.tr/2-YOLOv8/Snail_segment/Picture1.jpg"

# Load the image
image = cv2.imread(image_path)

# Run predictions with a confidence threshold of 0.1
results = model.predict(image, conf=0.1)
print(results)

# Define the directory to save the images
save_directory = "C:/Users/yavuz/OneDrive - uludag.edu.tr/2-YOLOv8/Snail_segment"

# Get the predicted masks
masks = results[0].masks

# Create a copy of the original image to apply masks
combined_image = np.zeros_like(image)  # Initially the combined image is all black

# For each predicted result
for result in results:
    masks = result.masks
    boxes = result.boxes
    class_names = result.names

    # Create a copy of the original image to apply masks
    combined_image = np.zeros_like(image)  # Initially the combined image is all black

    # For each predicted mask
    for i, mask in enumerate(masks):
        # Convert the mask to a numpy array if it's not already
        if isinstance(mask, torch.Tensor):
            mask = mask.cpu().detach().numpy()
        elif isinstance(mask, ultralytics.yolo.engine.results.Masks):
            mask = mask.masks.cpu().detach().numpy()
        elif isinstance(mask, list):
            mask = np.array(mask)

        # Resize the mask to match the original image size
        if mask.shape[0] != image.shape[0] or mask.shape[1] != image.shape[1]:
            mask = cv2.resize(mask, (image.shape[1], image.shape[0]))

        # Convert mask to binary (0 and 1)
        mask = (mask > 0).astype(np.uint8)

        # Apply the mask to the combined image
        combined_image[mask == 1] = image[mask == 1]  # keep the original color where mask is applied

        # Find the center of the mask
        M = cv2.moments(mask)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
        else:
            cX, cY = 0, 0

        # Write the class name to the center of the mask
        cv2.putText(combined_image, f'{class_names[0]}', (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

# Save the combined image in the specified directory
cv2.imwrite(f"{save_directory}/mask.png", combined_image)


