from ultralytics import YOLO
import cv2
import numpy as np
import torch


# Load the YOLOv8 model
model = YOLO("best.pt")

# Open the video file
video_path = "c5.mp4"
cap = cv2.VideoCapture(video_path)

# Define the codec and create a VideoWriter object to save the video
fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
out = cv2.VideoWriter('output13.mp4', fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

# Create a named window where the size can be changed
cv2.namedWindow("YOLOv8 Inference", cv2.WINDOW_NORMAL)

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLOv8 inference on the frame with a confidence threshold of 0.55
        results = model.predict(frame, conf=0.55)
        
        masks = results[0].masks

        if masks is not None:
            # For each predicted mask
            for i, mask in enumerate(masks):
                # Convert the mask to a numpy array if it's not already
                if isinstance(mask, torch.Tensor):
                    mask = mask.cpu().detach().numpy()
                elif hasattr(mask, 'masks'):
                    mask_data = mask.masks
                    if isinstance(mask_data, torch.Tensor):
                        mask = mask_data.cpu().detach().numpy()
                    else:
                        mask = np.array(mask_data)
                elif isinstance(mask, list):
                    mask = np.array(mask)
                else:
                    # Unknown type, skip this iteration
                    continue

                # Ensure mask is binary
                mask = (mask > 0.5).astype(np.uint8)
                
                # Convert the binary mask into a 3 channel mask to apply on the frame
                resized_mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
                mask_3channel = np.repeat(resized_mask[:, :, np.newaxis], 3, axis=2)


                # Use the mask to retain the segmented part and turn everything else to black
                segmented_frame = frame * mask_3channel

                # Display the segmented frame
                cv2.imshow("YOLOv8 Inference", segmented_frame)

                # Write the segmented frame into the output file
                out.write(segmented_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture and writer objects and close the display window
cap.release()
out.release()
cv2.destroyAllWindows()
