from ultralytics import YOLO
import cv2

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

        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        # Write the frame into the output file
        out.write(annotated_frame)

        # Display the annotated frame
        cv2.imshow("YOLOv8 Inference", annotated_frame)

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
