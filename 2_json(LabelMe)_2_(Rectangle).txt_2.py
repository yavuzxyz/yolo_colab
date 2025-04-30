import json
import os
from PIL import Image

def convert_labelme_to_yolo(json_filepath, output_dir, label_list):
    # Load the JSON file
    with open(json_filepath, "r") as f:
        data = json.load(f)

    # Prepare an empty list to store YOLO formatted labels
    yolo_labels = []

    # Loop over each labeled object in the image
    for item in data['shapes']:
        label = item['label']

        # If the label is not in our label list, skip this object
        if label not in label_list:
            continue

        # Get the label index
        label_idx = label_list.index(label)

        # Get the bounding box coordinates (in pixels)
        points = item['points']
        x_min = min(point[0] for point in points)
        y_min = min(point[1] for point in points)
        x_max = max(point[0] for point in points)
        y_max = max(point[1] for point in points)

        # Get the center, width, and height of the bounding box
        box_width = x_max - x_min
        box_height = y_max - y_min
        x_center = x_min + box_width / 2
        y_center = y_min + box_height / 2

        # Normalize the bounding box dimensions (relative to the image dimensions)
        img_width = data['imageWidth']
        img_height = data['imageHeight']
        x_center /= img_width
        y_center /= img_height
        box_width /= img_width
        box_height /= img_height

        # Store this label in the YOLO format
        yolo_labels.append([label_idx, x_center, y_center, box_width, box_height])

    # Write the YOLO labels to a new file
    filename = os.path.splitext(os.path.basename(json_filepath))[0]
    output_filepath = os.path.join(output_dir, filename + ".txt")
    with open(output_filepath, "w") as f:
        for label in yolo_labels:
            f.write(" ".join(str(x) for x in label) + "\n")

# Define the paths to the directories
json_dir = "C:\\Users\\yavuz\\OneDrive - uludag.edu.tr\\2-YOLOv8\\convert\\ABC11\\labels"
output_dir = "C:\\Users\\yavuz\\OneDrive - uludag.edu.tr\\2-YOLOv8\\convert\\ABC11\\images"

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Define the label list
label_list = ["a", "b", "Meyve"]

# Convert all JSON files in the directory
for filename in os.listdir(json_dir):
    if filename.endswith(".json"):
        json_filepath = os.path.join(json_dir, filename)
        convert_labelme_to_yolo(json_filepath, output_dir, label_list)
