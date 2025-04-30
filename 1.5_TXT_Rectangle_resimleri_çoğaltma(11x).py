import os
import cv2
import numpy as np
import math

images_folder = r"C:\Users\yavuz\OneDrive - uludag.edu.tr\1-YOLOv8\images_detect"
labels_folder = r"C:\Users\yavuz\OneDrive - uludag.edu.tr\1-YOLOv8\labels_detect"
output_folder = r"C:\Users\yavuz\OneDrive - uludag.edu.tr\1-YOLOv8\output"

# Check whether images_folder is a directory or a file
if os.path.isdir(images_folder):
    image_files = os.listdir(images_folder)
elif os.path.isfile(images_folder):
    image_files = [os.path.basename(images_folder)]
    images_folder = os.path.dirname(images_folder)
else:
    print(f"Invalid images_folder: {images_folder}")
    exit()

# Check whether labels_folder is a directory or a file
if os.path.isdir(labels_folder):
    label_files = os.listdir(labels_folder)
elif os.path.isfile(labels_folder):
    label_files = [os.path.basename(labels_folder)]
    labels_folder = os.path.dirname(labels_folder)
else:
    print(f"Invalid labels_folder: {labels_folder}")
    exit()
  
# Define rotation angles for image augmentation
rotation_angles = [90, 180, 270]

# ---------- Kod 1 Başlangıç ----------
# For each image file, perform augmentations
for image_file in image_files:
    # Read the image
    img = cv2.imread(os.path.join(images_folder, image_file))
    if img is None:
        print(f"Uyarı: {image_file} okunamadı.")
        continue
    rows, cols, _ = img.shape

    # Read the label file
    with open(os.path.join(labels_folder, image_file.replace('.jpg', '.txt')), 'r') as f:
        labels = f.readlines()

    # Rotate the image and label by specified angles
    for angle in rotation_angles:
        # Rotate the image
        M = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)
        img_rotated = cv2.warpAffine(img, M, (cols, rows))
        
        # Save rotated image with a new name
        new_img_name = f"rotated{angle}_" + image_file
        cv2.imwrite(os.path.join(output_folder, new_img_name), img_rotated)
        
        # Rotate label coordinates
        labels_rotated = []
        for i in range(len(labels)):
            class_id, x_center, y_center, width, height = map(float, labels[i].strip().split())
            rot_center_x = cols / 2.0  # half of image width
            rot_center_y = rows / 2.0  # half of image height
            M_rot = cv2.getRotationMatrix2D((rot_center_x, rot_center_y), angle, 1)
            # Rotate center point (convert normalized to pixel, then back later)
            x_center_rot = M_rot[0, 0] * x_center * cols + M_rot[0, 1] * y_center * rows + M_rot[0, 2]
            y_center_rot = M_rot[1, 0] * x_center * cols + M_rot[1, 1] * y_center * rows + M_rot[1, 2]
            # Rotate bounding box corners to recalc width and height
            box_points = np.array([
                [x_center - width / 2, y_center - height / 2],
                [x_center + width / 2, y_center - height / 2],
                [x_center - width / 2, y_center + height / 2],
                [x_center + width / 2, y_center + height / 2]
            ]) * np.array([cols, rows])
            rotated_box_points = cv2.transform(box_points.reshape(-1, 1, 2), M_rot)
            width_rot = np.max(rotated_box_points[..., 0]) - np.min(rotated_box_points[..., 0])
            height_rot = np.max(rotated_box_points[..., 1]) - np.min(rotated_box_points[..., 1])
            # Normalize the rotated center and box dimensions
            x_center_rot /= cols
            y_center_rot /= rows
            width_rot /= cols
            height_rot /= rows
            labels_rotated.append(f"{int(class_id)} {x_center_rot} {y_center_rot} {width_rot} {height_rot}\n")

        # Save rotated labels
        with open(os.path.join(output_folder, new_img_name.replace('.jpg', '.txt')), 'w') as f:
            f.writelines(labels_rotated)

    # Brighten the image
    img_light = cv2.convertScaleAbs(img, alpha=1, beta=60)
    new_img_name_light = "light_" + image_file
    cv2.imwrite(os.path.join(output_folder, new_img_name_light), img_light)

    # Add Gaussian noise
    mean = 0
    var = 190  # increase intensity
    sigma = var**0.5
    gauss = np.random.normal(mean, sigma, img.shape)
    img_noisy = np.clip(img + gauss, 0, 255).astype(np.uint8)
    new_img_name_noisy = "noisy_" + image_file
    cv2.imwrite(os.path.join(output_folder, new_img_name_noisy), img_noisy)
    
    # Create shadow effect
    shadow = np.ones(img.shape, dtype="uint8") * 255
    cv2.rectangle(shadow, (int(cols/4), int(rows/4)), (int(3*cols/4), int(3*rows/4)), (170, 170, 170), -1)
    img_shadowed = cv2.addWeighted(img, 0.6, shadow, 0.5, 0)
    new_img_name_shadowed = "shadowed_" + image_file
    cv2.imwrite(os.path.join(output_folder, new_img_name_shadowed), img_shadowed)

    # --- Grayscale Conversion ---
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Convert grayscale image back to 3 channels for saving as JPG
    img_gray_3ch = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
    new_img_name_gray = "gray_" + image_file
    cv2.imwrite(os.path.join(output_folder, new_img_name_gray), img_gray_3ch)
    
    # Copy label files for light, noisy, shadowed, and gray images
    with open(os.path.join(labels_folder, image_file.replace('.jpg', '.txt')), 'r') as f_in:
        lines = f_in.readlines()
        with open(os.path.join(output_folder, new_img_name_light.replace('.jpg', '.txt')), 'w') as f_out_light:
            f_out_light.writelines(lines)
        with open(os.path.join(output_folder, new_img_name_noisy.replace('.jpg', '.txt')), 'w') as f_out_noisy:
            f_out_noisy.writelines(lines)
        with open(os.path.join(output_folder, new_img_name_shadowed.replace('.jpg', '.txt')), 'w') as f_out_shadowed:
            f_out_shadowed.writelines(lines)
        with open(os.path.join(output_folder, new_img_name_gray.replace('.jpg', '.txt')), 'w') as f_out_gray:
            f_out_gray.writelines(lines)

# ---------- Kod 1 Bitiş ----------

# ---------- Kod 2 Başlangıç ----------
# Cropping, flipping, zoomed out operations
for image_file in image_files:
    img = cv2.imread(os.path.join(images_folder, image_file))
    rows, cols, _ = img.shape
    aspect_ratio = cols / rows

    with open(os.path.join(labels_folder, image_file.replace('.jpg', '.txt')), 'r') as f:
        labels = f.readlines()

    # Calculate bounding boxes from labels
    bboxes = []
    for label in labels:
        numbers = list(map(float, label.strip().split()))
        if len(numbers) == 5:  # Bounding box
            class_id, x_center, y_center, width, height = numbers
            x1 = (x_center - width / 2) * cols
            y1 = (y_center - height / 2) * rows
            x2 = (x_center + width / 2) * cols
            y2 = (y_center + height / 2) * rows
            bboxes.append((x1, y1, x2, y2))
        elif len(numbers) > 5:  # Polygon
            class_id = numbers[0]
            coordinates = numbers[1:]
            for i in range(0, len(coordinates), 2):
                x_center, y_center = coordinates[i], coordinates[i + 1]
                x1 = x_center * cols
                y1 = y_center * rows
                bboxes.append((x1, y1, x1, y1))

    x1_crop = max(0, min(x1 for x1, y1, x2, y2 in bboxes))
    y1_crop = max(0, min(y1 for x1, y1, x2, y2 in bboxes))
    x2_crop = min(cols, max(x2 for x1, y1, x2, y2 in bboxes))
    y2_crop = min(rows, max(y2 for x1, y1, x2, y2 in bboxes))

    crop_width = x2_crop - x1_crop
    crop_height = y2_crop - y1_crop

    # Adjust cropping area to maintain aspect ratio
    if crop_width < crop_height:
        adjust = (crop_height - crop_width) / 2
        x1_crop = max(0, x1_crop - adjust)
        x2_crop = min(cols, x2_crop + adjust)
    else:
        adjust = (crop_width - crop_height) / 2
        y1_crop = max(0, y1_crop - adjust)
        y2_crop = min(rows, y2_crop + adjust)

    crop_width = x2_crop - x1_crop
    crop_height = y2_crop - y1_crop

    img_cropped = img[int(y1_crop):int(y2_crop), int(x1_crop):int(x2_crop)]
    new_img_name_cropped = "cropped_" + image_file
    cv2.imwrite(os.path.join(output_folder, new_img_name_cropped), img_cropped)
    labels_cropped = []
    for label in labels:
        numbers = list(map(float, label.strip().split()))
        if len(numbers) == 5:
            class_id, x_center, y_center, width, height = numbers
            x_center_cropped = (x_center * cols - x1_crop) / (x2_crop - x1_crop)
            y_center_cropped = (y_center * rows - y1_crop) / (y2_crop - y1_crop)
            width_cropped = width * cols / (x2_crop - x1_crop)
            height_cropped = height * rows / (y2_crop - y1_crop)
            labels_cropped.append(f"{int(class_id)} {x_center_cropped} {y_center_cropped} {width_cropped} {height_cropped}\n")
        elif len(numbers) > 5:
            class_id = numbers[0]
            coordinates = numbers[1:]
            coords_cropped = []
            for i in range(0, len(coordinates), 2):
                x_center, y_center = coordinates[i], coordinates[i + 1]
                x_center_cropped = (x_center * cols - x1_crop) / (x2_crop - x1_crop)
                y_center_cropped = (y_center * rows - y1_crop) / (y2_crop - y1_crop)
                coords_cropped.extend([x_center_cropped, y_center_cropped])
            labels_cropped.append(f"{int(class_id)} {' '.join(map(str, coords_cropped))}\n")
    with open(os.path.join(output_folder, new_img_name_cropped.replace('.jpg', '.txt')), 'w') as f:
        f.writelines(labels_cropped)

    # --- flipped1 ---
    img_flipped1 = cv2.flip(img, -1)
    new_img_name = "flipped1_" + image_file
    cv2.imwrite(os.path.join(output_folder, new_img_name), img_flipped1)
    labels_flipped1 = []
    for i in range(len(labels)):
        class_id, x_center, y_center, width, height = map(float, labels[i].strip().split())
        x_center_flipped1 = 1 - x_center
        y_center_flipped1 = 1 - y_center
        labels_flipped1.append(f"{int(class_id)} {x_center_flipped1} {y_center_flipped1} {width} {height}\n")
    with open(os.path.join(output_folder, new_img_name.replace('.jpg', '.txt')), 'w') as f:
        f.writelines(labels_flipped1)

    # --- flipped2 ---
    img_flipped2 = cv2.flip(img, 1)
    new_img_name = "flipped2_" + image_file
    cv2.imwrite(os.path.join(output_folder, new_img_name), img_flipped2)
    labels_flipped2 = []
    for i in range(len(labels)):
        class_id, x_center, y_center, width, height = map(float, labels[i].strip().split())
        x_center_flipped2 = 1 - x_center
        labels_flipped2.append(f"{int(class_id)} {x_center_flipped2} {y_center} {width} {height}\n")
    with open(os.path.join(output_folder, new_img_name.replace('.jpg', '.txt')), 'w') as f:
        f.writelines(labels_flipped2)
        
    # --- zoomedout ---
    large_img = np.zeros((int(rows * 1.5), int(cols * 1.5), 3), dtype=np.uint8)
    start_row = (large_img.shape[0] - rows) // 2
    start_col = (large_img.shape[1] - cols) // 2
    large_img[start_row:start_row+rows, start_col:start_col+cols] = img
    new_img_name = "zoomedout_" + image_file
    cv2.imwrite(os.path.join(output_folder, new_img_name), large_img)
    labels_zoomedout = []
    for i in range(len(labels)):
        class_id, x_center, y_center, width, height = map(float, labels[i].strip().split())
        x_center_zoomedout = (x_center * cols + start_col) / large_img.shape[1]
        y_center_zoomedout = (y_center * rows + start_row) / large_img.shape[0]
        width_zoomedout = width * cols / large_img.shape[1]
        height_zoomedout = height * rows / large_img.shape[0]
        labels_zoomedout.append(f"{int(class_id)} {x_center_zoomedout} {y_center_zoomedout} {width_zoomedout} {height_zoomedout}\n")
    with open(os.path.join(output_folder, new_img_name.replace('.jpg', '.txt')), 'w') as f:
        f.writelines(labels_zoomedout)
# ---------- Kod 2 Bitiş
