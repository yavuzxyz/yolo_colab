import os
import cv2
import numpy as np
import math

def process_images(path, labels_folder, output_folder, rotation_angles=[90, 180, 270]):
    # Bu fonksiyon kullanılmamış; ana işlemler alt kısımda yer alıyor.
    pass

# Klasör yollarını tanımla
images_folder = r"C:\Users\yavuz\OneDrive - uludag.edu.tr\1-YOLOv8\abc_images"
labels_folder = r"C:\Users\yavuz\OneDrive - uludag.edu.tr\1-YOLOv8\abc_labels"
output_folder = r"C:\Users\yavuz\OneDrive - uludag.edu.tr\1-YOLOv8\output"

# Klasör kontrolü: görüntü dosyalarını al
if os.path.isdir(images_folder):
    image_files = os.listdir(images_folder)
elif os.path.isfile(images_folder):
    image_files = [os.path.basename(images_folder)]
    images_folder = os.path.dirname(images_folder)
else:
    print(f"Invalid images_folder: {images_folder}")
    exit()

# Klasör kontrolü: etiket dosyalarını al
if os.path.isdir(labels_folder):
    label_files = os.listdir(labels_folder)
elif os.path.isfile(labels_folder):
    label_files = [os.path.basename(labels_folder)]
    labels_folder = os.path.dirname(labels_folder)
else:
    print(f"Invalid labels_folder: {labels_folder}")
    exit()

rotation_angles = [90, 180, 270]

# Çıktı klasörünü oluştur (eğer yoksa)
os.makedirs(output_folder, exist_ok=True)

# ---------- Kod 1 Başlangıç ----------
# Her bir görüntü dosyası için augmentasyon işlemleri
for image_file in image_files:
    # Görüntüyü oku
    img = cv2.imread(os.path.join(images_folder, image_file))
    if img is None:
        print(f"Uyarı: {image_file} okunamadı.")
        continue
    rows, cols, _ = img.shape

    # --- Döndürme işlemi ---
    for angle in rotation_angles:
        # Görüntüyü belirtilen açıda döndür
        M = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)
        img_rotated = cv2.warpAffine(img, M, (cols, rows))

        # Yeni isimle output_folder'a kaydet
        new_img_name = f"rotated{angle}_" + image_file
        cv2.imwrite(os.path.join(output_folder, new_img_name), img_rotated)

        # Etiket dosyasını oku
        with open(os.path.join(labels_folder, image_file.replace('.jpg', '.txt')), 'r') as f:
            labels = f.readlines()

        for i in range(len(labels)):
            # Etiketi parse et
            line = labels[i].strip().split()
            class_id = int(line[0])
            points = np.array(line[1:], dtype=float).reshape(-1, 2)  # koordinat çiftleri

            # Döndürme merkezini hesapla
            rot_center_x = cols / 2.0
            rot_center_y = rows / 2.0

            # Dönüşüm matrisini oluştur (aynı açı)
            M_rot = cv2.getRotationMatrix2D((rot_center_x, rot_center_y), angle, 1)

            # Her bir noktayı döndür ve normalize et
            points_rotated = []
            for point in points:
                x, y = point
                # Not: Noktaları orijinal görüntü boyutlarına göre döndürmek için
                x_rot = M_rot[0, 0] * x * cols + M_rot[0, 1] * y * rows + M_rot[0, 2]
                y_rot = M_rot[1, 0] * x * cols + M_rot[1, 1] * y * rows + M_rot[1, 2]
                x_rot /= cols
                y_rot /= rows
                points_rotated.append([x_rot, y_rot])

            # Yeni etiket satırını oluştur
            labels[i] = f"{class_id} " + " ".join(map(str, np.array(points_rotated).flatten())) + "\n"

        # Yeni etiketleri output_folder'a kaydet
        with open(os.path.join(output_folder, new_img_name.replace('.jpg', '.txt')), 'w') as f:
            f.writelines(labels)

    # --- Aydınlatma (Light) ---
    img_light = cv2.convertScaleAbs(img, alpha=1, beta=60)
    new_img_name_light = "light_" + image_file
    cv2.imwrite(os.path.join(output_folder, new_img_name_light), img_light)

    # --- Gauss Gürültüsü ---
    mean = 0
    var = 190  # Gürültü şiddeti
    sigma = var**0.5
    gauss = np.random.normal(mean, sigma, img.shape)
    img_noisy = np.clip(img + gauss, 0, 255).astype(np.uint8)
    new_img_name_noisy = "noisy_" + image_file
    cv2.imwrite(os.path.join(output_folder, new_img_name_noisy), img_noisy)

    # --- Gölge Efekti ---
    shadow = np.ones(img.shape, dtype="uint8") * 255
    cv2.rectangle(shadow, (int(cols/4), int(rows/4)), (int(3*cols/4), int(3*rows/4)), (170, 170, 170), -1)
    img_shadowed = cv2.addWeighted(img, 0.6, shadow, 0.5, 0)
    new_img_name_shadowed = "shadowed_" + image_file
    cv2.imwrite(os.path.join(output_folder, new_img_name_shadowed), img_shadowed)

    # --- Gri Ton (Grayscale) Dönüşümü ---
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Gri resmi 3 kanallı hale getirerek kaydedelim (örneğin JPG formatında)
    img_gray_3ch = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
    new_img_name_gray = "gray_" + image_file
    cv2.imwrite(os.path.join(output_folder, new_img_name_gray), img_gray_3ch)

    # Etiket dosyalarını oku ve orijinal etiketleri output_folder'a kopyala
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
# Cropping, flipping, zoomed out işlemleri
for image_file in image_files:
    img = cv2.imread(os.path.join(images_folder, image_file))
    rows, cols, _ = img.shape
    aspect_ratio = cols / rows

    with open(os.path.join(labels_folder, image_file.replace('.jpg', '.txt')), 'r') as f:
        labels = f.readlines()

    # Cropping işlemi: Bounding box'ları toplayıp uygun kesim bölgesini hesaplama
    bboxes = []
    for label in labels:
        numbers = list(map(float, label.strip().split()))
        if len(numbers) == 5:  # Bounding box
            class_id, x_center, y_center, width_box, height_box = numbers
            x1 = (x_center - width_box / 2) * cols
            y1 = (y_center - height_box / 2) * rows
            x2 = (x_center + width_box / 2) * cols
            y2 = (y_center + height_box / 2) * rows
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
    crop_center_x = x1_crop + crop_width / 2
    crop_center_y = y1_crop + crop_height / 2

    if crop_width / crop_height > aspect_ratio:
        crop_height = crop_width / aspect_ratio
    else:
        crop_width = crop_height * aspect_ratio

    x1_crop = max(0, crop_center_x - crop_width / 2)
    y1_crop = max(0, crop_center_y - crop_height / 2)
    x2_crop = min(cols, crop_center_x + crop_width / 2)
    y2_crop = min(rows, crop_center_y + crop_height / 2)

    img_cropped = img[int(y1_crop):int(y2_crop), int(x1_crop):int(x2_crop)]
    new_img_name_cropped = "cropped_" + image_file
    cv2.imwrite(os.path.join(output_folder, new_img_name_cropped), img_cropped)
    labels_cropped = []
    for label in labels:
        numbers = list(map(float, label.strip().split()))
        if len(numbers) == 5:  # Bounding box
            class_id, x_center, y_center, width_box, height_box = numbers
            x_center_cropped = (x_center * cols - x1_crop) / (x2_crop - x1_crop)
            y_center_cropped = (y_center * rows - y1_crop) / (y2_crop - y1_crop)
            width_cropped = width_box * cols / (x2_crop - x1_crop)
            height_cropped = height_box * rows / (y2_crop - y1_crop)
            labels_cropped.append(f"{int(class_id)} {x_center_cropped} {y_center_cropped} {width_cropped} {height_cropped}\n")
        elif len(numbers) > 5:  # Polygon
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
        line = labels[i].strip().split()
        class_id = int(line[0])
        points = np.array(line[1:], dtype=float).reshape(-1, 2)
        points_flipped1 = 1 - points
        labels_flipped1.append(f"{class_id} " + " ".join(map(str, points_flipped1.flatten())) + "\n")
    with open(os.path.join(output_folder, new_img_name.replace('.jpg', '.txt')), 'w') as f:
        f.writelines(labels_flipped1)

    # --- flipped2 ---
    img_flipped2 = cv2.flip(img, 0)
    new_img_name = "flipped2_" + image_file
    cv2.imwrite(os.path.join(output_folder, new_img_name), img_flipped2)
    labels_flipped2 = []
    for i in range(len(labels)):
        line = labels[i].strip().split()
        class_id = int(line[0])
        points = np.array(line[1:], dtype=float).reshape(-1, 2)
        points[:, 1] = 1 - points[:, 1]
        labels_flipped2.append(f"{class_id} " + " ".join(map(str, points.flatten())) + "\n")
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
        line = labels[i].strip().split()
        class_id = int(line[0])
        points = np.array(line[1:], dtype=float).reshape(-1, 2)
        points[:, 0] = (points[:, 0] * cols + start_col) / large_img.shape[1]
        points[:, 1] = (points[:, 1] * rows + start_row) / large_img.shape[0]
        labels_zoomedout.append(f"{class_id} " + " ".join(map(str, points.flatten())) + "\n")
    with open(os.path.join(output_folder, new_img_name.replace('.jpg', '.txt')), 'w') as f:
        f.writelines(labels_zoomedout)
# ---------- Kod 2 Bitiş