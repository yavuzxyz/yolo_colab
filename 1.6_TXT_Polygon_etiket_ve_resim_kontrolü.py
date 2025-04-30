import os
import matplotlib.pyplot as plt
import cv2
from matplotlib.patches import Polygon

def draw_labels(img_file, lbl_file):
    # Resmi oku
    image = cv2.imread(img_file)
    if image is None:
        print(f"Uyarı: {img_file} okunamadı.")
        return
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width = image.shape[:2]

    # Etiket dosyasını oku
    try:
        with open(lbl_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"Uyarı: Etiket dosyası {lbl_file} bulunamadı.")
        return

    # Görselleştirme için çizim oluştur
    fig, ax = plt.subplots(1)
    ax.imshow(image)

    for line in lines:
        line = line.strip()
        if not line:
            continue
        elements = line.split()
        # İlk eleman etiket numarası (class_id)
        class_id = int(elements[0])
        
        # Poligonun noktalarını al (normalize edilmiş koordinatlar)
        points = [(float(elements[i]), float(elements[i+1])) for i in range(1, len(elements), 2)]
        # Orijinal piksel koordinatlarına çevir
        points = [(x * width, y * height) for (x, y) in points]

        # Poligonu oluştur ve çizime ekle
        poly = Polygon(points, edgecolor='r', alpha=0.5, fill=False)
        ax.add_patch(poly)
        # İsteğe bağlı: Poligonun içine sınıf numarasını yazdırabilirsiniz
        centroid = (sum(x for x, y in points) / len(points), sum(y for x, y in points) / len(points))
        ax.text(centroid[0], centroid[1], str(class_id), color='blue', fontsize=12, ha='center')

    plt.axis('off')
    plt.show()

# Klasör yolları
images_dir = r"C:\Users\yavuz\OneDrive - uludag.edu.tr\1-YOLOv8\images_detect"
labels_dir = r"C:\Users\yavuz\OneDrive - uludag.edu.tr\1-YOLOv8\labels_detect"

# Klasörlerin geçerli olup olmadığını kontrol et
if os.path.isdir(images_dir) and os.path.isdir(labels_dir):
    image_files = [os.path.join(images_dir, f) for f in os.listdir(images_dir)
                   if os.path.isfile(os.path.join(images_dir, f)) and f.lower().endswith(".jpg")]
    label_files = [os.path.join(labels_dir, f.replace('.jpg', '.txt')) for f in os.listdir(images_dir)
                   if os.path.isfile(os.path.join(images_dir, f)) and f.lower().endswith(".jpg")]
elif os.path.isfile(images_dir) and os.path.isfile(labels_dir):
    image_files = [images_dir]
    label_files = [labels_dir]
else:
    print("Girdiğiniz yollar geçerli değil. Lütfen kontrol edin.")
    exit()

# Her resim ve etiket dosyası için çizim işlemini gerçekleştir
for img_file, lbl_file in zip(image_files, label_files):
    draw_labels(img_file, lbl_file)
