import json
import os

input_dir = r"C:\Users\yavuz\OneDrive - uludag.edu.tr\1-YOLOv8\cbs etiketlenenler"
output_dir = r"C:\Users\yavuz\OneDrive - uludag.edu.tr\1-YOLOv8\output"

# COCO json dosyasını oku
with open(os.path.join(input_dir, "aculops labels_my-project-name_2025-03-21-03-56-31.json"), 'r', encoding='utf-8') as f:
    data = json.load(f)

print(data.keys())

# Çıktı klasörünü oluştur (eğer yoksa)
os.makedirs(output_dir, exist_ok=True)

# Eğer etiketlerin sıfırdan başlamasını istiyorsanız, bu ofseti -1 olarak ayarlayın.
label_offset = 0

# COCO etiketlerini YOLO formatında poligon segmentasyonuna dönüştür
for img in data['images']:
    filename = img['file_name']
    width = img['width']
    height = img['height']

    output_file = os.path.join(output_dir, os.path.splitext(filename)[0] + '.txt')

    with open(output_file, 'w', encoding='utf-8') as f:
        for ann in data['annotations']:
            if ann['image_id'] == img['id']:
                segmentations = ann.get('segmentation', [])
                if not segmentations:
                    continue
                # İlk poligonu kullanıyoruz
                poly = segmentations[0]
                norm_points = []
                for i, coord in enumerate(poly):
                    if i % 2 == 0:  # x koordinatları
                        norm_points.append(coord / width)
                    else:           # y koordinatları
                        norm_points.append(coord / height)
                # COCO'daki kategori id'sine label_offset ekleyerek, etiketlerin sıfırdan başlamasını sağlıyoruz.
                label = int(ann['category_id']) + label_offset
                line = f"{label} " + " ".join(f"{p:.6f}" for p in norm_points)
                f.write(line + "\n")

print("Poligon tabanlı etiket dosyaları oluşturuldu.")
