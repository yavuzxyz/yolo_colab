import json
import os
import cv2

# Dosya yollarını ayarlayın
vgg_json_file = r"C:\Users\yavuz\OneDrive - uludag.edu.tr\1-YOLOv8\labels_my-project-name_2025-04-28-10-00-11.json"
images_dir = r"C:\Users\yavuz\OneDrive - uludag.edu.tr\1-YOLOv8\abc"
output_dir = r"C:\Users\yavuz\OneDrive - uludag.edu.tr\1-YOLOv8\output"

# VGG JSON dosyasını oku
with open(vgg_json_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

print("VGG JSON dosyası yüklendi. Toplam resim sayısı:", len(data))

# Çıktı klasörünü oluştur (eğer yoksa)
os.makedirs(output_dir, exist_ok=True)

# Etiketlerin sıfırdan başlaması için offset ayarı (örneğin, 0 ile başlasın)
label_offset = 0

# Etiket dönüşümü için etiket haritası
# Örneğin; VGG JSON'da tüm etiketler "a" ise, bunu 0 olarak kullanıyoruz.
label_mapping = {"a": 0}

# VGG JSON dosyası, resim isimlerini anahtar olarak içeriyor.
for image_filename, image_info in data.items():
    # "filename" anahtarındaki değer genellikle resim dosya adıdır
    filename = image_info.get("filename", image_filename)
    image_path = os.path.join(images_dir, filename)
    image = cv2.imread(image_path)
    if image is None:
        print(f"Uyarı: {image_path} okunamadı. Atlanıyor.")
        continue
    height, width = image.shape[:2]

    # Çıktı dosyasının yolunu oluştur
    output_file = os.path.join(output_dir, os.path.splitext(filename)[0] + '.txt')

    with open(output_file, 'w', encoding='utf-8') as f_out:
        # "regions" anahtarı, resimdeki tüm anotasyonları (poligonları) içerir.
        regions = image_info.get("regions", {})
        for region in regions.values():
            shape_attrs = region.get("shape_attributes", {})
            # Sadece "polygon" tipindeki anotasyonları işleyelim
            if shape_attrs.get("name") != "polygon":
                continue
            all_points_x = shape_attrs.get("all_points_x", [])
            all_points_y = shape_attrs.get("all_points_y", [])
            if not all_points_x or not all_points_y:
                continue
            if len(all_points_x) != len(all_points_y):
                print(f"Uyarı: {filename} için x ve y noktaları eşleşmiyor.")
                continue

            # Normalize edilmiş koordinatları hesapla: x1, y1, x2, y2, ...
            norm_points = []
            for x, y in zip(all_points_x, all_points_y):
                norm_points.append(x / width)
                norm_points.append(y / height)

            # Etiketi, region_attributes içinden al; örnekte "a" -> 0 olarak belirlenmiştir.
            label_str = region.get("region_attributes", {}).get("label", "")
            label = label_mapping.get(label_str, 0) + label_offset

            # YOLO formatı: <object-class> <x1> <y1> <x2> <y2> ... <xn> <yn>
            line = f"{label} " + " ".join(f"{p:.6f}" for p in norm_points)
            f_out.write(line + "\n")

print("VGG JSON dosyasından YOLO formatına poligon etiket dosyaları oluşturuldu.")
