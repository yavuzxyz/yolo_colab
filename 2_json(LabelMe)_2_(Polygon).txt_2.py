import os
import json
import glob
from PIL import Image

# Resimlerin ve JSON dosyalarının bulunduğu dizinler
img_dir = r"C:\Users\yavuz\OneDrive - uludag.edu.tr\1-YOLOv8\images_segment"
json_dir = r"C:\Users\yavuz\OneDrive - uludag.edu.tr\1-YOLOv8\labels_segment"

# Çıktıların kaydedileceği output klasörü
output_dir = r"C:\Users\yavuz\OneDrive - uludag.edu.tr\1-YOLOv8\output"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Kullanılan etiket listesi
label_list = ["gal"]

# Resim dosyalarını al (sadece yaygın resim formatlarını işleyelim)
img_files = glob.glob(os.path.join(img_dir, "*"))
valid_exts = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]

for img_file in img_files:
    base_name, ext = os.path.splitext(os.path.basename(img_file))
    
    # Sadece resim dosyaları üzerinde işlem yap
    if ext.lower() not in valid_exts:
        continue

    # JSON dosyasını bulmak için esnek bir arama yapalım
    json_pattern = os.path.join(json_dir, base_name + ".*")
    json_candidates = glob.glob(json_pattern)
    json_file = None
    for candidate in json_candidates:
        if candidate.lower().endswith(".json"):
            json_file = candidate
            break

    if not json_file:
        print(f"JSON dosyası bulunamadı: {json_pattern}")
        continue

    # JSON verilerini güvenli bir şekilde yükle
    try:
        with open(json_file, 'r', encoding='utf-8') as file:
            data = json.load(file)
    except Exception as e:
        print(f"JSON dosyası okunurken hata oluştu: {json_file} - {e}")
        continue

    # Resim boyutlarını al
    try:
        with Image.open(img_file) as img:
            img_width, img_height = img.size
    except Exception as e:
        print(f"Resim açılamadı: {img_file} - {e}")
        continue

    # Çıktı metin dosyasının yolu output klasöründe oluşturulacak
    txt_file_name = os.path.join(output_dir, base_name + ".txt")
    
    # .txt dosyasına yazma işlemi
    with open(txt_file_name, 'w', encoding='utf-8') as txt_file:
        for shape in data.get('shapes', []):
            points = shape.get('points', [])
            if not points:
                continue
            # Noktaları normalize et
            points_x = [point[0] / img_width for point in points]
            points_y = [point[1] / img_height for point in points]
            
            # Sınıf etiketini al ve kontrol et
            class_label = shape.get('label')
            if class_label not in label_list:
                continue
            class_index = label_list.index(class_label)
            
            # Sınıf indeksini ve koordinatları yaz
            txt_file.write(str(class_index))
            for x, y in zip(points_x, points_y):
                txt_file.write(f" {x} {y}")
            txt_file.write("\n")
