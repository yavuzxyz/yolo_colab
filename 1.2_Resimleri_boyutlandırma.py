import os
from PIL import Image

input_dir = r"C:\Users\yavuz\OneDrive - uludag.edu.tr\1-YOLOv8\abc"
output_dir = r"C:\Users\yavuz\OneDrive - uludag.edu.tr\1-YOLOv8\output"

# Oluşturulacak klasörün var olup olmadığını kontrol edmek ve varsa hata oluşturmamak için exist_ok=True kullanılır.
os.makedirs(output_dir, exist_ok=True)

# Sıralı yeni isimlendirme için sayaç başlatılır.
counter = 1

# Input dizinindeki tüm dosyalar üzerinde dönülür.
for filename in sorted(os.listdir(input_dir)):
    if filename.endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):  # Sadece belirtilen resim formatları işlenir.
        img = Image.open(os.path.join(input_dir, filename))
        width, height = img.size
        aspect = width / float(height)

        # İdeal resim boyutları tanımlanır.
        ideal_width = 640
        ideal_height = 640
        ideal_aspect = ideal_width / float(ideal_height)

        # Resmin en-boy oranıyla ideal en-boy oranı karşılaştırılır.
        if aspect > ideal_aspect:
            # Resim ideal aspect'ten genişse, sol ve sağ kenarları kırpılır.
            new_width = int(ideal_aspect * height)
            offset = (width - new_width) / 2
            resize = (offset, 0, width - offset, height)
        else:
            # Resim ideal aspect'ten dar veya eşitse, üst ve alt kenarları kırpılır.
            new_height = int(width / ideal_aspect)
            offset = (height - new_height) / 2
            resize = (0, offset, width, height - offset)

        # Kırpma ve yeniden boyutlandırma işlemleri gerçekleştirilir.
        thumb = img.crop(resize).resize((ideal_width, ideal_height), Image.ANTIALIAS)
        
        # Yeni isimlendirme a1, a2, a3 vb. şeklinde yapılır.
        new_filename = "a" + str(counter) + ".jpg"
        thumb.save(os.path.join(output_dir, new_filename), 'JPEG')

        # Sayaç bir arttırılır.
        counter += 1
