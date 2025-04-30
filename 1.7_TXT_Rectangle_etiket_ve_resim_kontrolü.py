import os
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_labels(image_file, label_file):
    """
    Verilen resim ve etiket dosyası için bounding box'ları çizerek görselleştirir.
    """
    # Resmi oku ve doğrula
    image = cv2.imread(image_file)
    if image is None:
        print(f"Hata: {image_file} okunamadı.")
        return
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width = image.shape[:2]

    # Etiket dosyasını oku
    try:
        with open(label_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"Hata: {label_file} bulunamadı.")
        return

    # Plot oluştur ve resmi ekle
    fig, ax = plt.subplots(1, figsize=(10, 10))
    ax.imshow(image)
    ax.set_title(os.path.basename(image_file))
    ax.axis('off')

    # Her satır için bounding box'ı çiz
    for line in lines:
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            print(f"Uyarı: Beklenmeyen format ({label_file}): {line}")
            continue

        try:
            class_id = int(parts[0])
            x_center = float(parts[1])
            y_center = float(parts[2])
            box_width = float(parts[3])
            box_height = float(parts[4])
        except ValueError:
            print(f"Uyarı: Değer dönüştürme hatası ({label_file}): {line}")
            continue

        # Denormalize: normalize edilmiş koordinatları resim boyutuna göre ayarla
        x_center *= width
        y_center *= height
        box_width *= width
        box_height *= height

        # Sol üst köşe koordinatlarını hesapla
        x_left = x_center - box_width / 2
        y_top = y_center - box_height / 2

        # Bounding box'ı çiz
        rect = patches.Rectangle((x_left, y_top), box_width, box_height,
                                 linewidth=2, edgecolor='red', facecolor='none')
        ax.add_patch(rect)
        # Opsiyonel: Bounding box'ın üzerine sınıf numarasını yaz
        ax.text(x_left, y_top - 5, str(class_id),
                color='yellow', fontsize=12, backgroundcolor='black')

    plt.tight_layout()
    plt.show()
    plt.close(fig)

def get_files_from_dir(directory, extensions):
    """
    Belirtilen dizindeki, verilen uzantılara sahip dosyaların sıralı listesini döndürür.
    """
    files = [os.path.join(directory, f) for f in os.listdir(directory)
             if os.path.isfile(os.path.join(directory, f)) and f.lower().endswith(extensions)]
    return sorted(files)

def main():
    # Resim ve etiket klasörlerinin yollarını ayarlayın
    # image_dir = r"C:\Users\yavuz\OneDrive - uludag.edu.tr\1-YOLOv8\images_detect"
    # label_dir = r"C:\Users\yavuz\OneDrive - uludag.edu.tr\1-YOLOv8\labels_detect"
    image_dir = r"C:\Users\yavuz\OneDrive - uludag.edu.tr\1-YOLOv8\output"
    label_dir = r"C:\Users\yavuz\OneDrive - uludag.edu.tr\1-YOLOv8\output"

    # Desteklenen resim uzantıları: jpg, jpeg, png
    image_files = get_files_from_dir(image_dir, ('.jpg', '.jpeg', '.png'))
    # Etiket dosyaları: .txt
    label_files = get_files_from_dir(label_dir, ('.txt',))

    # Her resmin ilgili etiket dosyasını eşleştirmek için, dosya baz isimlerini kullanın.
    label_dict = {os.path.splitext(os.path.basename(f))[0]: f for f in label_files}

    for image_file in image_files:
        base_name = os.path.splitext(os.path.basename(image_file))[0]
        label_file = label_dict.get(base_name)
        if label_file:
            draw_labels(image_file, label_file)
        else:
            print(f"Uyarı: {image_file} için etiket dosyası bulunamadı.")

if __name__ == '__main__':
    main()
