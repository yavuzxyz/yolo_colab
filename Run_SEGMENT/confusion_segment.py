from ultralytics import YOLO
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import glob
import cv2

# Model yolu
model_path = "best.pt"
model = YOLO(model_path)

# Görüntü ve etiket dosyalarının yolları
image_files = glob.glob("images/*.jpg")
label_files = glob.glob("labels/*.txt")

# Sınıf sayısı ve adları
BACKGROUND_CLASS = -1
classes = [BACKGROUND_CLASS, 0] # Arka plan ve diğer sınıflar
class_names = ["background", "Chestnut gall"] # Arka plan ve diğer sınıf isimleri

# # Sınıf sayısı ve adları
# BACKGROUND_CLASS = -1
# classes = [BACKGROUND_CLASS, 0, 1, 2] # Arka plan ve diğer sınıflar
# class_names = ["background", "Healthy leaf", "Thrips damage", "Spider mites damage"] # Arka plan ve diğer sınıf isimleri


# Gerçek ve tahmin edilen sınıflar
y_true = []
y_pred = []

for image_path, label_path in zip(image_files, label_files):
    # Tahminleri al
    img = cv2.imread(image_path)
    results = model(img, conf=0.50)
    predicted_classes = [int(cls) for cls in results[0].boxes.cls.tolist()]

    # Gerçek etiketleri al
    class_ids = []
    with open(label_path, 'r') as file:
        for line in file.readlines():
            class_index = int(line.split()[0]) # Sınıf indeksi ilk eleman
            class_ids.append(class_index)

    # Eksik etiketlere arka plan sınıfı ekle
    while len(predicted_classes) < len(class_ids):
        predicted_classes.append(BACKGROUND_CLASS)
    while len(class_ids) < len(predicted_classes):
        class_ids.append(BACKGROUND_CLASS)

    y_true.extend(class_ids)
    y_pred.extend(predicted_classes)

# Confusion matrisi oluştur
cm = confusion_matrix(y_true, y_pred, labels=classes)

# Confusion matrisini metin olarak yazdır
print("\t" + "\t".join(class_names))  # sınıf isimlerini yazdır
for i, row in enumerate(cm):
    print(f"{class_names[i]}\t", end="")
    print("\t".join(str(cell) for cell in row))


# Matplotlib ile confusion matrisi çizdir
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt=".2f", xticklabels=class_names, yticklabels=class_names)
plt.ylabel('Real')
plt.xlabel('Predicted')
plt.title("Confusion Matrix")

# Görüntüyü bir dosyaya kaydet
image = "confusionmatrix.png"
plt.savefig(image)
plt.close() # Eklediğim bu satır, gösterimin kaldırılmasını sağlar

# Kaydedilen dosyayı OpenCV ile oku
image = cv2.imread(image)

# Görüntüyü OpenCV ile göster
cv2.imshow("Confusion Matrix", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
