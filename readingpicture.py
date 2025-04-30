"image 2 matris"
    import matplotlib.pyplot as plt
    import cv2
    import numpy as np
    # Görüntüyü yükle
    img = cv2.imread(r"C:\Users\yavuz\OneDrive - uludag.edu.tr\2-YOLOv8\a (4).jpg")   
    print(img)
    
    # OpenCV BGR formatını kullanır, bu nedenle kanalların sırasını değiştiririz.
    rgb_matris = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    print(rgb_matris)
    
    average_rgb = np.mean(rgb_matris, axis=(0, 1))
    print(f'Average RGB values: {average_rgb}') 

    # Matrisi bir resim olarak görüntüleme
    plt.imshow(img)
    plt.show()  # Resmi görüntüler

"matris 2 image"

    import matplotlib.pyplot as plt
    import numpy as np
    import cv2  
    #OpenCV genellikle BGR (Blue-Green-Red) kanal sırasını kullanır

    # 2 kanallı (Gri-Alfa) matris oluşturma
    # Gri tonlamalı bir görüntüde, piksel değerleri 0 (siyah) ve 255 (beyaz) arasında değişir. 
    matris = np.array([[[0, 255]],  # Tamamen opak (255) siyah
                       [[255, 255]]], dtype=np.uint8)  # Tamamen opak (255) beyaz
    # Alfa kanalını görmezden gelir ve sadece gri tonlamalı kanalı alırız
    gri_matris = matris[:, :, 0]
    # Matrisi bir resim olarak görüntüleme
    plt.imshow(gri_matris, cmap='gray')
    plt.show()  # Resmi görüntüler


    # 3 kanallı (BGR) matris oluşturma (3 (yükseklik), 1 (genişlik), 3 (BGR)) = 3x1
    matris = np.array([[[255, 0, 100]], [[0, 0, 255]], [[0, 255, 0]]], dtype=np.uint8)
    
    # 3 kanallı (BGR) matris oluşturma (3 (yükseklik), 2 (genişlik), 3 (BGR)) = 3x2
    matris = np.array([[[255, 0, 100], [0, 0, 255]], 
                          [[0, 0, 0], [0, 255, 0]],
                          [[0, 255, 0], [255, 0, 100]]], dtype=np.uint8)
    
    # 4 kanallı (BGRA) matris oluşturma (5 (yükseklik), 1 (genişlik), 4 (BGRA)) = 5x1
    matris = np.array([[[255, 0, 0, 0]], [[255, 0, 0, 255]], 
                          [[0, 0, 255, 0]], [[10, 10, 10, 255]], 
                          [[100, 100, 100, 0]]], dtype=np.uint8)
    
    print(matris)
    
    # görüntüleme
    
    # OpenCV BGR formatını kullanır, bu nedenle kanalların sırasını değiştiririz.
    rgb_matris = cv2.cvtColor(matris, cv2.COLOR_BGR2RGB)
    # Matrisi bir resim olarak görüntüleme
    plt.imshow(rgb_matris)
    plt.show()  # Resmi görüntüler
    
    # Matrisi bir resim olarak görüntüleme
    cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
    cv2.imshow('Image', matris)
    # cv2.imwrite(r"C:")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

"EXTRACT"

import cv2
import numpy as np
import os
import re

# Image directory
image_directory = r"C:\Users\yavuz\OneDrive - uludag.edu.tr\1-spyder\projeler\renk_populasyon\domates\pr"
save_directory = r"C:\Users\yavuz\OneDrive - uludag.edu.tr\1-spyder\projeler\renk_populasyon\domates\pr"

# Get all image files in the directory
image_files = [f for f in os.listdir(image_directory) if f.endswith('.png') or f.endswith('.jpg')]

# Sort files by number in filename
image_files.sort(key=lambda f: int(re.sub('\D', '', f)))

for i, image_file in enumerate(image_files):
    # Load the image
    image = cv2.imread(os.path.join(image_directory, image_file))

    # Remove all [0, 0, 0] pixels and reshape the image
    new_im = image.reshape(-1, 3)  # Reshape to 2D array
    new_im = new_im[~np.all(new_im == [0, 0, 0], axis=1)]  # Remove [0, 0, 0] rows

    # Reshape the image back to 3D array
    width = int(np.sqrt(len(new_im)))  # Width is the square root of the number of pixels
    height = len(new_im) // width  # Height is the number of pixels divided by the width
    new_im = new_im[:width * height].reshape(height, width, 3)  # Reshape to 3D array

    # Convert BGR to RGB
    rgb_image = cv2.cvtColor(new_im, cv2.COLOR_BGR2RGB)

    # Save the new image in the same directory with .jpg format
    cv2.imwrite(os.path.join(save_directory, f"a{i}.jpg"), cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR))



"Other extraxt denemeleri"

import numpy as np
import cv2  
import matplotlib.pyplot as plt

# 3 kanallı (BGR) matris oluşturma (4 (yükseklik), 2 (genişlik), 3 (BGR)) = 4x2
matris = np.array([[[255, 0, 100], [0, 0, 0]], 
                   [[0, 60, 0], [0, 255, 0]],
                   [[0, 0, 0], [0, 255, 150]],
                   [[0, 255, 0], [255, 0, 100]]], dtype=np.uint8)

# Matrisi RGB şeklinde görüntüleme
rgb_matris = cv2.cvtColor(matris, cv2.COLOR_BGR2RGB)
plt.imshow(rgb_matris)
plt.show()

# [0, 0, 0] piksellerini çıkar
new_matris = [pix for row in matris for pix in row if not np.all(pix == [0, 0, 0])]

# Yeni matrisi numpy dizisine dönüştür
new_matris = np.array(new_matris)

# Matrisi yeniden şekillendir
width = int(np.sqrt(len(new_matris)))  # Genişliği karekök olarak belirle
height = len(new_matris) // width       # Yüksekliği belirle

# Görüntü matrisini yeniden şekillendir
new_matris = new_matris[:width*height].reshape((height, width, 3))

# Yeni matrisi RGB'ye dönüştürme ve görüntüleme
rgb_new_matris = cv2.cvtColor(new_matris, cv2.COLOR_BGR2RGB)
plt.imshow(rgb_new_matris)
plt.show()
    
    import matplotlib.pyplot as plt
    import cv2
    import numpy as np
    # Görüntüyü yükle
    img = cv2.imread(r"C:\Users\yavuz\OneDrive - uludag.edu.tr\1-spyder\projeler\renk_populasyon\domates\d2.jpg")
    print(img)

    # [0, 0, 0] piksellerini çıkar
    new_matris = [pix for row in img for pix in row if not np.all(pix == [0, 0, 0])]

    # Yeni matrisi numpy dizisine dönüştür
    new_matris = np.array(new_matris)

    # Matrisi yeniden şekillendir
    width = int(np.sqrt(len(new_matris)))  # Genişliği karekök olarak belirle
    height = len(new_matris) // width       # Yüksekliği belirle

    # Görüntü matrisini yeniden şekillendir
    new_matris = new_matris[:width*height].reshape((height, width, 3))

    # Yeni matrisi RGB şeklide görüntüleme
    rgb_new_matris = cv2.cvtColor(new_matris, cv2.COLOR_BGR2RGB)
    plt.imshow(rgb_new_matris)
    plt.show()
    
    # RGB görüntüyü BGR formatına geri dönüştür ve dosyaya yaz
    bgr_new_matris = cv2.cvtColor(rgb_new_matris, cv2.COLOR_RGB2BGR)
    cv2.imwrite(r"C:\Users\yavuz\OneDrive - uludag.edu.tr\1-spyder\projeler\renk_populasyon\domates\d22.jpg", bgr_new_matris)
    
"resim formatı ve sayısını değiştirme"

import cv2
import os
import re

# Image directory
image_directory = r"C:\Users\yavuz\OneDrive\Masaüstü\New folder"
save_directory = r"C:\Users\yavuz\OneDrive\Masaüstü\New folder"

# Get all image files in the directory
image_files = [f for f in os.listdir(image_directory) if f.endswith('.png') or f.endswith('.jpg')]

# Sort files by number in filename
image_files.sort(key=lambda f: int(re.sub('\D', '', f)))

# Start index
start_index = 1

for i, image_file in enumerate(image_files, start=start_index):
    # Load the image
    image = cv2.imread(os.path.join(image_directory, image_file))

    # Save the new image in the same directory with .jpg format and numbered name
    cv2.imwrite(os.path.join(save_directory, f"{i}.jpg"), image)


