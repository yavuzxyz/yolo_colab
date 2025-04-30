import os
from PIL import Image

directory = r"C:\Users\yavuz\OneDrive - uludag.edu.tr\1-YOLOv8\abc"
new_directory = r"C:\Users\yavuz\OneDrive - uludag.edu.tr\1-YOLOv8\output"

if not os.path.exists(new_directory):
    os.makedirs(new_directory)

files = os.listdir(directory)
image_files = [file for file in files if file.endswith(('png', 'jpeg', "webp", "JPG", 'jpg', 'tiff', 'bmp', 'gif'))]

for file in image_files:
    img = Image.open(os.path.join(directory, file))

    # Convert the image to jpeg
    rgb_img = img.convert('RGB')

    # Extract the file name without extension
    file_name_without_extension = os.path.splitext(file)[0]

    # Create the new file name with jpg extension
    new_file_name = file_name_without_extension + '.jpg'

    # Save the new image with new extension in the new directory
    rgb_img.save(os.path.join(new_directory, new_file_name))

print("İŞLEM TAMAMLANDI.")

