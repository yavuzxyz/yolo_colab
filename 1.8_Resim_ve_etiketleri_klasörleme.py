import sys
import os
import argparse
import shutil
from random import sample

class Args:
    train = 80
    validation = 19
    test = 1
    folder = r"C:\Users\yavuz\OneDrive - uludag.edu.tr\1-YOLOv8\output"  # Kaynak klasör (resimler ve etiketler burada)
    dest = r"C:\Users\yavuz\OneDrive - uludag.edu.tr\1-YOLOv8\output123"  # Çıktı klasörü artık "output_detect"

args = Args()

# Check if the code is running inside Spyder, and if so use the hardcoded Args
if any('spyder' in string for string in sys.argv):
    args = Args()
else:
    # Else, parse the command line arguments as usual
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--train", type=int, default=args.train, help="Percentage of train set")
    parser.add_argument("--validation", type=int, default=args.validation, help="Percentage of validation set")
    parser.add_argument("--test", type=int, default=args.test, help="Percentage of test set")
    parser.add_argument("--folder", type=str, default=args.folder, help="Folder that contain image and label files")
    parser.add_argument("--dest", type=str, default=args.dest, help="Destination folder for output_detect")

    args = parser.parse_args()

def get_difference_from_2_list(list1, list2):
    set_list1 = set(list1)
    set_list2 = set(list2)
    diff = list(set_list1.difference(set_list2))
    return diff

def get_split_data(list_id):
    # Total count is global 'count'
    n_train = (count * args.train) / 100
    train = sample(list_id, int(n_train))
    list_id = get_difference_from_2_list(list_id, train)
    n_valid = (count * args.validation) / 100
    valid = sample(list_id, int(n_valid))
    test = get_difference_from_2_list(list_id, valid)
    return train, valid, test

def make_folder():
    folders = ["images", "labels"]
    inner_folders = ["train", "val", "test"]

    if not os.path.isdir(args.dest):
        os.mkdir(args.dest)

    for folder in folders:
        path = os.path.join(args.dest, folder)
        if not os.path.isdir(path):
            os.mkdir(path)
        
        for in_folder in inner_folders:
            inner_path = os.path.join(path, in_folder)
            if not os.path.isdir(inner_path):
                os.mkdir(inner_path)     

def copy_image(file, id_folder):
    inner_folders = ["train", "val", "test"]

    # Image
    source = os.path.join(args.folder, file)
    out_dest = os.path.join(args.dest, 'images')
    destination = os.path.join(out_dest, inner_folders[id_folder])    
    try:
        shutil.copy(source, destination)
    except shutil.SameFileError:
        print("Source and destination represent the same file.")

    # Labels   
    separator = file.find(".")
    filename = file[0:separator]
    for ext in [".txt", ".json"]:
        label_file = filename + ext
        if os.path.isfile(os.path.join(args.folder, label_file)):
            source = os.path.join(args.folder, label_file)
            out_dest = os.path.join(args.dest, 'labels')
            destination = os.path.join(out_dest, inner_folders[id_folder])
            try:
                shutil.copy(source, destination)
            except shutil.SameFileError:
                print("Source and destination represent the same file.") 
            break

# Check train set percentage validity
if (args.train < args.validation) or (args.train < args.test):
    print("Train set must have the highest percentage")
    exit()

# Check total percentage
total = args.train + args.validation + args.test
if total > 100:
    print("Total percentage must be 100%")
    exit()

# Count number of data
count = 0
list_id = []
for file in os.listdir(args.folder):
    if file.endswith(".jpg") or file.endswith(".png"):       
        list_id.append(count)
        count += 1

train, valid, test = get_split_data(list_id)
make_folder()

count = 0
for file in os.listdir(args.folder):
    if file.endswith(".jpg") or file.endswith(".png"):
        if count in train:          
            copy_image(file, 0)
        elif count in valid:         
            copy_image(file, 1)
        else:         
            copy_image(file, 2)            
        count += 1
