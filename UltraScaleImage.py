
import cv2
from cv2 import dnn_superres
import os

from PIL import Image



program_folder_path = os.getcwd()
model = ""
directory_out = "UltraImage"
directory_in = "ImputImage"

directory_out_path = os.path.join(program_folder_path, directory_out)
directory_in_path = os.path.join(program_folder_path,directory_in)

#create directory if it doesn't exist
if  not os.path.isdir(directory_out_path):
    mode = 0o666
    os.mkdir(directory_out_path,mode)
    print("creating directory UltraImage")

#create directory if it doesn't exist
if not os.path.isdir(directory_in_path):

    mode = 0o666
    os.mkdir(directory_in_path,mode)
    print("creating directory ImputImage")
    exit()
    
images_paths =  os.listdir(directory_in_path)
if len(images_paths)==0:
    print("There are no images in the directory")
    exit()


sr = dnn_superres.DnnSuperResImpl_create()
path = 'TrainedModels\EDSR_x4.pb'
sr.readModel(path)
sr.setModel('edsr', 4)

for image_name in images_paths:
    image_full_path = os.path.join(directory_in_path,image_name)
    image = cv2.imread(image_full_path)
    if image is None:
        print(f"Failed to load image: {image_full_path}")
        continue

    image_out=os.path.join(directory_out_path,image_name)

    upscaled = sr.upsample(image)
    cv2.imwrite(image_out, upscaled)
