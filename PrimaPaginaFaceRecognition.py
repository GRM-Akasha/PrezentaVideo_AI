
import cv2
import os 
import random
import numpy as np
from keras.engine.topology import get_source_inputs
from keras.models import load_model
from keras.preprocessing import image as kimage
from keras_vggface.utils import preprocess_input

from skimage import exposure
from matplotlib import pyplot as plt
from PIL import Image
plt.rcParams.update({'figure.max_open_warning': 0})
from alexnet_pytorch import AlexNet

import UltraScaleImage as myh

#model = load_model('path_to_your_model.h5')
#model = AlexNet.from_pretrained('alexnet', num_classes=10)

#Aplication Progrtam Interface (API)
from tensorflow.keras.models import Model
import tensorflow.keras.layers as lay

indexImageSave:(int) = 1
def SaveBRGArrayAsImage(array,path):
    global indexImageSave
    array = cv2.cvtColor(array, cv2.COLOR_BGR2RGB)
    image_name = "Image_Face_" + str(indexImageSave) + ".png"
    indexImageSave = indexImageSave + 1
    image_path = os.path.join(path,image_name)
    Image.fromarray(array).save(image_path)
    return image_name

def preprocess_image(image_path):
    img = Image.open(image_path)
    img = img.resize((160, 160))  # Resize image to match FaceNet input size
    img = kimage.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img, version=2)  # Preprocess input according to FaceNet requirements
    return img

def IncrementalResizeImage(array,current_x,current_y,new_x,new_y,steps):#to be done
    x_step:int = new_x // steps
    y_step:int = new_y // steps
    if(current_x>new_x):
       x_step=-x_step
    if(current_y>new_y):
       y_step=-y_step

    next_x:int = current_x + x_step
    next_y:int = current_y + y_step
    while next_x!=new_x or next_y!=new_y:
        array = cv2.resize( array, [next_x, next_y], interpolation = cv2.INTER_CUBIC)
        next_x = next_x + x_step
        next_y = next_y + y_step
    array = cv2.resize( array, [new_x, new_y], interpolation = cv2.INTER_CUBIC)
    return array


def DeleteFilesInDirectory(directory_path):
   try:
     files = os.listdir(directory_path)
     for file in files:
       file_path = os.path.join(directory_path, file)
       if os.path.isfile(file_path):
         os.remove(file_path)
     print("All files deleted successfully.")
   except OSError:
     print("Error occurred while deleting files.")

cap = cv2.VideoCapture(0)

program_folder_path = os.getcwd()
directory_name = "Photo_saved"
directory_path = os.path.join(program_folder_path, directory_name)

if os.path.isdir(directory_path):
    DeleteFilesInDirectory(directory_path)
else:
    mode = 0o666
    os.mkdir(directory_path,mode)

#dir_list = os.listdir(path) REMINDER asta e ls
#os.chdir('../') REMINDER asta ma scoate din folder

crop_face = np.zeros((227,227))
if not (cap.isOpened()):
    print("Could not open video device")

#cap.set(3, 176)
#cap.set(4, 144)
    
image_index: int = 1

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    #print(type(frame))
    if not ret:
        break


    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #frame_gray_adjust = exposure.rescale_intensity(frame_gray, in_range=(170/255, 250/255), out_range=(0, 1))
    frame_gray_equ = cv2.equalizeHist(frame_gray)

    face_classifier = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    face = face_classifier.detectMultiScale(
        frame_gray_equ, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40)
    )
    for (x, y, w, h) in face:
        crop_face = frame[y:y+h,x:x+w]
        crop_face = cv2.resize(crop_face,[166,166],interpolation=cv2.INTER_CUBIC)
    
    # Perform face recognition using the pretrained FaceNet model
        #embeddings = model.predict(crop_face)
        #print(embeddings)
        #print("Embeddings:", embeddings)
        #crop_face = array = cv2.resize( array, [166, 166], interpolation = cv2.INTER_CUBIC)
        crop_path = SaveBRGArrayAsImage(crop_face,directory_path)

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 4)


    #hist, bins = np.histogram(frame_gray,256,[0, 256])
    #hist2, bins = np.histogram(frame_gray_equ,256,[0, 256])
    cv2.imshow('Original', frame)
    cv2.imshow('Gray', frame_gray)
    cv2.imshow('GrayEqu', frame_gray_equ)
    #cv2.imshow('Adjust', frame_gray_adjust)
    cv2.imshow('Crop', crop_face)
    #plt.figure("gray")
    #plt.hist(hist,bins=255,range=(0,255))
    #plt.figure("gray equ")
    #plt.hist(hist2,bins=255,range=(0,255))

    # Update the histogram plot
    #plt.pause(0.5) # Pause for a short time to update the plot

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    if indexImageSave>=10:
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
