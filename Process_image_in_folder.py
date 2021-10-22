import os
from PIL import Image
import splitfolders 
import matplotlib.pyplot as plt
import numpy as np
import glob, os, sys
from splitfolders import *
import shutil
from sklearn.model_selection import train_test_split

pth     = os.getcwd()
newsize = (170,55)
folder  = pth
source  = "Z:/new_try/Pytorch-Handwritten-Mathematical-Expression-Recognition/off_image_val/archive/batch_1/background_images"

#preprocessing the image dataset

for filepath in glob.iglob(os.path.join(folder, '*.jpg')):
    image = Image.open(filepath).convert('L')
    #print('Maximum RGB value in this image {}'.format(image.max()))
    max_value = max(list(image.getdata()));
    threshold = int(((3)*max_value)//4);
    new_filepath = os.path.splitext(filepath)[0] + '.bmp'
    image = image.point(lambda p: p < threshold and 255)
    image = image.resize(newsize,Image.BICUBIC)
    image.save(new_filepath)
    os.remove(filepath)

#Splitting data into training and test set

#data = os.listdir(input_folder)
#train, valid = train_test_split(data, test_size=0.2, random_state=1)

#copying test and train datasets to separate folders



