'''
Python 3.6 
Pytorch 0.4
Written by Hongyu Wang in Beihang university
'''
import os
import sys
import pandas as pd
import pickle as pkl
import numpy
from scipy.misc import imread, imresize, imsave
import imageio

image_path='Z:\\new_try\\Pytorch-Handwritten-Mathematical-Expression-Recognition\\Training\\'

outFile='offline-train.pkl'
oupFp_feature=open(outFile,'wb')

features={}

channels=1

sentNum=0

scpFile=open('Z:\\new_try\\Pytorch-Handwritten-Mathematical-Expression-Recognition\\train_caption.txt')
while 1:
    line=scpFile.readline().strip() # remove the '\r\n'
    print(line)
    if not line:
        break
    else:
        key = line.split('\t')[0]
        image_file = image_path + key + '.bmp'
        im = imageio.imread(image_file)
        mat = numpy.zeros([channels, im.shape[0], im.shape[1]], dtype='uint8')
        for channel in range(channels):
            image_file = image_path + key + '.bmp'
            im = imread(image_file)
            mat[channel,:,:] = im
        sentNum = sentNum + 1
        features[key] = mat
        if sentNum / 500 == sentNum * 1.0 / 500:
            print('process sentences ', sentNum)

print('load images done. sentence number ',sentNum)

pkl.dump(features,oupFp_feature)
print('save file done')
oupFp_feature.close()
