# -*- coding: utf-8 -*-
"""
Created on Sun Jan 21 12:09:02 2018

@author: fcp018
"""
###############################################################################
# Libraries import
import numpy as np
import pandas as pd
import os
import sys
import matplotlib.pyplot as plt
import cv2

import itertools
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.utils import np_utils
from keras.models import model_from_json
from sklearn.metrics import log_loss
from keras import backend as K
from keras.utils import np_utils
from keras.layers.normalization import BatchNormalization
K.set_image_dim_ordering('th')
from keras.preprocessing.image import ImageDataGenerator

###############################################################################

# Quick list of all the location of raw data
# "imgs" is the main folder for all of raw data
# "imgs_sub" is the folder for the 500 images subset with 50 images per class
# "imgs_sub2" is the folder for the 500 images subset but 
# gathered in only two classes (0= safe (50 images), 1= distracted(450 images))

image_drv = './imgs/'
train_drv = image_drv + "train/"
test_drv = image_drv + "test/"

# For driver classification list, add "_sub" or "_sub2 at the end 
# depending on the folder used
driver_imgs_list = "./driver_imgs_list.csv"  

# Define some image and class parameters
# color type: 1 = grey, 3 = rgb
color_type = 1
img_rows, img_cols = 224, 224
nb_classes = 10  # 2 for "imgs_sub2"; 10 for others

###############################################################################

# Read the training set classification list
train_driver_list = pd.read_csv(driver_imgs_list)
print("Training data quick read: \n{}".format(train_driver_list.head(10)))

test_images_list = os.listdir(test_drv)
print("\nTest data filnames quick read: \n{}".format(test_images_list[0:9]))


# Get the unique driver ids
driver_ids = []
for id, driver in train_driver_list.iterrows():
    if driver['subject'] not in driver_ids:
        driver_ids.append(driver['subject'])
print("Found {} drivers: {}".format(len(driver_ids), driver_ids))



###############################################################################
# List validation ids


# Function to split drivers into training and validation set.
# Note the split is by driver ids. Same driver not in both sets.
def partition(lst, n):
    division = len(lst) / n
    return [lst[round(division * i):round(division * (i + 1))] for i in range(n)]

np.random.seed(123)  # set seed could be used here to duplicate results.
np.random.shuffle(driver_ids)
valid_ids = partition(driver_ids, 8)

valid_ids= [['p066', 'p021', 'p024'],
 ['p026', 'p072', 'p015'],
 ['p052', 'p061', 'p042', 'p081'],
 ['p016', 'p041', 'p050'],
 ['p047', 'p064', 'p035'],
 ['p049', 'p002', 'p012', 'p039'],
 ['p056', 'p051', 'p022'],
 ['p075', 'p014', 'p045']]

valid_ids
i=6  # choose the valid id set to use for this data generation.

valid_ids[i]


def train_valid_driver_split(train_driver_list):
    
    # Chose random # of drivers depending on "split" % provided
    
    
    train_split = [ driver for driver in train_driver_list if driver not in valid_ids[i]]
    # Take the remaining drivers into the validation list
    valid_split = valid_ids[i]
    return train_split, valid_split

# Create the training and validation splits for driver ids
driver_train_list, driver_valid_list = train_valid_driver_split(driver_ids)
print(driver_train_list)
print(driver_valid_list)

###############################################################################

#Fucntion to create train and validation data set using the driver lists.

def get_train_validation_data(train_driver_list, filter = driver_train_list):

    images = []
    labels = []
    total = 0  # use total to verify all images read for sanity check
    
     # combine two "for loops" in one line to get a pandas series from a generator.
    for driver_row in [ drvr for drvr in train_driver_list[train_driver_list.subject.isin(filter)].loc[:, 'classname':'img'].iterrows() ]:   # if drvr[1]['subject'] in filter 
        driver = driver_row[1]  # Drop index # from the pd series
        label = int((driver['classname'])[1:]) # get corresponding class #
        
        # read and load image
        path = train_drv + "c"+ str(label) + "/" + driver['img']
        if color_type == 1:
            img = cv2.imread(path, 0).transpose()  # read in grey
        elif color_type == 3:
            img = cv2.imread(path).transpose() # read in rgb colors
            
        # resize images
        image = cv2.resize(img, (img_cols, img_rows))
        
        # append image and class # from each iteration into lists
        images.append(image)
        labels.append(label)
        total += 1
        if total % 100 == 0:
            print("."),
    print("\nProcessed {} rows.".format(total))
    
    images = np.array(images, dtype=np.uint8)
    images = images.astype('float32')
    images /= 255
    
    labels = np.array(labels, dtype=np.uint8)
    labels = np_utils.to_categorical(labels, nb_classes)

    return images, labels


###############################################################################
    
# create train and validation sets for images and class labels.
print("Creating training set:")
x_train, y_train = get_train_validation_data(train_driver_list, driver_train_list)
print("Creating validation data:")
x_valid, y_valid = get_train_validation_data(train_driver_list, driver_valid_list)

# reshape images for the convolution layers
x_train = x_train.reshape(x_train.shape[0], color_type, img_rows, img_cols)
x_valid = x_valid.reshape(x_valid.shape[0], color_type, img_rows, img_cols)


from skimage.transform import rotate
def show_image(image):
    img = image.squeeze()
    plt.axis("off")
    #plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.imshow(np.fliplr(rotate(img, 270)), cmap='gray', interpolation='nearest')  
    # Manipulation to display image in the correct orientation!
    plt.show() 
    #print image.shape
    #plt.imshow(image)
    
    

# create a grid of 3x3 images
for i in range(0, 9):
    #plt.imshow(x_train[i].reshape(img_rows, img_cols), cmap=plt.get_cmap('gray'))
    show_image(x_train[i])   
#Show the plot
    plt.show()
    plt.imshow(x_train[i].reshape(img_rows, img_cols), cmap=plt.get_cmap('gray'))
    plt.show()
    break




import pickle
# generate file name to identify the valid ids group used

if len(valid_ids[i]) == 3:
    data_name = str(driver_valid_list[0] + "_" + driver_valid_list[1] + "_" + 
                driver_valid_list[2] +  "_" +
                str(color_type) + "_" + str(img_rows) +
                "_" + str(img_cols) + ".pk1")
else:
    data_name = str(driver_valid_list[0] + "_" + driver_valid_list[1] + "_" + 
                driver_valid_list[2] +  "_" + driver_valid_list[3] + "_" +
                str(color_type) + "_" + str(img_rows) +
                "_" + str(img_cols) + ".pk1")

file_name =  str("x_train" + "_" + data_name)

with open (file_name, 'wb') as f:
    pickle.dump (x_train,f)
    
    
file_name2 =  str("y_train_xy_valid" + "_" + data_name)

with open (file_name2, 'wb') as f:
    pickle.dump (y_train, f)
    pickle.dump (x_valid, f)
    pickle.dump (y_valid, f)


