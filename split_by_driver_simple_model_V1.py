# -*- coding: utf-8 -*-
"""
Created on Sun Jan 21 12:09:02 2018

@author: fcp018
"""

# Libraries import
import numpy as np
import pandas as pd
import os
import sys
import matplotlib.pyplot as plt
import cv2
import glob
import cv2

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.models import model_from_json
from sklearn.metrics import log_loss
from keras import backend as K
from keras.utils import np_utils


# Quick list of all the location of raw data
# "imgs" is the main folder for all of raw data
# "imgs_sub" is the folder for the 500 images subset with 50 images per class
# "imgs_sub2" is the folder for the 500 images subset but 
# gathered in only two classes (0= safe (50 images), 1= distracted(450 images))

image_drv = './imgs_sub/'
train_drv = image_drv + "train/"
test_drv = image_drv + "test/"

# For driver classification list, add "_sub" or "_sub2 at the end 
# depending on the folder used
driver_imgs_list = "./driver_imgs_list_sub.csv"  

# Define some image and class parameters
# color type: 1 = grey, 3 = rgb
color_type = 1 
img_rows, img_cols = 96, 128
nb_classes = 10  # 2 for "imgs_sub2"; 10 for others


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


# Function to split drivers into training and validation set.
# Note the split is by driver ids. Same driver not in both sets.

def train_valid_driver_split(train_driver_list, split = 1.0):
    driver_valid_list = []
    
    # Chose random # of drivers depending on "split" % provided
    #np.random.seed(123)  # set seed could be used here to duplicate results.
    
    train_split = list(np.random.choice(train_driver_list, int(len(train_driver_list)*split), replace = False))
    # Take the remaining drivers into the validation list
    valid_split = [ driver for driver in train_driver_list if driver not in train_split]
    return train_split, valid_split

# Create the training and validation splits for driver ids
driver_train_list, driver_valid_list = train_valid_driver_split(driver_ids, split = 0.9)
print(driver_train_list)
print(driver_valid_list)


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



# create train and validation sets for images and class labels.
print("Creating training set:")
x_train, y_train = get_train_validation_data(train_driver_list, driver_train_list)
print("Creating validation data:")
x_valid, y_valid = get_train_validation_data(train_driver_list, driver_valid_list)

# reshape images for the convolution layers
x_train = x_train.reshape(x_train.shape[0], color_type, img_rows, img_cols)
x_valid = x_valid.reshape(x_valid.shape[0], color_type, img_rows, img_cols)

# create a simple convolution model with keras
# Set model batch size and epoch parameters.
batch_size = 50
nb_epoch = 10

# number of convolutional filters to use
nb_filters = 32
# size of pooling area for max pooling
nb_pool = 2
# convolution kernel size
nb_conv = 3

# read saved model, if any
model_from_cache = 0
if model_from_cache == 1:
    model = read_model()
    model.compile(loss='categorical_crossentropy', optimizer='adadelta')
else:
    model = Sequential()
    
    # first convolution layer
    model.add(Convolution2D(nb_filters, (nb_conv, nb_conv),
                            border_mode='valid',
                            input_shape=(1, img_rows, img_cols), data_format = 'channels_first'))
    model.add(Activation('relu'))
    
    # 2nd cn layer
    model.add(Convolution2D(nb_filters, nb_conv, nb_conv))
    model.add(Activation('relu'))
    # max pooling and dropout
    model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
    model.add(Dropout(0.25))
    # flatten and dropout layers
    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    # fully connected layer with softmax for probablities
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))
    
    # compile and train model
    model.compile(loss='categorical_crossentropy', optimizer='adadelta',metrics=['accuracy'])

    model.fit(x_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch,
              verbose=1, validation_data=(x_valid, y_valid))
 
# Score model on train and validation sets.
train_score = model.evaluate(x_train, y_train, verbose=0)
print('train score: ', train_score)

valid_score = model.evaluate(x_valid, y_valid, verbose=0)
print('valid score: ', valid_score)

