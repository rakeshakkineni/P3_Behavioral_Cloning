from keras.models import Sequential, Model
from keras.layers import Cropping2D, Lambda, Dropout
from keras.layers.convolutional import Conv2D
from keras.layers.core import Flatten, Dense
import os
import csv

# Read the driving_log.csv file and append the read strings to to samples
samples = []
with open('./Merged_Data_CAnti_Clock/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)
        
# Split the training and validation samples from the actual data
from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

import cv2
import numpy as np
import sklearn
from sklearn.utils import shuffle  

#******************************************************************************
# The function generator shall perform the following actions
# 1. Shuffle the input sample strings 
# 2. Load the Center , Left and Right images and steering angle from the path 
#    given in samples list
# 3. Left and Right Steering angle is extracted by adding and substracting a 
#    correction factor of 0.2
# 4. Append the images and steering angle , shuffle and return.

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            correction = 0.2    
            images = []
            angles = []
            for batch_sample in batch_samples:
                name = './Merged_Data_CAnti_Clock/IMG/'
                # Center Images
                center_image = cv2.imread(name+batch_sample[0].split('/')[-1])
                center_angle = float(batch_sample[3])
                # Left Images
                left_angle = center_angle + correction
                left_image = cv2.imread(name+batch_sample[1].split('/')[-1])
                # Left Images
                right_angle = center_angle - correction
                right_image = cv2.imread(name+batch_sample[2].split('/')[-1])
                
                images.extend([center_image, left_image, right_image] )
                angles.extend([center_angle, left_angle, right_angle])

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            #print(np.shape(X_train[0]))
            yield shuffle(X_train, y_train)

# train and validation generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

ch, row, col = 3, 160, 320  # UnTrimmed image format

#*****************************Model Definition***********************************
model = Sequential()

# Preprocess incoming data, Normalize the images to have a value in range -1 to 1
model.add(Lambda(lambda x: x/127.5 - 1.,
        input_shape=( row, col,ch),
        output_shape=( row, col, ch)))
# Cropp Sky on the top and Car bonnet from the bottom of the images. 
model.add(Cropping2D(cropping=((70,25), (0,0)), input_shape=(160,320,3))) # Cropp the unwanted images
# Convolution Layer
model.add(Conv2D(24,(5,5),strides=(2,2),activation="relu")) # Output Dim: 31 x 158 x 24
model.add(Conv2D(36,(5,5),strides=(2,2),activation="relu")) # Output Dim: 14 x 77 x 36
model.add(Conv2D(48,(5,5),strides=(2,2),activation="relu")) # Output Dim: 5 x 37 x 48
model.add(Conv2D(64,(3,3),activation="relu"))               # Output Dim: 3 x 35 x 64
model.add(Conv2D(64,(3,3),activation="relu"))               # Output Dim: 1 x 33 x 64
#Fully Connected Layer 
model.add(Flatten())                                        # Output Dim: 2112
model.add(Dense(100))                                       # Output Dim: 100
model.add(Dense(50))                                        # Output Dim: 50
model.add(Dropout(0.1))                                     # Output Dim: 50
model.add(Dense(1))                                         # Output Dim: 1

model.summary()                                             # Print the model summary 
#compile and train the model
model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_generator,samples_per_epoch= len(train_samples),validation_data=validation_generator,nb_val_samples=len(validation_samples),nb_epoch=3)
#Save the results of the model
model.save('model.h5')
