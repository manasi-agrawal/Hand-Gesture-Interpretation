

import tensorflow.keras.backend as K
from tensorflow.keras.layers import (Input, Conv2D, MaxPooling2D, Flatten, 
                                     Dense, Dropout, BatchNormalization)
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers

import os
import zipfile
import random
import tensorflow as tf
import shutil
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from shutil import copyfile

gpu_options = tf.GPUOptions(allow_growth=True)
session = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options))

Training_dir = os.path.join("F:/BE Project/ourdataset/train")

train_datagen = ImageDataGenerator(rescale = 1./255.,
                                   horizontal_flip = True,
                                   width_shift_range = 0.2,
                                   height_shift_range = 0.2,
                                   rotation_range = 20,
                                   zoom_range = 0.15,
                                   brightness_range = [-1,2],
                                   fill_mode = 'nearest')

train_generator = train_datagen.flow_from_directory(Training_dir,
                                                    target_size = (150, 150),
                                                    batch_size = 128,
                                                    class_mode = 'categorical')


Validation_dir = os.path.join("F:/BE Project/ourdataset/test")

valid_datagen = ImageDataGenerator(rescale = 1./255.,
                                   brightness_range = [-1,2],
                                   fill_mode = 'nearest')

valid_generator = valid_datagen.flow_from_directory(Validation_dir,
                                                    target_size = (150, 150),
                                                    batch_size = 128,
                                                    class_mode = 'categorical')


model = Sequential([
        Conv2D(64, kernel_size=(3, 3), 
               activation='relu',
               padding='same', 
               input_shape=(150, 150, 3)),      
        BatchNormalization(),
        
        Conv2D(64, kernel_size=(3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),

        Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(128, kernel_size=(3, 3), activation='relu'),
        BatchNormalization(),        
        MaxPooling2D(pool_size=(2, 2)),
        
        Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(256, kernel_size=(3, 3), activation='relu'),
        BatchNormalization(),    
        MaxPooling2D(pool_size=(2, 2)),   
        
        Flatten(),
        
        Dense(512, activation='relu'),
        
        Dense(512, activation='relu'),
        
        Dense(16, activation='softmax')
    ])
        
adam = Adam(lr=0.0001, decay=1e-6)
   
model.compile(optimizer=adam,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

history = model.fit_generator(train_generator,
                              epochs=25,
                              steps_per_epoch = 77,
                              verbose=1,
                              validation_data=valid_generator,
                              validation_steps = 30)


model.save('ver3model.model')

session.close()





