

import os
import zipfile
import random
import tensorflow as tf
import shutil
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from shutil import copyfile

from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.applications.inception_v3 import InceptionV3

#Training_dir = os.path.join("F:/BE Project/dataset2/asl_alphabet_train/Training")

Training_dir = os.path.join("F:/BE Project/ourdataset/6letter/train")

train_datagen = ImageDataGenerator(rescale = 1./255.,
                                   horizontal_flip = True,
                                   width_shift_range = 0.2,
                                   height_shift_range = 0.2,
                                   rotation_range = 20,
                                   zoom_range = 0.15,
                                   brightness_range = [-0.5,2],
                                   fill_mode = 'nearest')

train_generator = train_datagen.flow_from_directory(Training_dir,
                                                    target_size = (150, 150),
                                                    batch_size = 64,
                                                    class_mode = 'categorical')


Validation_dir = os.path.join("F:/BE Project/ourdataset/6letter/test")

valid_datagen = ImageDataGenerator(rescale = 1./255.,
                                   horizontal_flip = True,
                                   zoom_range = 0.15,
                                   brightness_range = [-0.5,2],
                                   fill_mode = 'nearest')

valid_generator = valid_datagen.flow_from_directory(Validation_dir,
                                                    target_size = (150, 150),
                                                    batch_size = 64,
                                                    class_mode = 'categorical')


#load pretrained InceptionV3 module
local_weights_file = "F:/BE Project/dataset2/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5"

pretrained_model = InceptionV3(input_shape = (150,150,3),
                               include_top = False,
                               weights = None)

pretrained_model.load_weights(local_weights_file)

for layer in pretrained_model.layers:
    layer.trainable = False
  
#choose your own last layer from inceptionv3    
last_layer = pretrained_model.get_layer('mixed7')
last_output = last_layer.output

#build a new model on top of this
x = layers.Flatten()(last_output)
x = layers.Dense(256, activation = 'relu')(x)

#add dropout layer to reduce overfitting
x = layers.Dropout(0.25)(x)
x = layers.Dense(128, activation = 'relu')(x)

#add dropout layer to reduce overfitting
x = layers.Dropout(0.35)(x)
x = layers.Dense(6, activation = 'softmax')(x)

model = Model(pretrained_model.input, x)
model.compile(optimizer = RMSprop(lr = 0.0001),
              loss = 'categorical_crossentropy',
              metrics = ['acc'])


history = model.fit_generator(train_generator,
                              epochs=50,
                              steps_per_epoch = 58,
                              verbose=1,
                              validation_data=valid_generator,
                              validation_steps = 24)


model.save('letter6_version2_20epochs.h5')

import matplotlib.pyplot as plt
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']


epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()

plt.plot(epochs, loss, 'r', label='Training Loss')
plt.plot(epochs, val_loss, 'b', label='Validation Loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()

'''
import cv2
import skimage
from skimage.transform import resize
import numpy as np

a = cv2.imread("F:/BE Project/kaggle image dataset/images_custom/I.jpeg")
#print (a)

img = skimage.transform.resize(a, (150, 150, 3), mode="constant")
img_arr = np.asarray(img)
new_model = tf.keras.models.load_model("newmodel.model")
#print (type(new_model))

prediction = new_model.predict([[img_arr]])


maxElement = np.amax(prediction)
#print (maxElement)


# Get the indices of maximum element in numpy array
result = np.where(prediction == np.amax(prediction))
 
#print('Returned tuple of arrays :', result)
print('List of Indices of maximum element :', result[1])
'''