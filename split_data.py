# -*- coding: utf-8 -*-

import os
import zipfile
import random
import tensorflow as tf
import shutil
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from shutil import copyfile
from os import getcwd
##import split_folders

'''
#make destination folders
try:
 
    os.mkdir('F:/BE Project/dataset2/asl_alphabet_train/Training')
    os.mkdir('F:/BE Project/dataset2/asl_alphabet_train/Testing')
    os.mkdir('F:/BE Project/dataset2/asl_alphabet_train/Training/A')
    os.mkdir('F:/BE Project/dataset2/asl_alphabet_train/Training/B')
    os.mkdir('F:/BE Project/dataset2/asl_alphabet_train/Testing/A')
    os.mkdir('F:/BE Project/dataset2/asl_alphabet_train/Testing/B')
    
except OSError:
    pass
'''

#split_folders.ratio('F:/BE Project/dataset2/dataset/input/', output="F:/BE Project/dataset2/dataset/output/", seed=1337, ratio=(.8, .2))



#randomly split data into training and validation sets
def split_data(SOURCE, TRAINING, TESTING, SPLIT_SIZE):

    dataset = []
     
    for unitData in os.listdir(SOURCE):
        data = SOURCE + unitData        
        dataset.append(unitData)


    train_len = int(len(dataset) * SPLIT_SIZE)
    test_len = int(len(dataset) - train_len)


    shuffle = random.sample(dataset, len(dataset))
    train_set = dataset[0:train_len]
    test_set = dataset[-test_len:]

    for unitData in train_set:
        temp_train_set = SOURCE + unitData
        final_train_set = TRAINING + unitData
        copyfile(temp_train_set, final_train_set)
    
    for unitData in test_set:
        temp_test_set = SOURCE + unitData
        final_test_set = TESTING + unitData
        copyfile(temp_test_set, final_test_set)





rop_SOURCE_DIR = 'D:/MyprojectROP/Mansi/source_train/ROP/'
TRAINING_rop_DIR = 'D:/MyprojectROP/Mansi/train/ROP'
TESTING_rop_DIR = 'D:/MyprojectROP/Mansi/test/ROP'

norop_SOURCE_DIR = 'D:/MyprojectROP/Mansi/source_train/No_ROP/'
TRAINING_norop_DIR = 'D:/MyprojectROP/Mansi/train/No_ROP'
TESTING_norop_DIR = 'D:/MyprojectROP/Mansi/test/No_ROP'

'''
I_SOURCE_DIR = 'F:/BE Project/ourdataset/source/W/'
TRAINING_I_DIR = 'F:/BE Project/ourdataset/train/W/'
TESTING_I_DIR = 'F:/BE Project/ourdataset/test/W/'


X_SOURCE_DIR = 'F:/BE Project/ourdataset/source/X/'
TRAINING_X_DIR = 'F:/BE Project/ourdataset/train/X/'
TESTING_X_DIR = 'F:/BE Project/ourdataset/test/X/'

U_SOURCE_DIR = 'F:/BE Project/ourdataset/source/U/'
TRAINING_U_DIR = 'F:/BE Project/ourdataset/train/U/'
TESTING_U_DIR = 'F:/BE Project/ourdataset/test/U/'

'''


split_size = .75


split_data(rop_SOURCE_DIR, TRAINING_rop_DIR, TESTING_rop_DIR, split_size)
split_data(norop_SOURCE_DIR, TRAINING_norop_DIR, TESTING_norop_DIR, split_size)

'''
split_data(X_SOURCE_DIR, TRAINING_X_DIR, TESTING_X_DIR, split_size)
split_data(U_SOURCE_DIR, TRAINING_U_DIR, TESTING_U_DIR, split_size)
split_data(O_SOURCE_DIR, TRAINING_O_DIR, TESTING_O_DIR, split_size)
split_data(Q_SOURCE_DIR, TRAINING_Q_DIR, TESTING_Q_DIR, split_size)
split_data(Z_SOURCE_DIR, TRAINING_Z_DIR, TESTING_Z_DIR, split_size)
split_data(Y_SOURCE_DIR, TRAINING_Y_DIR, TESTING_Y_DIR, split_size)
'''



#print(len(os.listdir('F:/BE Project/dataset2/asl_alphabet_train/Training/B/')))
#print(len(os.listdir('F:/BE Project/dataset2/asl_alphabet_train/Testing/B/')))