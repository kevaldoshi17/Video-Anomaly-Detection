import os
import numpy as np
import tensorflow as tf
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
import glob
from keras.models import load_model
import time
from keras.callbacks import TensorBoard, ModelCheckpoint, CSVLogger, LearningRateScheduler, ReduceLROnPlateau

from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense, Activation
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers import Dense, GlobalAveragePooling2D, Dropout, MaxPooling2D, Input, AveragePooling2D, Flatten
from keras.models import Model

from keras.layers import Dense, GlobalAveragePooling2D, Dropout, MaxPooling2D
from keras.models import load_model
from keras.regularizers import l2
import keras

# config = tf.ConfigProto( device_count = {'GPU': 2 , 'CPU': 12} ) 
# sess = tf.Session(config=config) 
# keras.backend.set_session(sess)

# from tensorflow.python.client import device_lib
# print(device_lib.list_local_devices())


# from keras import backend as K
# K.tensorflow_backend._get_available_gpus()

start = time.time()

#Define Path
model_path = './models/model9.h5'
model_weights_path = './models/weights9.h5'
test_path = 'Data/Test/'

#Load the pre-trained models
model = load_model(model_path)
model.load_weights(model_weights_path)
model.layers.pop()
x = model.layers[-2].output
# x = Dropout(.5)(x)
# x = Flatten()(x)
# x = Dense(2048)(x)
# x = Dropout(0.5)(x)
# x = Dense(2048,activation='relu',kernel_regularizer=l2(weight_decay))(x)
model = Model(inputs=model.input, outputs=x)
model.summary()

#Define image parameters
img_width, img_height = 120, 100
correct = 0
total = 0
#Prediction Function
def predict(file):
  # test_paths = os.listdir(file)
  One_Class = list()
  for t in file:
    x = load_img('Data/Test/'+t, target_size=(img_width,img_height))
    x = img_to_array(x)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    One_Class.append(x)

  One_Class = np.array(One_Class)
  One_Class = One_Class.reshape(1,16,100,120,3)
  array = model.predict(One_Class)
  # print(array)
  #rint(result)
  # result = np.argmax(result)
  #print(answer)
  return array

#Walk the directory for every image
# test_datagen = ImageDataGenerator(rescale=1. / 255)

test_paths = os.listdir(test_path)

batch_size = 32
Classes = list()
for f in range(0,len(test_paths)-16,16):
  print(f)
  One_Class = predict(test_paths[f:f+16])
  print(np.size(One_Class))
  # One_Class = np.c_[One_Class,np.zeros(One_Class.shape[0])+int(f)]
  Classes.append(One_Class)

# test_generator = test_datagen.flow_from_directory(test_path,target_size=(img_height, img_width),batch_size=batch_size,class_mode='categorical')




# for f in test_paths:
#   print(f)
#   One_Class = list()
#   if os.path.isdir(test_path+f
#     ):
#     One_Class = model.predict()
  # One_Class = np.c_[One_Class,np.zeros(One_Class.shape[0])+int(f)]

  # Classes.append(One_Class)





Features = np.array(Classes)
np.save('Features_Test.npy',Features)    
    # print(np.size(ind),ind,time.time()-start)

# print(correct,total,np.divide(correct,total))
