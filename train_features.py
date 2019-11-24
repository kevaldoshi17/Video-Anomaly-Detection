import sys
import os
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.callbacks import TensorBoard, ModelCheckpoint, CSVLogger, LearningRateScheduler, ReduceLROnPlateau

from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense, Activation
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers import Dense, GlobalAveragePooling2D, Dropout, MaxPooling2D, Input, AveragePooling2D, Flatten
from keras.models import Model
from keras.layers import Input,Conv2D,MaxPooling2D,UpSampling2D,Conv3D
from keras.layers import Dense, GlobalAveragePooling2D, Dropout, MaxPooling2D,MaxPool3D
from keras.optimizers import RMSprop,SGD
from keras import callbacks
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
import glob
from keras.models import load_model
from keras.regularizers import l2

from tweaked_ImageGenerator_v2 import ImageDataGenerator
import time
import numpy as np
import time
import keras



train_data_path = 'Data/'
validation_data_path = 'Test/'

"""
Parameters
"""
img_width, img_height = 120,100
batch_size = 8
samples_per_epoch = 64
pool_size = 2
classes_num = 11
lr = 0.0001


def c3d_model():
    input_shape = (16,100,120,3)
    weight_decay = 0.005
    nb_classes = 2

    inputs = Input(input_shape)
    x = Conv3D(64,(3,3,3),strides=(1,1,1),padding='same',
               activation='relu',kernel_regularizer=l2(weight_decay))(inputs)
    x = MaxPool3D((2,2,1),strides=(2,2,1),padding='same')(x)

    x = Conv3D(128,(3,3,3),strides=(1,1,1),padding='same',
               activation='relu',kernel_regularizer=l2(weight_decay))(x)
    x = MaxPool3D((2,2,2),strides=(2,2,2),padding='same')(x)

    x = Conv3D(128,(3,3,3),strides=(1,1,1),padding='same',
               activation='relu',kernel_regularizer=l2(weight_decay))(x)
    x = MaxPool3D((2,2,2),strides=(2,2,2),padding='same')(x)

    x = Conv3D(256,(3,3,3),strides=(1,1,1),padding='same',
               activation='relu',kernel_regularizer=l2(weight_decay))(x)
    x = MaxPool3D((2,2,2),strides=(2,2,2),padding='same')(x)

    x = Flatten()(x)
    x = Dense(2048,activation='relu',kernel_regularizer=l2(weight_decay))(x)
    x = Dropout(0.5)(x)
    
    x = Dense(nb_classes,kernel_regularizer=l2(weight_decay))(x)
    x = Activation('softmax')(x)

    model = Model(inputs, x)
    return model


#Define Path
# model_path = './models/model1.h5'
# model_weights_path = './models/weights1.h5'


# base_model = InceptionV3(weights='imagenet', include_top=False, input_tensor=Input(shape=(img_height, img_width, 3)))
# weight_decay = 0.005
# ######Pooling Layer
# base_model.summary()
# # base_model.layers.pop()
# # base_model.summary()

# x = base_model.output
# # x = Dropout(.5)(x)
# # x = Flatten()(x)
# # x = Dense(2048,activation='relu',kernel_regularizer=l2(weight_decay))(x)
# # x = Dropout(0.5)(x)
# # x = Dense(2048,activation='relu',kernel_regularizer=l2(weight_decay))(x)

# x = Dropout(.5)(x)

# x = Flatten()(x)
# # and a logistic layer -- let's say we have 200 classes
# predictions = Dense(1, activation='sigmoid',W_regularizer=l2(.0005))(x)




datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
# train_data=datagen.flow_from_directory('path/to/data', target_size=(x, y), batch_size=32, frames_per_step=4)

# def autoencoder(input_img):
#     #encoder
#     #input = 28 x 28 x 1 (wide and thin)
#     conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img) #28 x 28 x 32
#     pool1 = MaxPooling2D(pool_size=(2, 2))(conv1) #14 x 14 x 32
#     conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1) #14 x 14 x 64
#     pool2 = MaxPooling2D(pool_size=(2, 2))(conv2) #7 x 7 x 64
#     conv3 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool2) #7 x 7 x 128 (small and thick)

#     #decoder
#     conv4 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv3) #7 x 7 x 128
#     up1 = UpSampling2D((2,2))(conv4) # 14 x 14 x 128
#     conv5 = Conv2D(64, (3, 3), activation='relu', padding='same')(up1) # 14 x 14 x 64
#     up2 = UpSampling2D((2,2))(conv5) # 28 x 28 x 64
#     decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(up2) # 28 x 28 x 1
#     return decoded

# input_img = Input(shape = (240, 320, 3))
# autoencoder = Model(input_img, autoencoder(input_img))
# autoencoder.compile(loss='mean_squared_error', optimizer = RMSprop())
# autoencoder.summary()


log_dir = './tf-log/'
tb_cb = callbacks.TensorBoard(log_dir=log_dir, histogram_freq=0)
csv_logger = CSVLogger('model4.log')
cbks = [tb_cb,csv_logger]


# model = Model(inputs=base_model.input, outputs=predictions)

# for layer in base_model.layers:
#   layer.trainable = False


# model.compile(loss='binary_crossentropy',
#               optimizer=optimizers.SGD(lr=lr,momentum=0.9),
#               metrics=['accuracy'])


# train_datagen = ImageDataGenerator,rescale=1. / 255)

# test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = datagen.flow_from_directory(train_data_path,target_size=(img_height, img_width),batch_size=1, frames_per_step=16)
model = c3d_model()
lr = 0.005
sgd = SGD(lr=lr, momentum=0.9, nesterov=True)
model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
model.summary()


model.fit_generator(train_generator,steps_per_epoch = 100,epochs=10,callbacks=cbks,verbose=1)


target_dir = './models/'
if not os.path.exists(target_dir):
  os.mkdir(target_dir)
model.save('./models/model9.h5')
model.save_weights('./models/weights9.h5')