import os
os.environ['KERAS_BACKEND'] = 'theano'

from keras.models import Sequential,Model
from keras.layers import Dense, Dropout, Activation
from keras.regularizers import l2
from keras.optimizers import SGD ,Adagrad
from scipy.io import loadmat, savemat
from keras.models import model_from_json
import theano.tensor as T
import theano
import csv
import configparser
import collections
import time
import csv
import os
from os import listdir
import skimage.transform
from skimage import color
from os.path import isfile, join
import numpy as np
import numpy
from datetime import datetime
from scipy.spatial.distance import cdist,pdist,squareform
import theano.sandbox
import shutil
# import tensorflow as tf
# theano.sandbox.cuda.use('gpu0')




seed = 7
numpy.random.seed(seed)


def load_model(json_path):  # Function to load the model
    model = model_from_json(open(json_path).read())
    return model

def load_weights(model, weight_path):  # Function to load the model weights
    dict2 = loadmat(weight_path)
    dict = conv_dict(dict2)
    i = 0
    for layer in model.layers:
        weights = dict[str(i)]
        layer.set_weights(weights)
        i += 1
    return model

def conv_dict(dict2):
    i = 0
    dict = {}
    for i in range(len(dict2)):
        if str(i) in dict2:
            if dict2[str(i)].shape == (0, 0):
                dict[str(i)] = dict2[str(i)]
            else:
                weights = dict2[str(i)][0]
                weights2 = []
                for weight in weights:
                    if weight.shape in [(1, x) for x in range(0, 5000)]:
                        weights2.append(weight[0])
                    else:
                        weights2.append(weight)
                dict[str(i)] = weights2
    return dict

# Load Video

def load_dataset_One_Video_Features(Test_Video_Path):
    VideoPath =Test_Video_Path
    f = open(VideoPath, "r")
    words = f.read().split()
    num_feat = int(len(words) / 4096)
    count = -1;
    VideoFeatues = []
    for feat in range(0, num_feat):
        feat_row1 = np.float32(words[feat * 4096:feat * 4096 + 4096])
        count = count + 1
        if count == 0:
            VideoFeatues = feat_row1
        if count > 0:
            VideoFeatues = np.vstack((VideoFeatues, feat_row1))
    AllFeatures = VideoFeatues
    return  AllFeatures



print("Starting testing...")


AllTest_Video_Path = 'Test_Data/'
# AllTest_Video_Path contains C3D features (txt file)  of each video. Each file contains 32 features, each of 4096 dimensions.
Results_Path = '../Eval_Res_Test/'
# Results_Path is the folder where you can save your results
Model_dir='../'
# Model_dir is the folder where we have placed our trained weights
weights_path =  'weightsAnomalyL1L2_5000.mat'
# weights_path is Trained model weights

model_path = 'model.json'

if not os.path.exists(Results_Path):
       os.makedirs(Results_Path)

All_Test_files= listdir(AllTest_Video_Path)
All_Test_files.sort()

model=load_model(model_path)
print("Step 1 Doneeeeeeeeeeeeeeee")
load_weights(model, weights_path)

model.layers.pop()
x = model.layers[-2].output
# x = Dropout(.5)(x)
# x = Flatten()(x)
# x = Dense(2048)(x)
# x = Dropout(0.5)(x)
# x = Dense(2048,activation='relu',kernel_regularizer=l2(weight_decay))(x)
model = Model(inputs=model.input, outputs=x)
model.summary()


nVideos=len(All_Test_files)
time_before = datetime.now()
Test_preds = list()
indexes = list()
for iv in range(nVideos):
    Test_Video_Path = os.path.join(AllTest_Video_Path, All_Test_files[iv])
    inputs=load_dataset_One_Video_Features(Test_Video_Path) # 32 segments features for one testing video
    predictions = model.predict_on_batch(inputs)   # Get Test prediction for each of 32 video segments.
    aa=All_Test_files[iv]
    print(aa)
    aa=aa[0:-4]
    print(aa)
    A_predictions_path = savemat(Results_Path + aa + '.mat',{'prediction' : predictions})  # Save array of 1*32, containing Test score for each segment. Please see Evaluate Test Detector to compute  ROC.
    print ("Total Time took: " + str(datetime.now() - time_before))
    Test_preds.append(predictions)
    indexes.append(aa)
Test_preds = np.array(Test_preds)
np.save("Test_preds",Test_preds)
np.save("indexes",indexes)
