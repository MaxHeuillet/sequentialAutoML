###########################
###########################
#
#
# this file is the DGX pipeline that was used to generate the pipelines benchmarking
# The output of this pipeline is a 175 (pipelines) x 666 (datasets) performance matrix expressed in PRAUC
#
###########################
###########################

import click
from pathlib import Path

import gzip
import pickle as pkl
import pandas as pd
import numpy as np
import random
import scipy
from scipy import stats
import os
from os import listdir, path
from os.path import isfile, join
import time


from sklearn import linear_model
from sklearn import utils
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model
from sklearn.metrics import precision_recall_curve, auc
from sklearn.metrics import roc_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import RadiusNeighborsClassifier
from sklearn import ensemble

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense,Dropout,BatchNormalization, LeakyReLU,Conv1D, MaxPooling1D, Flatten, Input, GlobalAveragePooling1D
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import losses
from tensorflow.keras.layers import LSTM

from sktime.utils.load_data import load_from_tsfile_to_dataframe
from sktime.classification.distance_based import KNeighborsTimeSeriesClassifier
from sktime.classification.compose import TimeSeriesForestClassifier

from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

import signal, time, random

def reshape_folder(X):
    
    ## reshape the time series dataset to the right format
    ## (nb.observation, nb.timesteps, nb. features (read. dimensions) = 1 )
    ## input and outut = X = a dataset
    
    X = np.array(X)
    random.shuffle(X)
    timesteps = len(X[0])
    n_features = 1 # we only use uni-dimensional time-series datasets
    X = X.reshape( (X.shape[0], timesteps, n_features) )
    return X

def compute_rocauc(pred, gt):    
    #compute ROCAUC performance give the ground truth (gt) and the prediction of a pipeline (pred)
    fpr, tpr, thresh = roc_curve(gt, pred)
    rocauc = auc(fpr, tpr)
    return rocauc #performance metric

def compute_prauc(pred, gt):
    #compute PRAUC performance give the ground truth (gt) and the prediction of a pipeline (pred)
    prec, recall, thresh = precision_recall_curve(gt, pred)
    prauc = auc(recall, prec)
    return prauc #performance metric

def build_MLP(X, fold1, fold2): 
    ## build a MLP (multi-layer-perceptron) classifier
    ## the firt layer has fold1 nodes and layer2 has fold2 nodes
    ## input X = a dataset because we need some information in the input layer (nb features)
    model = Sequential()
    model.add( Dense(fold1, input_shape=( len(X.columns), ), activation='relu') )
    model.add( Dense(fold2, activation='relu') )
    model.add( Dense(1, activation='sigmoid') )
    opt = tf.keras.optimizers.Adam(lr=0.01, clipnorm=1)
    model.compile(loss='binary_crossentropy', optimizer=opt)
    return model #keras model object

def get_models(X):
    
    ## this function contains all the models included in the benchamrking
    ## are included Gradient Bossting Classfier, Supper Vector Classifiers....
    
    ## input = X = a dataset to have some information about the number of features to build the MLP (input-layer)

    
    ## the output = a dictionnary that contains all the models such as:
    ## key:model where key gives information on which pre-processing operations should be used
    ## the first element of the key indicates which pre-processing operation should be applied
    
    ## the exhaustive list is available in Appendix.
    ## the keyword 'NO' means no pre-processing
    ## ----------- 'RUS' corresponds to random under-sampling preprocessing operation
    ## ----------- 'ROS' corresponds to random over-sampling preprocessing operation
    ## ----------- 'WG' corresponds to using a cost sensitive loss function (weight of the class)
    ## the rest is a unique identifier for each model

    
    
    NO_GBa = ensemble.GradientBoostingClassifier(n_estimators = 100, learning_rate = 0.1, subsample = 0.5, max_depth = 3)
    NO_GBb = ensemble.GradientBoostingClassifier(n_estimators = 100, learning_rate = 0.1, subsample = 0.7, max_depth = 3)
    NO_GBc = ensemble.GradientBoostingClassifier(n_estimators = 100, learning_rate = 0.1, subsample = 0.9, max_depth = 3)
    NO_GBd = ensemble.GradientBoostingClassifier(n_estimators = 100, learning_rate = 0.1, subsample = 0.5, max_depth = 4)
    NO_GBe = ensemble.GradientBoostingClassifier(n_estimators = 100, learning_rate = 0.1, subsample = 0.7, max_depth = 4)
    NO_GBf = ensemble.GradientBoostingClassifier(n_estimators = 100, learning_rate = 0.1, subsample = 0.9, max_depth = 4)
    NO_GBg = ensemble.GradientBoostingClassifier(n_estimators = 100, learning_rate = 0.1, subsample = 0.5, max_depth = 5)
    NO_GBh = ensemble.GradientBoostingClassifier(n_estimators = 100, learning_rate = 0.1, subsample = 0.7, max_depth = 5)
    NO_GBi = ensemble.GradientBoostingClassifier(n_estimators = 100, learning_rate = 0.1, subsample = 0.9, max_depth = 5)
    
    NO_SVM1a = SVC(C = 0.01, kernel = 'linear', gamma='auto', probability = True)
    NO_SVM1b = SVC(C = 1, kernel = 'linear', gamma='auto', probability = True)
    NO_SVM1c = SVC(C = 10, kernel = 'linear', gamma='auto', probability = True)
    NO_SVM1d = SVC(C = 1000, kernel = 'linear', gamma='auto', probability = True)
    
    NO_SVM2a = SVC(C = 0.01, kernel = 'rbf', gamma='auto', probability = True)
    NO_SVM2b = SVC(C = 1, kernel = 'rbf', gamma='auto', probability = True)
    NO_SVM2c = SVC(C = 10, kernel = 'rbf', gamma='auto', probability = True)
    NO_SVM2d = SVC(C = 100, kernel = 'rbf', gamma='auto', probability = True)
    NO_SVM2e = SVC(C = 1000, kernel = 'rbf', gamma='auto', probability = True)
    NO_SVM2f = SVC(C = 1500, kernel = 'rbf', gamma='auto', probability = True)
    NO_SVM2g = SVC(C = 10000, kernel = 'rbf', gamma='auto', probability = True)
    
    WG_SVM1a = SVC(C = 0.01, kernel = 'linear', class_weight='balanced', gamma='auto', probability = True)
    WG_SVM1b = SVC(C = 1, kernel = 'linear', class_weight='balanced', gamma='auto', probability = True)
    WG_SVM1c = SVC(C = 10, kernel = 'linear', class_weight='balanced', gamma='auto', probability = True)
    WG_SVM1d = SVC(C = 1000, kernel = 'linear', class_weight='balanced', gamma='auto', probability = True)
    
    WG_SVM2a = SVC(C = 0.01, kernel = 'rbf', class_weight='balanced', gamma='auto', probability = True)
    WG_SVM2b = SVC(C = 1, kernel = 'rbf', class_weight='balanced', gamma='auto', probability = True)
    WG_SVM2c = SVC(C = 10, kernel = 'rbf', class_weight='balanced', gamma='auto', probability = True)
    WG_SVM2d = SVC(C = 100, kernel = 'rbf', gamma='auto', class_weight='balanced', probability = True)
    WG_SVM2e = SVC(C = 1000, kernel = 'rbf', gamma='auto', class_weight='balanced', probability = True)
    WG_SVM2f = SVC(C = 1500, kernel = 'rbf', gamma='auto', class_weight='balanced', probability = True)
    WG_SVM2g = SVC(C = 10000, kernel = 'rbf', gamma='auto', class_weight='balanced', probability = True)

    NO_KNN1 = KNeighborsClassifier(n_neighbors=2, weights = 'uniform', leaf_size = 3)
    NO_KNN2 = KNeighborsClassifier(n_neighbors=2, weights = 'uniform', leaf_size = 10)
    NO_KNN3 = KNeighborsClassifier(n_neighbors=2, weights = 'uniform', leaf_size = 50)
    NO_KNN4 = KNeighborsClassifier(n_neighbors=2, weights = 'uniform', leaf_size = 100)
    
    WG_KNN1 = KNeighborsClassifier(n_neighbors=2, weights = 'distance', leaf_size = 3)
    WG_KNN2 = KNeighborsClassifier(n_neighbors=2, weights = 'distance', leaf_size = 10)
    WG_KNN3 = KNeighborsClassifier(n_neighbors=2, weights = 'distance', leaf_size = 50)
    WG_KNN4 = KNeighborsClassifier(n_neighbors=2, weights = 'distance', leaf_size = 100)
    
    NO_RKNN1 = RadiusNeighborsClassifier(radius = 0.8, weights = 'uniform', leaf_size = 10, outlier_label = 'most_frequent' )
    NO_RKNN2 = RadiusNeighborsClassifier(radius = 0.8, weights = 'uniform', leaf_size = 30, outlier_label = 'most_frequent')
    NO_RKNN3 = RadiusNeighborsClassifier(radius = 0.8, weights = 'uniform', leaf_size = 50, outlier_label = 'most_frequent')
    NO_RKNN4 = RadiusNeighborsClassifier(radius = 1, weights = 'uniform', leaf_size = 10, outlier_label = 'most_frequent')
    NO_RKNN5 = RadiusNeighborsClassifier(radius = 1, weights = 'uniform', leaf_size = 30, outlier_label = 'most_frequent')
    NO_RKNN6 = RadiusNeighborsClassifier(radius = 1, weights = 'uniform', leaf_size = 50, outlier_label = 'most_frequent')
    NO_RKNN7 = RadiusNeighborsClassifier(radius = 1.2, weights = 'uniform', leaf_size = 10, outlier_label = 'most_frequent')
    NO_RKNN8 = RadiusNeighborsClassifier(radius = 1.2, weights = 'uniform', leaf_size = 30, outlier_label = 'most_frequent')
    NO_RKNN9 = RadiusNeighborsClassifier(radius = 1.2, weights = 'uniform', leaf_size = 50, outlier_label = 'most_frequent')
    
    WG_RKNN1 = RadiusNeighborsClassifier(radius = 0.8, weights = 'distance', leaf_size = 10, outlier_label = 'most_frequent')
    WG_RKNN2 = RadiusNeighborsClassifier(radius = 0.8, weights = 'distance', leaf_size = 30, outlier_label = 'most_frequent')
    WG_RKNN3 = RadiusNeighborsClassifier(radius = 0.8, weights = 'distance', leaf_size = 50, outlier_label = 'most_frequent')
    WG_RKNN4 = RadiusNeighborsClassifier(radius = 1, weights = 'distance', leaf_size = 10, outlier_label = 'most_frequent')
    WG_RKNN5 = RadiusNeighborsClassifier(radius = 1, weights = 'distance', leaf_size = 30, outlier_label = 'most_frequent')
    WG_RKNN6 = RadiusNeighborsClassifier(radius = 1, weights = 'distance', leaf_size = 50, outlier_label = 'most_frequent')
    WG_RKNN7 = RadiusNeighborsClassifier(radius = 1.2, weights = 'distance', leaf_size = 10, outlier_label = 'most_frequent')
    WG_RKNN8 = RadiusNeighborsClassifier(radius = 1.2, weights = 'distance', leaf_size = 30, outlier_label = 'most_frequent')
    WG_RKNN9 = RadiusNeighborsClassifier(radius = 1.2, weights = 'distance', leaf_size = 50, outlier_label = 'most_frequent')
                                                                                                         
    NO_RF1 = RandomForestClassifier(n_estimators=100, max_depth=3)
    NO_RF2 = RandomForestClassifier(n_estimators=100, max_depth=5)
    NO_RF3 = RandomForestClassifier(n_estimators=100, max_depth=10)
    NO_RF4 = RandomForestClassifier(n_estimators=100, max_depth=50)
    NO_RF5 = RandomForestClassifier(n_estimators=100, max_depth=100)
    
    WG_RF1 = RandomForestClassifier(n_estimators=100, max_depth=3, class_weight = 'balanced_subsample')
    WG_RF2 = RandomForestClassifier(n_estimators=100, max_depth=5, class_weight = 'balanced_subsample')
    WG_RF3 = RandomForestClassifier(n_estimators=100, max_depth=10, class_weight = 'balanced_subsample')
    WG_RF4 = RandomForestClassifier(n_estimators=100, max_depth=50, class_weight = 'balanced_subsample')
    WG_RF5 = RandomForestClassifier(n_estimators=100, max_depth=100, class_weight = 'balanced_subsample')

    NO_SGDCL11 = SGDClassifier(loss="log", penalty="l1", alpha = 0.0001)
    NO_SGDCL12 = SGDClassifier(loss="log", penalty="l1", alpha = 0.01)
    NO_SGDCL13 = SGDClassifier(loss="log", penalty="l1", alpha = 0.1)
    NO_SGDCL14 = SGDClassifier(loss="log", penalty="l1", alpha = 0.5)
    NO_SGDCL15 = SGDClassifier(loss="log", penalty="l1", alpha = 0.8) 
    
    WG_SGDCL11 = SGDClassifier(loss="log", penalty="l1", alpha = 0.0001, class_weight='balanced')
    WG_SGDCL12 = SGDClassifier(loss="log", penalty="l1", alpha = 0.01, class_weight='balanced')
    WG_SGDCL13 = SGDClassifier(loss="log", penalty="l1", alpha = 0.1, class_weight='balanced')
    WG_SGDCL14 = SGDClassifier(loss="log", penalty="l1", alpha = 0.5, class_weight='balanced')
    WG_SGDCL15 = SGDClassifier(loss="log", penalty="l1", alpha = 0.8, class_weight='balanced') 

    models = { 
               'WG_SVM1a': WG_SVM1a, 'WG_SVM1b': WG_SVM1b, 'WG_SVM1c': WG_SVM1c, 'WG_SVM1d': WG_SVM1d, 
               'WG_SVM2a': WG_SVM2a, 'WG_SVM2b': WG_SVM2b, 'WG_SVM2c': WG_SVM2c, 'WG_SVM2d': WG_SVM2d, 
               'WG_SVM2e': WG_SVM2e, 'WG_SVM2f': WG_SVM2f, 'WG_SVM2g': WG_SVM2g,
               'WG_KNN1':WG_KNN1, 'WG_KNN2':WG_KNN2, 'WG_KNN3':WG_KNN3, 'WG_KNN4':WG_KNN4,
               'WG_RKNN1':WG_RKNN1, 'WG_RKNN2':WG_RKNN2, 'WG_RKNN3':WG_RKNN3, 'WG_RKNN4':WG_RKNN4,
               'WG_RKNN5':WG_RKNN5, 'WG_RKNN6':WG_RKNN6, 'WG_RKNN7':WG_RKNN7, 'WG_RKNN8':WG_RKNN8, 'WG_RKNN9':WG_RKNN9,
               'WG_RF1':WG_RF1, 'WG_RF2':WG_RF2, 'WG_RF3':WG_RF3, 'WG_RF4': WG_RF4, 'WG_RF5': WG_RF5,
               'WG_SGDCL11':WG_SGDCL11, 'WG_SGDCL12':WG_SGDCL12, 'WG_SGDCL13':WG_SGDCL13, 'WG_SGDCL14': WG_SGDCL14,
               'WG_SGDCL15': WG_SGDCL15, 
               'WG_MLP1_1': build_MLP(X, 100, 50), 'WG_MLP1_10': build_MLP(X, 100, 50), 'WG_MLP1_100': build_MLP(X, 100, 50),
        
               'NO_GBa':NO_GBa, 'NO_GBb':NO_GBb, 'NO_GBc':NO_GBc, 'NO_GBd':NO_GBd, 'NO_GBe':NO_GBe,
               'NO_GBf':NO_GBf, 'NO_GBg':NO_GBg, 'NO_GBh':NO_GBh, 'NO_GBi':NO_GBi,
               'NO_SVM1a': NO_SVM1a, 'NO_SVM1b': NO_SVM1b, 'NO_SVM1c': NO_SVM1c, 'NO_SVM1d': NO_SVM1d, 
               'NO_SVM2a': NO_SVM2a, 'NO_SVM2b': NO_SVM2b, 'NO_SVM2c': NO_SVM2c, 'NO_SVM2d': NO_SVM2d,
               'NO_SVM2e': NO_SVM2e, 'NO_SVM2f': NO_SVM2f, 'NO_SVM2g': NO_SVM2g,
               'NO_KNN1': NO_KNN1, 'NO_KNN2': NO_KNN2, 'NO_KNN3': NO_KNN3, 'ROS_KNN4': NO_KNN4, 
               'NO_RKNN1':NO_RKNN1, 'NO_RKNN2':NO_RKNN2, 'NO_RKNN3':NO_RKNN3, 'NO_RKNN4':NO_RKNN4,
               'NO_RKNN5':NO_RKNN5, 'NO_RKNN6':NO_RKNN6, 'NO_RKNN7':NO_RKNN7, 'NO_RKNN8':NO_RKNN8, 'NO_RKNN9':NO_RKNN9,
               'NO_RF1': NO_RF1, 'NO_RF2': NO_RF2, 'NO_RF3': NO_RF3, 'NO_RF4': NO_RF4, 'NO_RF5': NO_RF5,
               'NO_SGDCL11':NO_SGDCL11, 'NO_SGDCL12':NO_SGDCL12, 'NO_SGDCL13':NO_SGDCL13, 'NO_SGDCL14': NO_SGDCL14,
               'NO_SGDCL15': NO_SGDCL15,
               'NO_MLP1_1': build_MLP(X, 100, 50), 'NO_MLP1_10': build_MLP(X, 100, 50), 'NO_MLP1_100': build_MLP(X, 100, 50),
  
               'ROS_GBa':NO_GBa, 'ROS_GBb':NO_GBb, 'ROS_GBc':NO_GBc, 'ROS_GBd':NO_GBd, 'ROS_GBe':NO_GBe,
               'ROS_GBf':NO_GBf, 'ROS_GBg':NO_GBg, 'ROS_GBh':NO_GBh, 'ROS_GBi':NO_GBi,
               'ROS_SVM1a': NO_SVM1a, 'ROS_SVM1b': NO_SVM1b, 'ROS_SVM1c': NO_SVM1c, 'ROS_SVM1d': NO_SVM1d, 
               'ROS_SVM2a': NO_SVM2a, 'ROS_SVM2b': NO_SVM2b, 'ROS_SVM2c': NO_SVM2c, 'ROS_SVM2d': NO_SVM2d, 
               'ROS_SVM2e': NO_SVM2e, 'ROS_SVM2f': NO_SVM2f, 'ROS_SVM2g': NO_SVM2g,
               'ROS_KNN1': NO_KNN1, 'ROS_KNN2': NO_KNN2, 'ROS_KNN3': NO_KNN3, 'ROS_KNN4': NO_KNN4, 
               'ROS_RKNN1':NO_RKNN1, 'ROS_RKNN2':NO_RKNN2, 'ROS_RKNN3':NO_RKNN3, 'ROS_RKNN4':NO_RKNN4,
               'ROS_RKNN5':NO_RKNN5, 'ROS_RKNN6':NO_RKNN6, 'ROS_RKNN7':NO_RKNN7, 'ROS_RKNN8':NO_RKNN8, 'ROS_RKNN9':NO_RKNN9,
               'ROS_RF1':NO_RF1, 'ROS_RF2': NO_RF2, 'ROS_RF3': NO_RF3, 'ROS_RF4': NO_RF4, 'ROS_RF5': NO_RF5,  
               'ROS_SGDCL11':NO_SGDCL11, 'ROS_SGDCL12':NO_SGDCL12, 'ROS_SGDCL13':NO_SGDCL13, 'ROS_SGDCL14': NO_SGDCL14,
               'ROS_SGDCL15': NO_SGDCL15,
               'ROS_MLP1_1': build_MLP(X, 100, 50), 'ROS_MLP1_10': build_MLP(X, 100, 50), 
               'ROS_MLP1_100': build_MLP(X, 100, 50), 
        
               'RUS_GBa':NO_GBa, 'RUS_GBb':NO_GBb, 'RUS_GBc':NO_GBc, 'RUS_GBd':NO_GBd, 'RUS_GBe':NO_GBe,
               'RUS_GBf':NO_GBf, 'RUS_GBg':NO_GBg, 'RUS_GBh':NO_GBh, 'RUS_GBi':NO_GBi,
               'RUS_SVM1a': NO_SVM1a, 'RUS_SVM1b': NO_SVM1b, 'RUS_SVM1c': NO_SVM1c, 'RUS_SVM1d': NO_SVM1d, 
               'RUS_SVM2a': NO_SVM2a, 'RUS_SVM2b': NO_SVM2b, 'RUS_SVM2a': NO_SVM2c, 'RUS_SVM2d': NO_SVM2d,
               'RUS_SVM2e': NO_SVM2e, 'RUS_SVM2f': NO_SVM2f, 'RUS_SVM2g': NO_SVM2g,
               'RUS_KNN1': NO_KNN1, 'RUS_KNN2': NO_KNN2, 'RUS_KNN3': NO_KNN3, 'RUS_KNN4': NO_KNN4, 
               'RUS_RKNN1':NO_RKNN1, 'RUS_RKNN2':NO_RKNN2, 'RUS_RKNN3':NO_RKNN3, 'RUS_RKNN4':NO_RKNN4,
               'RUS_RKNN5':NO_RKNN5, 'RUS_RKNN6':NO_RKNN6, 'RUS_RKNN7':NO_RKNN7, 'RUS_RKNN8':NO_RKNN8, 'RUS_RKNN9':NO_RKNN9,
               'RUS_RF1': NO_RF1,  'RUS_RF2': NO_RF2, 'RUS_RF3': NO_RF3, 'RUS_RF4': NO_RF4, 'RUS_RF5': NO_RF5,
               'RUS_SGDCL11':NO_SGDCL11, 'RUS_SGDCL12':NO_SGDCL12, 'RUS_SGDCL13':NO_SGDCL13, 'RUS_SGDCL14': NO_SGDCL14,
               'RUS_SGDCL15': NO_SGDCL15,
               'RUS_MLP1_1': build_MLP(X, 100, 50), 'RUS_MLP1_10': build_MLP(X, 100, 50), 
               'RUS_MLP1_100': build_MLP(X, 100, 50) }

    return models


def dataset_features(X, y, file_path):
    
    ## extraction of meta-features of a given dataset
    # X = features of this dataset
    # y = target
    
    size = Path(file_path).stat().st_size

    nb_class = len( np.unique(y) )
    nb_timeseries = X.shape[0]
    log_nb_timeseries = np.log( X.shape[0] )
    
    timesteps = X.shape[1]
    log_timesteps = np.log( X.shape[1] )

    entropy = scipy.stats.entropy(y)

    ratio = nb_timeseries/timesteps

    imbalance = sum(y)/len(y)
    
    vec = [ size, nb_class, nb_timeseries, log_nb_timeseries, timesteps, 
            log_timesteps, entropy, ratio, imbalance ]
    
    return vec #vector of meta-features


def dataseq_features(X,y):
    
    ## extraction of additional meta-features of a given dataset
    # X = features of this dataset
    # y = target

    skewness = scipy.stats.skew( X, axis=1)
    avg_skewness = np.mean( skewness )
    std_skewness = np.std(  skewness )
    min_skewness = np.min(  skewness )
    max_skewness = np.max(  skewness )
    
    kurto = scipy.stats.kurtosis( X, axis=1)
    avg_kurto = np.mean( kurto )
    std_kurto = np.std(  kurto )
    min_kurto = np.min(  kurto )
    max_kurto = np.max(  kurto )

    std = np.std(X, axis = 1)
    avg_std = np.mean( std )
    std_std = np.std(  std )
    min_std = np.min(  std )
    max_std = np.max(  std )
    
    mean = np.mean(X, axis = 1)
    avg_mean = np.mean( mean )
    std_mean = np.std(  mean )
    min_mean = np.min(  mean )
    max_mean = np.max(  mean )
    
    coef_var = scipy.stats.variation(X, axis=1)
    avg_cv = np.mean( coef_var )
    std_cv = np.std( coef_var )
    min_cv = np.min( coef_var )
    max_cv = np.max( coef_var )
    
    vec = [ avg_skewness, std_skewness, min_skewness, max_skewness,
            avg_kurto, std_kurto, min_kurto, max_kurto,
            avg_std, std_std, min_std, max_std, 
            avg_mean, std_mean, min_mean, max_mean, 
            avg_cv, std_cv, min_cv, max_cv ]
    
    return vec #vector of meta-features


def load_file(file_path):
    ## load the time-series datasets
    ## y is the target to predict
    ## X is the features to be used by the model
    df = pd.read_csv(file_path, encoding="utf-8", sep=',', header=None, index_col=False, low_memory=False)
    y = df[ df.columns[len(df.columns)-1]]
    X = df.drop( df.columns[len(df.columns)-1] , axis=1)
    return X, y


def one_file(X, y, file_path, output_path):
    
    ## this function is in charge of executing all the pipelines on one given dataset(read file)
    ## the variable 'liste_modele' includes all the pipelines 

    liste_modeles = [ 'WG_SVM1a', 'WG_SVM1b', 'WG_SVM1c', 'WG_SVM1d', 
  'WG_SVM2a', 'WG_SVM2b', 'WG_SVM2c', 'WG_SVM2d', 'WG_SVM2e', 'WG_SVM2f','WG_SVM2g',
  'WG_KNN1', 'WG_KNN2', 'WG_KNN3', 'WG_KNN4',
  'WG_RKNN1', 'WG_RKNN2', 'WG_RKNN3', 'WG_RKNN4',
  'WG_RKNN5', 'WG_RKNN6', 'WG_RKNN7', 'WG_RKNN8', 'WG_RKNN9',
  'WG_RF1', 'WG_RF2', 'WG_RF3', 'WG_RF4', 'WG_RF5',
  'WG_SGDCL11', 'WG_SGDCL12', 'WG_SGDCL13', 'WG_SGDCL14',
  'WG_SGDCL15', 'WG_MLP1_1', 'WG_MLP1_10', 'WG_MLP1_100',
                     
  'NO_GBa', 'NO_GBb', 'NO_GBc', 'NO_GBd', 'NO_GBe',
  'NO_GBf', 'NO_GBg', 'NO_GBh', 'NO_GBi',
  'NO_SVM1a', 'NO_SVM1b', 'NO_SVM1c', 'NO_SVM1d', 
  'NO_SVM2a', 'NO_SVM2b', 'NO_SVM2c', 'NO_SVM2d',  'NO_SVM2e', 'NO_SVM2f', 'NO_SVM2g', 
  'NO_KNN1', 'NO_KNN2', 'NO_KNN3', 'ROS_KNN4', 
  'NO_RKNN1', 'NO_RKNN2', 'NO_RKNN3', 'NO_RKNN4',
  'NO_RKNN5', 'NO_RKNN6', 'NO_RKNN7', 'NO_RKNN8', 'NO_RKNN9',
  'NO_RF1', 'NO_RF2', 'NO_RF3', 'NO_RF4', 'NO_RF5',
  'NO_SGDCL11', 'NO_SGDCL12', 'NO_SGDCL13', 'NO_SGDCL14',
  'NO_SGDCL15','NO_MLP1_1', 'NO_MLP1_10', 'NO_MLP1_100', 

  'ROS_GBa', 'ROS_GBb', 'ROS_GBc', 'ROS_GBd', 'ROS_GBe',
  'ROS_GBf', 'ROS_GBg', 'ROS_GBh', 'ROS_GBi',
  'ROS_SVM1a', 'ROS_SVM1b', 'ROS_SVM1c', 'ROS_SVM1d', 
  'ROS_SVM2a', 'ROS_SVM2b', 'ROS_SVM2c', 'ROS_SVM2d', 'ROS_SVM2e', 'ROS_SVM2f', 'ROS_SVM2g', 
  'ROS_KNN1', 'ROS_KNN2', 'ROS_KNN3', 'ROS_KNN4', 
  'ROS_RKNN1', 'ROS_RKNN2', 'ROS_RKNN3', 'ROS_RKNN4',
  'ROS_RKNN5', 'ROS_RKNN6', 'ROS_RKNN7', 'ROS_RKNN8', 'ROS_RKNN9',
  'ROS_RF1', 'ROS_RF2', 'ROS_RF3', 'ROS_RF4', 'ROS_RF5',  
  'ROS_SGDCL11', 'ROS_SGDCL12', 'ROS_SGDCL13', 'ROS_SGDCL14',
  'ROS_SGDCL15','ROS_MLP1_1', 'ROS_MLP1_10', 'ROS_MLP1_100', 
  
  'RUS_GBa', 'RUS_GBb', 'RUS_GBc', 'RUS_GBd', 'RUS_GBe',
  'RUS_GBf', 'RUS_GBg', 'RUS_GBh', 'RUS_GBi',
  'RUS_SVM1a', 'RUS_SVM1b', 'RUS_SVM1c', 'RUS_SVM1d', 
  'RUS_SVM2a', 'RUS_SVM2b', 'RUS_SVM2a', 'RUS_SVM2d', 'RUS_SVM2e', 'RUS_SVM2f', 'RUS_SVM2g',
  'RUS_KNN1', 'RUS_KNN2', 'RUS_KNN3', 'RUS_KNN4', 
  'RUS_RKNN1', 'RUS_RKNN2', 'RUS_RKNN3', 'RUS_RKNN4',
  'RUS_RKNN5', 'RUS_RKNN6', 'RUS_RKNN7', 'RUS_RKNN8', 'RUS_RKNN9',
  'RUS_RF1',  'RUS_RF2', 'RUS_RF3', 'RUS_RF4', 'RUS_RF5',
  'RUS_SGDCL11', 'RUS_SGDCL12', 'RUS_SGDCL13', 'RUS_SGDCL14','RUS_SGDCL15',
  'RUS_MLP1_1', 'RUS_MLP1_10','RUS_MLP1_100' ]
    
    K = len( liste_modeles )
    print('Number of arms:'+str(K) )
    kf = StratifiedKFold(n_splits=5, random_state=0, shuffle=True) #split the file on 5-cross-validation
    cross_val = kf.split(X, y)
    
    scaler = StandardScaler()

    for train_index, test_index in cross_val: # for each fold

        X_train, X_test = X.iloc[train_index,:],  X.iloc[test_index,:]
        y_train, y_test = y[train_index],  y[test_index]
        
        prauc_fold = [np.nan]*K # create empty vectors to collect the information (only prauc is used in the paper)
        rocauc_fold =  [np.nan]*K
        time_fold =  [np.nan]*K
        
        scaler.fit(X_train) # normalize the dataset features
        X_train = pd.DataFrame( scaler.transform(X_train) )
        X_test = pd.DataFrame( scaler.transform(X_test) )

        if len( np.unique(y_train) )>1 and len( np.unique(y_test) )>1: # this is to avoid hold-out error

            X_train = np.array(X_train)
            X_test = np.array(X_test)
            
            meta_features1 = dataset_features(X_train, y_train, file_path) #compute meta-features
            meta_features2 = dataseq_features(X_train, y_train) #compute additional set of meta-features

            
            for index, pipeline_id in enumerate(liste_modeles): #for each pipeline
  
                print(pipeline_id)
                prauc, timer = pipeline_execution(X, pipeline_id, X_train, y_train, X_test, y_test) #execute the pipeline
                prauc_fold[ index ]=prauc # add the prauc to the index that corresponds to this pipeline id
                time_fold[ index ]= timer
                tf.keras.backend.clear_session()
                total_time = total_time + timer
                print(total_time)
 

            vec = np.concatenate( ( [file_path], prauc_fold, time_fold, meta_features1, meta_features2) )
            vec = pd.DataFrame(vec).T
            vec.to_csv( os.path.join(output_path, 'database.csv'), mode='a', header=False, index=False)
            print(vec)
        
    return True


def pipeline_execution(X, pipeline_id, X_train, y_train, X_test, y_test):
    
    ## this function executes one selected pipeline for one given dataset and outputs the performance information that is then added to the banchmarking matrix
    #output prauc = performance value on PRAUC for the pipeline on the dataset
    #timer = a measure of how long it took to train the model (unused in the paper)

    models =  get_models(X)
    choice = pipeline_id.split('_') # the key of the pipeline dictionnary, the first element (choice[0]) indicates which pre-processing operation should be applied
    X_train, y_train = resampling( choice[0], X_train, y_train) 
    
    if ('MLP' in pipeline_id): ## the MLP (multilayer perceptron) need an IF condition because it is based on Keras package that has different functions than SKLEARN
        
        nb_epoch = int( choice[2] )
        weight_option = choice[0]
        weight = weights(weight_option, y_train)  
        
        t0 = time.time()
        models[pipeline_id].fit(X_train, y_train, epochs=nb_epoch, class_weight=weight, verbose=0)
        t1 = time.time()
        prediction = models[pipeline_id].predict(X_test)
           
    else: ## applies to all the functions based on SKLEARN:

        t0 = time.time()
        models[pipeline_id].fit(X_train, y_train)
        t1 = time.time()  
        prediction = models[pipeline_id].predict_proba(X_test)[:,1]
        
    tf.keras.backend.clear_session()
    del models # to free some memory space
    timer = t1 - t0
    prauc =  compute_prauc(prediction, y_test) 
    
    return prauc, timer 


def resampling(imb_str, X, y):
    
    ## this function aims at applying a pre-processing operation on the selected dataset
    ## ROS = random over sampling
    ## RUS = random under sampling
    

    if imb_str == 'ROS':    
        ros = RandomOverSampler(random_state=42)
        X, y = ros.fit_resample(X, y)   
        
    elif imb_str == 'RUS':    
        rus = RandomUnderSampler(random_state=42)
        X, y = rus.fit_resample(X, y) 
        
    X = np.reshape(X, (X.shape[0], X.shape[1]) )
        
    return X, y

def weights(weight_option, y_train):
    
    ## this function aims at applying a pre-processing operation on the selected dataset
    ## WG = cost-sensitive learning by applying a weighted loss to the learning process
    ##weight_option is given by the dictionnary key describing the pipeline
    
    if weight_option == 'WG':
        weight = utils.class_weight.compute_class_weight(class_weight='balanced', y=y_train, classes=np.unique(y_train) )
        weight = {0:weight[0], 1:weight[1] }
    else:
        weight = None
        
    return weight ## class weight to be added to the loss-function during the training
        
def generate_simulation(input_path, output_path):
    
    ## this function prepares the compuring pipeline
    ## it ranks the datasets by size so the smallest datasets are processed first to get material faster
    
    chunk = [f for f in listdir(input_path) if isfile( os.path.join(input_path, f))]
    chunk = [str(input_path)+x for x in chunk]
    
    sizes = [ (file_path, Path(file_path).stat().st_size) for file_path in chunk]
    sizes.sort(key=lambda x: x[1]) #order by size

    chunk = [ x[0] for x in sizes ]

    for file_path in chunk:
        
        print(file_path)
        X, y = load_file(file_path)
        one_file(X, y, file_path, output_path)
        
    return True



########################### commands to launch on the DGX computation cluster:

@click.command()
@click.option('--input-dir', default=None, help='Project input directory.')
@click.option('--output-dir', default=None, help='Experiment output directory.')
def main(input_dir, output_dir):
    
    input_dir=(os.path.join(input_dir, "data2/chunk1/"))
    
    if output_dir and not os.path.isdir(output_dir):
        os.makedirs(output_dir)
        print("Output folder created: {}".format(output_dir))
        
    generate_simulation(input_dir, output_dir )

    
if __name__ == "__main__":
    main()
