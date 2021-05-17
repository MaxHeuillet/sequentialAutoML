import click
from pathlib import Path

import time
import scipy
import scipy.stats as ss
import pandas as pd
import numpy as np
import random
import os
import scipy.stats as ss
from numpy import *
import random
import collections
from numpy import *
import math
import pickle as pkl
import gzip

import tensorflow as tf
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Input, Embedding, Flatten, Dot, Dense
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras import optimizers
from tensorflow.keras import losses

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle
from sklearn.model_selection import KFold
from sklearn.metrics import ndcg_score
from sklearn import preprocessing
from sklearn.preprocessing import PolynomialFeatures



# def distance_based(n_neighbors, y_train_rdm, y_test_rdm):

#     ########## algorithme de collaborative filtering type KNN
    
#     y_test_copy = y_test_rdm.copy().fillna(0)
#     y_train_copy = y_train_rdm.copy().fillna(0)
    
#     if len(y_train_rdm) == 1:

#         data_final = y_test_rdm.copy().fillna(0)
        
#     else:

#         sim = np.array([ scipy.spatial.distance.cosine(y_test_copy, y_train_copy.loc[train_id] )  for train_id in y_train_copy.index] )
#         neighbs = np.argsort(sim)[:n_neighbors]
#         neighbs_val = [ np.array(y_train_copy.loc[idx]) for idx in neighbs ]
#         neighbs_sim = [ sim[idx] for idx in neighbs ]
#         data_sparse = np.array( y_test_rdm.copy()  )[0]
        
#         step1 = np.array([ np.multiply( neighb, neighbs_sim[idx] ) for idx,neighb in enumerate(neighbs_val) ])
#         preds = np.nansum( step1, axis=0 )  / np.sum(neighbs_sim)
#         data_final =  preds
#         #[ preds[idx] if math.isnan(value) else value for idx, value in enumerate(data_sparse) ]
        
#     return data_final

def load_weights(model, name, output_dir):
    
    ### load the weights of the matirx factorization (noted P and Q in the paper) for a new episode
    
    try:
        model.load_weights( os.path.join(output_dir, name) )
        
    except tf.errors.NotFoundError as e:
        pass
        
    return model

def one_fold(output_dir, concat, val, y_concat, y_val, option, epochs, latent_size, knowledge, fold_id):
    
    ## input:
    # output_dir - directory used to save and load the weights P(t) and Q(t)
    # y_concat - training data, is called concat because at each round we concatenate the new knowledge to the previous knowledge
    # concat - training meta-features, un-used in the paper
    #  y_val - this corresponds to the validation data, it corresponds to 1 dataset where we only know C(t) pipelines
    # val - for the given dataset in validation, a set of meta-features (un-used in the paper)
    # option - which CF algorithm is chosen?
    # epochs - number of epochs for the training set to 75
    #latent_size - dimension of P(t) and Q(t) set to 40
    # corresponds to parameter $c$ knowledge
    # fold_id - in the experiment we run 10 cross validation, we need this to load the weights corresponding to the right fold
    
    ####### train the matrix factorization and predict
    ### we use the user/item terminology from the recommender system comunity in this algorithm
    ### user = dataset
    ### item = pipeline
    
    ## output: in the paper history is un-used, it corresponds to the loss value at each epoch


    num_users = 666 # number of datasets
    num_items = len(y_concat.columns) #number of de pipelines
    
    ### compute the bias to perform a bias-aware matrix factorization
    
    global_bias = y_concat.sum().sum() / np.count_nonzero(~np.isnan(y_concat)) # average performance on all the training dataset

    
    #### compute  bias for each user and item 
    concat_user_bias = y_concat.mean(axis=1) - global_bias # average observed performance for one dataset in the training set
    concat_item_bias = y_concat.mean(axis=0) - global_bias # average observed performance for one pipeline in the training set
    concat['user_bias'] = np.array( concat_user_bias[ concat['users'] ] )
    concat['item_bias'] = np.array( concat_item_bias[ concat['items'] ] )
    concat['global_bias'] = np.array( [global_bias] * len( concat['users'] ) )

    
    ### same mais pour lobservation sur laquelle on veut faire la prediction
    val_user_bias = y_val.mean(axis=1) - global_bias # average observed performance for one dataset in the val set
    val_item_bias = y_val.mean(axis=0) - global_bias # average observed performance for one pipeline in the val set
    val_item_bias[np.isnan(val_item_bias)] = 0
    val['user_bias'] = np.array( val_user_bias[ val['users'] ] )
    val['item_bias'] = np.array( val_item_bias[ val['items'] ] )
    val['global_bias'] = np.array( [global_bias] * len( val['users'] ) )

#     if option == '1' : # un-used in the paper
#         print('#### MF')
#         t0 = time.time() 
#         model = get_MF(latent_size, num_users, num_items)
#         model = load_weights(model, 'mf{}{}'.format(knowledge, fold_id), output_dir)#### charger les poids du precendent round
#         reg_params = [0.0001, 0.0001] ### regularization
#         model, history = train_MF(model, reg_params, epochs, False, concat) 
#         model.save_weights( os.path.join(output_dir, 'mf{}{}'.format(knowledge, fold_id) ) )
#         t1 = time.time() 
#         prediction = predict(model, val, y_val, option)
    if option == '2' : #bias aware matrix factorization
        t0 = time.time() 
        model = get_MF_bias(latent_size, num_users, num_items ) # at each episode, generate a model object
        model = load_weights(model, 'mf_bias{}{}'.format(knowledge, fold_id), output_dir) #load the weights P(t-1) and Q(t-1)
        reg_params = [0.0001, 0.0001] ### regularization
        model, history = train_MF(model, reg_params, epochs, True, concat) #train
        model.save_weights( os.path.join( output_dir, 'mf_bias{}{}'.format(knowledge, fold_id)) ) #save P(t) and Q(t)
        t1 = time.time() 
        prediction = predict(model, val, y_val, option) # get the prediction \bar R(t) to be used by the recommendation policy eventually
        
    elif option == '3' : #neural collaborative filtering algorithm, same logic as previous paragraph
        print('#### NeurCF')
        t0 = time.time() 
        model = get_NeurCF(latent_size, num_users, num_items)
        model = load_weights(model, 'neur_cf{}{}'.format(knowledge, fold_id), output_dir)
        reg_params = [0.0001, 0.0001, 0.0001, 0.0001] ### regularization
        model, history = train_NeurCF(model, reg_params, epochs, option, concat)
        model.save_weights( os.path.join( output_dir, 'neur_cf{}{}'.format(knowledge, fold_id) ) )
        t1 = time.time() 
        prediction = predict(model, val, y_val, option) # get the prediction \bar R(t) to be used by the recommendation policy eventually

        
    item_features =  model.trainable_weights[1].numpy() ### in the paper this corresponds to P(t)
    timer = t1 - t0
       
    
    return history, prediction, timer, item_features



def get_item_features(y_concat, concat, epochs, latent_size):

    num_users =len(y_concat.index)
    num_items = len(y_concat.columns)
    
    global_bias = y_concat.sum().sum() / np.count_nonzero(~np.isnan(y_concat))

    concat_user_bias = y_concat.mean(axis=1) - global_bias
    concat_item_bias = y_concat.mean(axis=0) - global_bias
    concat['user_bias'] = np.array( concat_user_bias[ concat['users'] ] )
    concat['item_bias'] = np.array( concat_item_bias[ concat['items'] ] )
    concat['global_bias'] = np.array( [global_bias] * len( concat['users'] ) )

    model = get_MF_bias(latent_size, num_users, num_items )
    reg_params = [0.0001, 0.0001]
    model, history = train_MF(model, reg_params, epochs, True, concat)

    item_features =  model.trainable_weights[1].numpy()
    
    return item_features

def compute_loss(y_pred, y_true):
    
    # computes the mean squared error loss function 
    # y_pred = the prediction of the mode
    # y_true = the true value
    
    y_true = np.reshape( y_true, (-1,1) )
    diff = tf.math.subtract( y_pred, y_true )
    square = tf.math.square( diff )
    loss = tf.math.reduce_mean( square )
    return loss #loss function value


def train_MF(model, reg_params, epochs, bias, train):
    
    ### train matrix factorization
    
    opt = tf.keras.optimizers.Adam(lr=0.01, clipnorm=1)
    
    loss_history = []

    for epoch in range(epochs): # for each 75 epoch

        with tf.GradientTape() as tape: # update the gradients manually
            
            if bias: # if we use a bias aware matrix factorization

                pred_train = model( (train['users'], train['items'], train['user_bias'], train['item_bias'], train['global_bias'] ), training=True) 
                loss_train =  compute_loss(pred_train, train['perfs']) # compute the loss

            else: 
                
                pred_train = model( (train['users'], train['items'] ), training=True) 
                loss_train =  compute_loss(pred_train, train['perfs']) 

        grads = tape.gradient(loss_train, model.trainable_weights ) # update the gradients given the loss
        gdts = opt.apply_gradients( zip(grads, model.trainable_weights) )
        
        loss_history.append(loss_train) ## add the current loss value to the history for each epoch

    return model, loss_history


def train_NeurCF(model, reg_params, epochs, option, train):
    
    ### train neural CF matrix factorization
    
    opt = tf.keras.optimizers.Adam(lr=0.01, clipnorm=1)
    
    loss_history = []
    
    for epoch in range(epochs): # for each 75 epoch

        with tf.GradientTape() as tape: # update the gradients manually
            
            if option == '3': #### NeurCF
                
                pred_train = model( (train['users'], train['items'] ), training=True) 
                loss_train =  compute_loss(pred_train, train['perfs']) # compute the loss
                                                                 
            elif option == '4': #### Final NeurCF, un-used in the paper, corresponds to NeuralCF + meta-features
                
                pred_train = model( (train['users'], train['items'],  train['user_bias'], train['item_bias'], train['global_bias'], train['context']), training=True) 
                loss_train =  compute_loss(pred_train, train['perfs'])  # compute the loss
                                                                                               

        grads = tape.gradient(loss_train, model.trainable_weights ) # update the gradients given the loss
        gdts = opt.apply_gradients( zip(grads, model.trainable_weights) )
        
        loss_history.append(reg_loss_train) ## add the current loss value to the history for each epoch

    return model, loss_history



def predict(model, val, prior_measures, option):
    
    ### given the chosen cf-algorithm (option) predict the unknown performances for a validation dataset (val)
    
    nb_users = len(np.unique(val['users']))
    nb_items = len(np.unique(val['items']))
                 
    
    if option == '1': # simple matrix factorization
        prediction = model( ( val['users'], val['items'] ) )
        
    elif option == '2': # bias aware matrix factorization
        prediction = model( ( val['users'], val['items'], val['user_bias'], val['item_bias'], val['global_bias']) )

    elif option == '3': #neural CF
        prediction = model( ( val['users'], val['items'] ) )
    
    elif option == '4': # neural CF + meta features
        prediction = model( ( val['users'], val['items'], val['user_bias'], val['item_bias'], val['global_bias'], val['context']) )
                                                            
    prediction_matrix = pd.DataFrame( np.zeros( (nb_users, nb_items) ), index = prior_measures.index ) # empty matrix first

    ### on remplit la matrice avec les predictions du modele
    for usr, itm, vls in zip( val['users'], val['items'], prediction ): # fill the matrix with the predicted values
        
        prediction_matrix.at[usr,itm] = vls.numpy()[0]
    
    return prediction_matrix

####################################################################################
####################################################################################
# CF ALGORITHM INSTANCIATION FUNCTIONS
####################################################################################
####################################################################################



def get_MF_bias(latent_size, num_users, num_items):
    
    # this function returns a TF object that corresponds to a bias aware matrix factorization
    # input
    # latent_size - size of the dimensions in P(t) and Q(t)
    # num_users - number of datasets 
    # num_items - number of pipelines (175)
    
    #### correspond a la factorization de matrice + bias (bias aware matrix factorization) dans le rapport
    
    users = Input(shape=(1,), dtype='int32', name = 'user_input')
    items = Input(shape=(1,), dtype='int32', name = 'item_input')
    
    user_bias = Input(shape=(1,), dtype='float32', name = 'user_bias')
    item_bias = Input(shape=(1,), dtype='float32', name = 'item_bias')
    global_bias = Input(shape=(1,), dtype='float32', name = 'global')
    
    ########### MF PART:
    
    MF_Embedding_User = Embedding(input_dim = num_users, output_dim = latent_size, 
                                  embeddings_regularizer = tf.keras.regularizers.l2(0.01),
                                  name = 'mf_embedding_user', input_length=1)
    
    MF_Embedding_Item = Embedding(input_dim = num_items, output_dim = latent_size, 
                                  embeddings_regularizer = tf.keras.regularizers.l2(0.01),
                                  name = 'mf_embedding_item', input_length=1) 
    
    mf_user_latent = Flatten()( MF_Embedding_User(users) )
    mf_item_latent = Flatten()( MF_Embedding_Item(items) )

    MF_vec = tf.multiply(mf_user_latent, mf_item_latent)
    MF_vec = tf.concat( (MF_vec, user_bias, item_bias, global_bias), axis=1  )
    
    ########### FINAL
 
    final_dense = Dense(1, activation='sigmoid')
    
    prediction = final_dense(  MF_vec ) 
    
    model = Model([users, items, user_bias, item_bias, global_bias], prediction) # get TF model object
    
    return model


def get_MF(latent_size, num_users, num_items):
    
        
    # this function returns a TF object that corresponds to a matrix factorization
    # input
    # latent_size - size of the dimensions in P(t) and Q(t)
    # num_users - number of datasets 
    # num_items - number of pipelines (175)
    
    
    #### simple matrix factorization 
    
    users = Input(shape=(1,), dtype='int32', name = 'user_input')
    items = Input(shape=(1,), dtype='int32', name = 'item_input')

    ########### MF PART:
    
    MF_Embedding_User = Embedding(input_dim = num_users, output_dim = latent_size, 
                                  embeddings_regularizer = tf.keras.regularizers.l2(0.01),
                                  name = 'mf_embedding_user', input_length=1)
    
    MF_Embedding_Item = Embedding(input_dim = num_items, output_dim = latent_size, 
                                  embeddings_regularizer = tf.keras.regularizers.l2(0.01),
                                  name = 'mf_embedding_item', input_length=1) 
    
    mf_user_latent = Flatten()( MF_Embedding_User(users) )
    mf_item_latent = Flatten()( MF_Embedding_Item(items) )

    MF_vec = tf.multiply(mf_user_latent, mf_item_latent)

    ########### FINAL
 
    final_dense = Dense(1, activation='sigmoid')
    
    prediction = final_dense(  MF_vec ) 
    
    model = Model([users, items], prediction) # get TF model object
    
    return model

####################################################################################
####################################################################################
######################## NEURAL COLABORATIVE FILTERING
####################################################################################
####################################################################################
                                                    

def get_NeurCF(latent_size, num_users, num_items):
    
    ### corresponds to a neural CF matrix factorization
    # input
    # latent_size - size of the dimensions in P(t) and Q(t)
    # num_users - number of datasets 
    # num_items - number of pipelines (175)
    
    users = Input(shape=(1,), dtype='int32', name = 'user_input')
    items = Input(shape=(1,), dtype='int32', name = 'item_input')

    ########### MF PART:
    
    MF_Embedding_User = Embedding(input_dim = num_users, output_dim = latent_size, 
                                  embeddings_regularizer = tf.keras.regularizers.l2(0.01),
                                  name = 'mf_embedding_user', input_length=1)
    
    MF_Embedding_Item = Embedding(input_dim = num_items, output_dim = latent_size, 
                                  embeddings_regularizer = tf.keras.regularizers.l2(0.01),
                                  name = 'mf_embedding_item', input_length=1) 
    
    mf_user_latent = Flatten()( MF_Embedding_User(users) )
    mf_item_latent = Flatten()( MF_Embedding_Item(items) )

    MF_vec = tf.multiply(mf_user_latent, mf_item_latent)

    ########### MLP PART:
    
    MLP_Embedding_User = Embedding(input_dim = num_users, output_dim = latent_size, 
                                   embeddings_regularizer = tf.keras.regularizers.l2(0.01),
                                   name = "mlp_embedding_user", input_length=1)
    
    MLP_Embedding_Item = Embedding(input_dim = num_items, output_dim = latent_size, 
                                   embeddings_regularizer = tf.keras.regularizers.l2(0.01),
                                   name = 'mlp_embedding_item', input_length=1)  

    
    mlp_user_latent = Flatten()( MLP_Embedding_User(users) )
    mlp_item_latent = Flatten()( MLP_Embedding_Item(items) )

    MLP_vec = tf.concat( (mlp_user_latent, mlp_item_latent), axis=1 )
        
    
    ########### NEUR-MF:

    final_dense = Dense(1, activation='sigmoid')
                
    vector = tf.concat( (MF_vec, MLP_vec), axis=1 )
    prediction = final_dense( vector ) 
    
    model = Model([users, items], prediction) # get TF model object
    
        
    return model  
    
                                                                 
def get_NeurCF_final(latent_size, num_users, num_items, nb_sidefeatures):
    
    ###### not used in the paper
    
    users = Input(shape=(1,), dtype='int32', name = 'user_input')
    items = Input(shape=(1,), dtype='int32', name = 'item_input')
    
    user_bias = Input(shape=(1,), dtype='float32', name = 'user_bias')
    item_bias = Input(shape=(1,), dtype='float32', name = 'item_bias')
    global_bias = Input(shape=(1,), dtype='float32', name = 'global')
    
    kwargs = Input(shape=(nb_sidefeatures,), dtype='float32', name = 'kwargs')
    
    
    ########### MF PART:
    
    MF_Embedding_User = Embedding(input_dim = num_users, output_dim = latent_size, 
                                  embeddings_regularizer = tf.keras.regularizers.l2(0.01),
                                  name = 'mf_embedding_user', input_length=1)
    
    MF_Embedding_Item = Embedding(input_dim = num_items, output_dim = latent_size, 
                                  embeddings_regularizer = tf.keras.regularizers.l2(0.01),
                                  name = 'mf_embedding_item', input_length=1) 
    
    mf_user_latent = Flatten()( MF_Embedding_User(users) )
    mf_item_latent = Flatten()( MF_Embedding_Item(items) )

    MF_vec = tf.multiply(mf_user_latent, mf_item_latent)
    MF_vec = tf.concat( (MF_vec, user_bias, item_bias, global_bias), axis=1  )
        
    ########### MLP PART:
    
    MLP_Embedding_User = Embedding(input_dim = num_users, output_dim = latent_size, 
                                   embeddings_regularizer = tf.keras.regularizers.l2(0.01),
                                   name = "mlp_embedding_user", input_length=1)
    
    MLP_Embedding_Item = Embedding(input_dim = num_items, output_dim = latent_size, 
                                   embeddings_regularizer = tf.keras.regularizers.l2(0.01),
                                   name = 'mlp_embedding_item', input_length=1)  
        
    dense1 = Dense(40)
    dense2 = Dense(20)
    dense3 = Dense(10)
    
    mlp_user_latent = Flatten()( MLP_Embedding_User(users) )
    mlp_item_latent = Flatten()( MLP_Embedding_Item(items) )

    MLP_vec = tf.concat( (mlp_user_latent, mlp_item_latent, kwargs), axis=1 )
        
    MLP_vec = dense3( dense2( dense1( MLP_vec ) ) )
    
    ########### NEUR-MF:

    dense_out1 = Dense(33)
    dense_out2 = Dense(20) 
    final_dense = Dense(1, activation='sigmoid')
                
    vector = tf.concat( (MF_vec, MLP_vec), axis=1 )
    prediction = final_dense( dense_out2( dense_out1( vector ) ) )
    
    model = Model([users, items, user_bias, item_bias, global_bias,kwargs ], prediction)
    
    return model
                                                                 


