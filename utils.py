###########################
###########################
#
# in this file there are useful functions to conduct:
# 1) the exploration policies (KNN, Random, LinUCB)
# 2) the sequential updates of the knowledge R used by the agent
#
###########################
###########################


import click
from pathlib import Path

import tensorflow as tf
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Input, Embedding, Flatten, Dot, Dense
from tensorflow.keras.models import Model
from tensorflow.keras import optimizers
from tensorflow.keras import losses

import pandas as pd
import numpy as np
import random
import os
import scipy.stats as ss
from numpy import *
import math
import random
import gzip
import pickle as pkl
import collections
from scipy.spatial import distance
from itertools import groupby


from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle
from sklearn.model_selection import KFold
from sklearn.metrics import ndcg_score
from sklearn import preprocessing
from sklearn.preprocessing import PolynomialFeatures


##############################################################
#################### RANDOM EXPLORATION POLICY
##############################################################

def create_random(A, b, truth, knowledge):
    
    ### corresponds to the random exploration policy in the paper
    ## input : truth = the pipelines benchmarking
    ## input knowledge = corresponds in the paper to parameter $c$
    ## input A, b = un-used in the function but correspond to the weights of LinUCB exploration policy
    
    target = np.empty( (len(truth.index),len(truth.columns)) ) ## an empty matrix/vector that is filled with the selected values C(t)
    target[:] = np.NaN
    target = pd.DataFrame(target, index = truth.index)

    idx = truth.index[0]
    choix = random.sample( set(truth.columns), knowledge) 
    for cell in choix: # fill the empy matrix 'target' with the selected knowledge
        target.loc[idx,cell] = truth.loc[idx,cell]
            
    return A, b, target, choix

##############################################################
#################### KNN DISTANCE EXPLORATION POLICY
##############################################################

def create_distance(A, b, knowledge, X_selection, y_selection, X_substr, truth):
    
    ### corresponds to the KNN exploration policy in the paper
    ## input : truth = the pipelines benchmarking
    ## input knowledge = corresponds in the paper to parameter $c$
    ## input A, b = un-used in the function but correspond to the weights of LinUCB exploration policy
    ## input y_selection correspond to a pipeline benchmark made of 140 datasets that is kept appart and not used in the knowledge of our agents, following the method used in Fusi. et. al. 2017 and AutoSklearn to explore $c$ first pipelines
    ## input X_slection corresponds to the meta-features for each of the 140 dataset 
    
    target = np.empty( (len(truth.index),len(truth.columns)) )
    target[:] = np.NaN
    target = pd.DataFrame(target, index = truth.index) ## empty matrix/row that will be filled with the selected values $C(t)
    idx = truth.index[0] # we keep index unchanged
    
    sim = np.array( [ distance.cityblock(X_selection.loc[dataset], X_substr ) for dataset in X_selection.index] ) #for the current dataset, compute L1 similarity to the 140 datasets in meta-feature space(X_selection)
    neighbs = np.argsort(sim)[:knowledge] # select the $c$ nearest neighbors
    select_final = [ np.argmax(y_selection.loc[index]) for index in neighbs ] # for each neighbor, select its best pipeline
    
    while all_equal(select_final): # if the $c$ neighbors all point to the same pipeline, add the n-1 nearest neighbor until you get 2 different pipelines (it is required for collaborative filtering algorithms to have at least 2 performance instances in C(t) for a given dataset in order to infer the unknown performances)
        knowledge += 1 
        neighbs = np.argsort(sim)[:knowledge]
        select_final = [ np.argmax(y_selection.loc[index]) for index in neighbs ]
        
    for cell in select_final: # fill the empty matrix/knowledge with the selected pipelines
        target.loc[idx,cell] = truth.loc[idx,cell]
        
    return A, b, target, select_final

def all_equal(iterable):
    ## check if all the elements in a list are all equal == all the pipelines selected are identical
    g = groupby(iterable)
    return next(g, True) and not next(g, False)



##############################################################
#################### LINCUB EXPLORATION POLICY
##############################################################

def create_linucb_item(A, b, truth, item_features, knowledge, alpha):
    
    ### correspond LinUCB Exploration Policy  in the paper
    ## input : truth = the pipelines benchmarking
    ## input knowledge = corresponds in the paper to parameter $c$
    ## input A, b = uthe weights of LinUCB exploration policy 
    ## alpha == optimism parameter, set to 0 and 0.1 in the experiments
    
    target = np.empty( (len(truth.index),len(truth.columns)) ) ## empty matrix/row of knowledge
    target[:] = np.NaN
    target = pd.DataFrame(target, index = truth.index)
            
    K = 175
 
    pbt = []

    for a in range(K):
        x = item_features[a] 
        theta = np.linalg.inv( A[a] ).dot( b[a] )
        p_temp = theta.T @ x + alpha * np.sqrt( x.T @  np.linalg.inv( A[a] ).dot(x) ) ## this corresponds to the UCB
        pbt.append( p_temp )
          
    choix = np.argsort( pbt )[-knowledge:] ## select the $c$ pipelines with the highest UCB
    
    idx = truth.index[0]
    for chx in choix: ## update the algorithm weights for next episode
        reward = truth.loc[idx,chx]
        target.loc[idx, chx] = reward
        A[chx] = A[chx] + np.outer( item_features[chx], item_features[chx] )
        b[chx] = b[chx] + reward * item_features[chx]
        
    return A, b, target, choix


#####################################################################
#################### USEFULL FUNCTIONS
######################################################################

def load_data_175_avg(input_dir, nb_datasets):
    
    ## this function loads the pipeline benchmarking
    
#     data_dir = 'your_dir/RECOMMENDER-AUTOML/data'
    data_dir = '/nas-data/ModelReco/outputs'
#     data_dir = '/home/mheuillet/nas/corpus-balancing/ModelReco/outputs'
      
    #### UNIDIM 3.1
    df1 = pd.read_csv(  os.path.join(data_dir,'model-reco-job-0.1.dev33-g235c521.d20210207223236/database.csv' ), header=None)
    #### UNIDIM 3.2
    df2 = pd.read_csv( os.path.join(data_dir,'model-reco-job-0.1.dev33-g235c521.d20210207223337/database.csv'), header=None)
    #### UNIDIM 3.3
    df3 = pd.read_csv( os.path.join(data_dir,'model-reco-job-0.1.dev33-g235c521.d20210207223444/database.csv'), header=None)
    #### UNIDIM 3.4
    df4 = pd.read_csv( os.path.join(data_dir,'model-reco-job-0.1.dev33-g235c521.d20210207223833/database.csv'), header=None)

    df = pd.concat( [df1, df2, df3, df4] ) #concatenate the 4 datasets
    df = df.dropna()
    df = df.reset_index(drop=True)
    
    liste = set(df[0]) ## this is to substract a given number of datasets (unused in the paper)
    rdm = random.sample( liste, k = min( nb_datasets, len( liste ) ) )
    df = df[ df[0].isin(rdm) ]
    
    ## group_by mean the result of the 5-folds cross validation
    df_GB = df.groupby(0).mean()
    df_GB.index = range( len(df_GB.index) )
    df_GB = df_GB.T.reset_index(drop=True).T
    
    y_GB = df_GB[ range(175) ]
    scaler2 = StandardScaler()
    X_GB = df_GB[ range(350, 379)  ] # extract the meta-features
    scaler2.fit(X_GB) #normalize them
    X_GB = pd.DataFrame( scaler2.transform(X_GB), index = y_GB.index )
    
    return X_GB, y_GB

##############################################################
######## FUNCTIONS TO UPDATE SEQUENTIALLY THE KNOWLEDGE AT EACH EPISODE
##############################################################


def pred_formating(y,X):
    
    ## this function changes the format of 1 single row of the pipelines benchmarking from matrix to a dictionnary
    ## input y = one row fro the matrix of performances 666 x 175
    ## input X one from (corresponding) in the matrix of meta-features
    ## output: reformated dictionnary
    ## we use the terminology users/items where users = datasets and items = pipelines from the Recommender Systems litterature
    
    idx = y.index[0]
    taille_action = len(y.columns)
    formated = {'users':np.array([idx]*taille_action), 'items':np.array([i for i in range(taille_action) ]),'context':np.array( [X.loc[idx,].to_list()] * taille_action ) }
    
    return formated


def formating(y, X):
    
    ## this function changes the format for all the rows of the pipelines benchmarking from matrix to a dictionnary
    ## input y = the matrix of performances 666 x 175
    ## input X the matrix of meta-features
    ## output: reformated dictionnary
    ## we use the terminology users/items where users = datasets and items = pipelines from the Recommender Systems litterature

    
    users = []
    perfs = []
    items = []
    total_context = []
    
    for idx in y.index :
        
        query = y[ y.index==idx ].T.dropna()
        perfs.extend( query[idx] )
        users.extend( [idx]*len(query) )
        items.extend( query.index ) 
        total_context.extend( [X.loc[idx,].to_list()] * len(query)  )
        

    users = np.array(users)
    items = np.array(items)
    perfs = np.array(perfs, dtype=float)
    total_context =  np.array(total_context, dtype=float)
    
    formated = {'users':users, 'items':items, 'perfs':perfs, 'context':total_context }
    
    return formated

def append(old, new, X):
    
    ## this function appends the information about several pipeline performances from one dataset to the knowledge  dictionary
    ## this aims to sequentially update the knowledge of the agent
    ## in the paper this corresponds to the update from R(t-1) to R'(t)
    ## input old = dictionnary that contains the knowledge from previous episodes 
    ## input new = information to be added to the dictionnary that is farmatted as a pandas dataframe made of 1 row

    idx = new.index[0] # keep the index unchanged
    query = new.T.dropna() # in case the knowledge is sparse

    old['users'] = np.append( old['users'], [idx]*len(query)  )
    old['items'] = np.append( old['items'], query.index  )
    old['perfs'] = np.append( old['perfs'], query[idx]  )
    old['context'] = np.append(  old['context'] ,  [X.loc[idx,].to_list()] * len(query) , axis=0 )

    return old

def append_update(old, dataset_id, pipeline_id, perf):
    
    ## this function appends information about 1 pipeline performance from one dataset to the knowledge  dictionary
    ## this aims to sequentially update the knowledge of the agent 
    ## in the paper this corresponds to the update from R(t-1) to R'(t)
    ## input old = dictionnary that contains the knowledge from previous episodes 
    ## input dataset_id / pipeline_id to locate where to add the knowledge
    ## input performance = add performances
    
    old['users'] = np.append( old['users'], dataset_id  )
    old['items'] = np.append( old['items'], pipeline_id  )
    old['perfs'] = np.append( old['perfs'], perf  )

    return old ## updated knowledge dictionnary
    
def update_argmax(truth, prediction):
    
    ## this function gives information about the recommendation policy
    ## input truth : the observed truth from the pipeline benchmarking
    ## input prediction : the predicted performances from the CF-algorithm
   

    top = float( truth[ np.argmax( truth ) ] )
    reward = float( truth[np.argmax( prediction )]  ) ## recommended pipeline = argmax ( \bar R(t) )
    rcd_pipeline_id = np.argmax( prediction )
    top_pipeline_id = np.argsort( truth.values )[0][-10:]

    return reward, top, rcd_pipeline_id, top_pipeline_id
