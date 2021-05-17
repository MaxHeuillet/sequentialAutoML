###############################################
##### this file corresponds to the LINUCB Exploration Policy subfigures in Figure 1
##################################################

import click
from pathlib import Path

import pickle as pkl
import gzip
import pandas as pd
import tensorflow as tf
import numpy as np
import random
import os
import math
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle

import models 
import utils

def update(A, b, concat, y_concat, item_features, pipeline_id, dataset_idx, reward, nb_steps, step_size):
    
    ## this corresponds to the update of R(t-1) to R'(t)  in the paper
 
    ## input A, b = correspond to the weights of LinUCB exploration policy
    ## y_concat - training data, is called concat because at each round we concatenate the new knowledge to the previous knowledge
    ## concat - training meta-features, un-used in the paper
    ## reward : observed performance for pipeline_idx on dataset_idx

    ## item_features - the features P(t) in the paper
    ## nb_steps = #count the number of past episode
    ## step_size corresponds to s parameter in the paper

    
    if math.isnan(y_concat.loc[dataset_idx,pipeline_id]) and nb_steps>=step_size: #line 25 in Algorithm1 of the paper
        
        ## update LinUCB
        A[pipeline_id] += np.outer( item_features[pipeline_id], item_features[pipeline_id] )
        b[pipeline_id] += reward * item_features[pipeline_id]

    if math.isnan(y_concat.loc[dataset_idx,pipeline_id])  : #from R(t-1) to R'(t), line 15-16
        ### ajouter la connaissance a Rc
        concat = utils.append_update(concat, dataset_idx, pipeline_id, reward)
        y_concat.loc[dataset_idx, pipeline_id] = reward

    return concat, y_concat, A, b
    
def regret_linucb(X_init, y_init, step_size, knowledge, alpha, output_dir, fold_id):
    
        
    ## this corresponds to the computation of the curbs in the subfigures LinUCB Exploration policy for one fold 
    ## input 
    ## input X_init = correspond to the meta-features (un-used)
    ## y_init - corresponds to the exhaustive pipeline benchmark 529 x 175
    ## input knowledge = corresponds in the paper to parameter $c$
    ## alpha - optimism against uncertainty parameter
    
    
    reward = []
    K = 175 #number of pipelines
    epochs = 75 #number of epochs
    latent_size = 40 #dimension of the latent representations 
    nb_steps = 0 #count the number of past episode
    buffer = 0 # this icunt the number of past episodes since last update of P
    
    ### add and store some experiment data here
    explored2 = [] #explored pipelines
    explored3 = []
    
    recommended2 = [] #recommended pipelines
    recommended3 = []
    
    optimal = [] #optimal pipeline from the 175 pipelines benchmark (k^{\star}_{t})

    #### initialization of the weights for each UCB weights associated to a CF algorithm
#     A1 = np.array([np.diag(np.ones(shape=40)) for _ in np.arange(K)])
#     b1 = np.array([np.zeros(shape=40) for _ in np.arange(K)])
    A2 = np.array([np.diag(np.ones(shape=40)) for _ in np.arange(K)])
    b2 = np.array([np.zeros(shape=40) for _ in np.arange(K)])
    A3 = np.array([np.diag(np.ones(shape=40)) for _ in np.arange(K)])
    b3 = np.array([np.zeros(shape=40) for _ in np.arange(K)])
    
    
    #### initialization of the models and save P(t) and Q(t)
    
    model2 = models.get_MF_bias(latent_size, 670, K) #bias aware matrix factorization
    model2.save_weights( os.path.join(output_dir, 'mf_bias{}'.format(knowledge) ) )
    item_features2 = model2.trainable_weights[1].numpy()
    
    model3 = models.get_NeurCF(latent_size, 670, K) #neural collaborative filtering
    model3.save_weights( os.path.join(output_dir, 'neur_cf{}'.format(knowledge) ) )
    item_features3 = model3.trainable_weights[1].numpy()

    for dataset_idx in y_init.index: ### for each episode...
        
        ### extract in the benchmark the row corresponding to this episode 
        y_substr = pd.DataFrame( y_init.loc[dataset_idx] ).T
        
        ### and their associated meta features (un-used in the paper)
        X_substr = pd.DataFrame( X_init.loc[dataset_idx] ).T
        
        ###### if in burn-in phase then explore randomly, with LinUCB otherwise
        ### matrix factorization+bias
        A2, b2, y2, choix2 = utils.create_random(A2, b2, y_substr, knowledge) if nb_steps<step_size else utils.create_linucb_item(A2, b2, y_substr, item_features2, knowledge, alpha)
        explored2.append(choix2)
        ### neuralCF
        A3, b3, y3, choix3 = utils.create_random(A3, b3, y_substr, knowledge) if nb_steps<step_size else utils.create_linucb_item(A3, b3, y_substr, item_features3, knowledge, alpha)
        explored3.append(choix3)

        if dataset_idx == 0:
            ## if first episode, we need to initialize concat2, concat3 objects that are dictionnaries, then at each episode we add the supllementary information

            concat2 = utils.formating(y2, X_substr)
            y_concat2 = y2.copy()  ### sparse matrix genered with LinUCB exploration policy + matrix factorization+bias
            
            concat3 = utils.formating(y3, X_substr)
            y_concat3 = y3.copy()  ### sparse matrix genered with LinUCB exploration policy + neuralCF

        else:

            ### matrix facto+bias
            y_concat2 = pd.concat([y_concat2, y2])
            concat2 = utils.append(concat2, y2, X_substr) ## append the information about current episode to the knowledge R(t)
            ### neuralCF
            y_concat3 = pd.concat([y_concat3, y3])
            concat3 = utils.append(concat3, y3, X_substr)

        ##### this is to put into a dictionnary format the observation that is in the test
        test2 = utils.pred_formating(y2, X_substr)
        test3 = utils.pred_formating(y3, X_substr)
 
        ### MF+bias
        history2, prediction2, timer2, update_features2 = models.one_fold(output_dir,concat2, test2, y_concat2, y2, '2', epochs, latent_size, knowledge, fold_id) # train the model on 75 epochs
        reward_mf_bias, top, rcd_pipeline_id, top_pipeline_id = utils.update_argmax(y_substr, prediction2)
        recommended2.append(rcd_pipeline_id) # get the recommendation obtained after inference
        concat2, y_concat2, A2, b2 = update(A2, b2, concat2, y_concat2, item_features2, rcd_pipeline_id, dataset_idx, reward_mf_bias, nb_steps, step_size) # update the knowledge R(t)
        tf.keras.backend.clear_session()
        
        ### NeurCF, same logic as previous paragraph
        history3, prediction3, timer3, update_features3 = models.one_fold(output_dir,concat3, test3, y_concat3, y3, '3', epochs, latent_size, knowledge, fold_id)
        reward_neurcf, top, rcd_pipeline_id, top_pipeline_id = utils.update_argmax(y_substr, prediction3)
        recommended3.append(rcd_pipeline_id)
        concat3, y_concat3, A3, b3 = update(A3, b3, concat3, y_concat3, item_features3, rcd_pipeline_id, dataset_idx, reward_neurcf, nb_steps, step_size)
        tf.keras.backend.clear_session()


        #### Best of C(t) baselines
        y_ucb2 = y2.copy()
        y_ucb2[ np.isnan(y_ucb2) ] = 0
        reward_ucb2, top, rcd_pipeline_id, top_pipeline_id = utils.update_argmax(y_substr, y_ucb2)
        
        y_ucb3 = y3.copy()
        y_ucb3[ np.isnan(y_ucb3) ] = 0
        reward_ucb3, top, rcd_pipeline_id, top_pipeline_id = utils.update_argmax(y_substr, y_ucb3)
        
        
        #### update latent features P(t) to P' every s steps
        nb_steps = nb_steps +1
        buffer = buffer + 1
        item_features1 = update_features1 if buffer == step_size-1 else item_features1
        item_features2 = update_features2 if buffer == step_size-1 else item_features2
        item_features3 = update_features3 if buffer == step_size-1 else item_features3
        buffer = buffer if buffer < step_size else 0
        
        ## store and save info
        reward.append( [reward_mf, reward_mf_bias, reward_neurcf, reward_knn, reward_rdm, reward_ucb1, reward_ucb2, reward_ucb3, top] )
        full_exploration = [explored1, explored2, explored3]
        full_recommendation = [ recommended1, recommended2, recommended3] 
        optimal.append(top_pipeline_id)
        
        with gzip.open( os.path.join(output_dir, 'LINUCB_reward_{}_{}.pkl.gz'.format(fold_id, knowledge) )  ,'wb') as f:
            pkl.dump(reward,f)
            pkl.dump(full_exploration,f)
            pkl.dump(full_recommendation,f)
            pkl.dump(optimal,f)
        
    return True



@click.command()
@click.option('--input-dir', default=None, help='Project input directory.')
@click.option('--output-dir', default=None, help='Experiment output directory.')
def main(input_dir, output_dir):

    if output_dir and not os.path.isdir(output_dir):
        os.makedirs(output_dir)
        print("Output folder created: {}".format(output_dir))
        
    #######################################
    ############# PARAMETERS TO EXPLORE
    #######################################
    
    for fold_id in range(10):
    
        X_init, y_init = utils.load_data_175_avg('a', 666)
        X_init, y_init = shuffle(X_init, y_init)
        y_init = y_init.reset_index(drop=True)
        X_init = X_init.reset_index(drop=True)
        alpha = 0 ## parametre de la exploration policy UCB
        step_size = 10 ### quand est ce que on update les latent features P et la taille de linitialization

        print(' c = 2')
        regret_linucb(X_init, y_init, step_size, 2, alpha, output_dir, fold_id)

        print('c = 6')
        regret_linucb(X_init, y_init, step_size, 6, alpha, output_dir, fold_id)
    
if __name__ == "__main__":
    main()

