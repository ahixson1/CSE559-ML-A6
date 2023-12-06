'''
Part of catELMo
(c) 2023 by  Pengfei Zhang, Michael Cai, Seojin Bang, Heewook Lee, and Arizona State University.
See LICENSE-CC-BY-NC-ND for licensing.
'''

import sys
import time
import os
import argparse
import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm
import tensorflow as tf
from numpy import mean, std
from tensorflow import keras
from tensorflow.math import subtract

from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RepeatedKFold, train_test_split
from sklearn.metrics import precision_recall_fscore_support,roc_auc_score, precision_score, recall_score, f1_score
#from keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import EarlyStopping

from keras.layers import Input, Flatten, Dense, Dropout, LeakyReLU
from keras.models import Model
from keras.layers.merge import concatenate
from tensorflow.keras.layers import (
    BatchNormalization, SeparableConv2D, MaxPooling2D, Activation, Flatten, Dropout, Dense, LayerNormalization
)
import gc

warnings.filterwarnings('ignore')
warnings.simplefilter(action='ignore', category=FutureWarning)
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"]='0,1,2'


def get_inputs(embedding_type):
    if embedding_type == 'catelmo':
        dat = pd.read_pickle("catelmo.pkl")
#         dat = dat.sample(frac=1).reset_index(drop=True)     
    elif embedding_type == 'tcrbert':
        dat = pd.read_pickle("tcrbert.pkl")
#         dat = dat.sample(frac=1).reset_index(drop=True) 
    return dat


def load_data_split(dat,split_type, seed):
    n_fold = 5
    idx_test_fold = 0
    idx_val_fold = -1
    idx_test = None
    idx_train = None
    x_pep = dat.epi
    x_tcr = dat.tcr
    
    if split_type == 'random':
        n_total = len(x_pep)
    elif split_type == 'epi':
        unique_peptides = np.unique(x_pep)
        n_total = len(unique_peptides)
    elif split_type == 'tcr':
        unique_tcrs = np.unique(x_tcr)
        n_total = len(unique_tcrs)
        
    np.random.seed(seed)    
    idx_shuffled = np.arange(n_total)
    np.random.shuffle(idx_shuffled)
    
    # Determine data split from folds
    n_test = int(round(n_total / n_fold))
    n_train = n_total - n_test

    # Determine position of current test fold
    test_fold_start_index = idx_test_fold * n_test
    test_fold_end_index = (idx_test_fold + 1) * n_test

    if split_type == 'random':
        # Split data evenly among evenly spaced folds
        # Determine if there is an outer testing fold
        if idx_val_fold < 0:
            idx_test = idx_shuffled[test_fold_start_index:test_fold_end_index]
            idx_train = list(set(idx_shuffled).difference(set(idx_test)))
        else:
            validation_fold_start_index = args.idx_val_fold * n_test
            validation_fold_end_index = (args.idx_val_fold + 1) * n_test
            idx_test_remove = idx_shuffled[test_fold_start_index:test_fold_end_index]
            idx_test = idx_shuffled[validation_fold_start_index:validation_fold_end_index]
            idx_train = list(set(idx_shuffled).difference(set(idx_test)).difference(set(idx_test_remove)))
    elif split_type == 'epi':
        if idx_val_fold < 0:
            idx_test_pep = idx_shuffled[test_fold_start_index:test_fold_end_index]
            test_peptides = unique_peptides[idx_test_pep]
            idx_test = [index for index, pep in enumerate(x_pep) if pep in test_peptides]
            idx_train = list(set(range(len(x_pep))).difference(set(idx_test)))
        else:
            validation_fold_start_index = args.idx_val_fold * n_test
            validation_fold_end_index = (args.idx_val_fold + 1) * n_test
            idx_test_remove_pep = idx_shuffled[test_fold_start_index:test_fold_end_index]
            test_remove_peptides = unique_peptides[idx_test_remove_pep]
            idx_test_pep = idx_shuffled[validation_fold_start_index:validation_fold_end_index]
            test_peptides = unique_peptides[idx_test_pep]
            idx_test = [index for index, pep in enumerate(x_pep) if pep in test_peptides]
            idx_test_remove = [index for index, pep in enumerate(x_pep) if pep in test_remove_peptides]
            idx_train = list(set(range(len(x_pep))).difference(set(idx_test)).difference(set(idx_test_remove)))
    elif split_type == 'tcr':
        if idx_val_fold < 0:
            idx_test_tcr = idx_shuffled[test_fold_start_index:test_fold_end_index]
            test_tcrs = unique_tcrs[idx_test_tcr]
            idx_test = [index for index, tcr in enumerate(x_tcr) if tcr in test_tcrs]
            idx_train = list(set(range(len(x_tcr))).difference(set(idx_test)))
        else:
            validation_fold_start_index = args.idx_val_fold * n_test
            validation_fold_end_index = (args.idx_val_fold + 1) * n_test
            idx_test_remove_tcr = idx_shuffled[test_fold_start_index:test_fold_end_index]
            test_remove_tcrs = unique_tcrs[idx_test_remove_tcr]
            idx_test_tcr = idx_shuffled[validation_fold_start_index:validation_fold_end_index]
            test_tcrs = unique_tcrs[idx_test_tcr]
            idx_test = [index for index, tcr in enumerate(x_tcr) if tcr in test_tcrs]
            idx_test_remove = [index for index, tcr in enumerate(x_tcr) if tcr in test_remove_tcrs]
            idx_train = list(set(range(len(x_tcr))).difference(set(idx_test)).difference(set(idx_test_remove)))

    testData = dat.iloc[idx_test, :].sample(frac=1).reset_index(drop=True)
    trainData = dat.iloc[idx_train, :].sample(frac=1).reset_index(drop=True)
    

    print('================check Overlapping========================')
    print('number of overlapping tcrs: ', str(len(set(trainData.tcr).intersection(set(testData.tcr)))))
    print('number of overlapping epitopes: ', str(len(set(trainData.epi).intersection(set(testData.epi)))))
    
    # tcr_split testing read 
    X1_test_list, X2_test_list, y_test_list = testData.tcr_embeds.to_list(), testData.epi_embeds.to_list(),testData.binding.to_list()
    X1_test, X2_test, y_test = np.array(X1_test_list), np.array(X2_test_list), np.array(y_test_list)
    # tcr_split training read 
    X1_train_list = trainData.tcr_embeds.to_list()
    X2_train_list = trainData.epi_embeds.to_list()
    y_train_list = trainData.binding.to_list()
    X1_train = np.array(X1_train_list)
    del X1_train_list, testData, trainData, X1_test_list, X2_test_list, y_test_list
    gc.collect()
    X2_train = np.array(X2_train_list)
    y_train = np.array(y_train_list)
    return  X1_train, X2_train, y_train, X1_test, X2_test, y_test#, testData, trainData


def train_(embedding_name,X1_train, X2_train, y_train, X1_test, X2_test, y_test, split):
    # define two sets of inputs
    inputA = Input(shape=(len(X1_train[0]),))
    inputB = Input(shape=(len(X2_train[0]),))

    #x = Dense(2048,kernel_initializer = 'he_uniform')(inputA)
    #x = BatchNormalization()(x)
    #x = Dropout(0.3)(x)
    #x = tf.nn.silu(x)

    x = Dense(1536,kernel_initializer = 'he_uniform')(inputA)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)
    x = tf.nn.silu(x)
    x = Model(inputs=inputA, outputs=x)
    
    y = Dense(1536,kernel_initializer = 'he_uniform')(inputB)
    y = BatchNormalization()(y)
    y = Dropout(0.4)(y)
    y = tf.nn.silu(y)
    y = Model(inputs=inputB, outputs=y)

    #y = Dense(2048,kernel_initializer = 'he_uniform')(y)
    #y = BatchNormalization()(y)
    #y = Dropout(0.4)(y)
    #y = tf.nn.silu(y)
#     combined = concatenate([x.output, y.output, abs(subtract(x.output,y.output))])
    combined = concatenate([x.output, y.output])
    
    z = Dense(1024)(combined)
    z = BatchNormalization()(z)
    z = Dropout(0.4)(z)
    z = tf.nn.silu(z)

    #
    #z = Dense(1024)(z)
    #z = BatchNormalization()(z)
    #z = Dropout(0.3)(z)
    #z = tf.nn.silu(z)

    z = Dense(1, activation='sigmoid')(z)
    model = Model(inputs=[x.input, y.input], outputs=z)
    model.compile(loss = 'binary_crossentropy', optimizer = 'adam')
    model.summary()
    
    ## model fit
    checkpoint_filepath = 'models/catELMo_changed/' + embedding_name + ' ' + split + '.hdf5'
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_filepath,
                                                                    save_weights_only=True,
                                                                    monitor='val_loss',
                                                                    mode='min',
                                                                    save_best_only=True)
    
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience = 30)
    #model.fit([X1_train,X2_train], y_train, verbose=0, validation_split=0.20, epochs=200, batch_size = 32, callbacks=[es, model_checkpoint_callback])
    model.fit([X1_train,X2_train], y_train, verbose=1, validation_split=0.20, epochs=200, batch_size = 32, callbacks=[es, model_checkpoint_callback])
#     model.save('models/' + embedding_name + '.hdf5')
    yhat = model.predict([X1_test, X2_test])
    # from pdb import set_trace as bp
    # bp()

    auc = roc_auc_score(y_test, yhat)  
    print('================Performance========================')
    print(embedding_name+'AUC: ' + str(auc))

    
    yhat[yhat>=0.5] = 1
    yhat[yhat<0.5] = 0
    
    accuracy = accuracy_score(y_test, yhat)
    precision1 = precision_score(
        y_test, yhat, pos_label=1, zero_division=0)
    precision0 = precision_score(
        y_test, yhat, pos_label=0, zero_division=0)
    recall1 = recall_score(y_test, yhat, pos_label=1, zero_division=0)
    recall0 = recall_score(y_test, yhat, pos_label=0, zero_division=0)
    f1macro = f1_score(y_test, yhat, average='macro')
    f1micro = f1_score(y_test, yhat, average='micro')
   
    results = precision_recall_fscore_support(y_test,yhat, average='macro')
    print('precision_recall_fscore_macro ' + str(results))
    print('acc is '  + str(accuracy))
    print('precision1 is '  + str(precision1))
    print('precision0 is '  + str(precision0))
    print('recall1 is '  + str(recall1))
    print('recall0 is '  + str(recall0))
    print('f1macro is '  + str(f1macro))
    print('f1micro is '  + str(f1micro))

    return auc, accuracy, results[0], precision1, precision0, results[1], recall1, recall0, f1macro, f1micro
    
def main(embedding, split,fraction,seed, gpu):
    os.environ["CUDA_VISIBLE_DEVICES"]=gpu
    dat = get_inputs(embedding)
    tr_dat = dat
    runs = 5
    auc, accuracy, precision, precision1, precision0, recall, recall1, recall0, f1macro, f1micro = [np.zeros(runs, dtype='float') for i in range(10)]
    for idx in range(0, runs):
        tr_dat = dat.sample(frac=fraction, replace=True, random_state=seed).reset_index(drop=True) # comment this out if no fraction used
        #X1_train, X2_train, y_train, X1_test, X2_test, y_test, testData, trainData = load_data_split(tr_dat,split, seed)
        X1_train, X2_train, y_train, X1_test, X2_test, y_test = load_data_split(tr_dat,split, seed)
        auc[idx], accuracy[idx], precision[idx], precision1[idx], precision0[idx], recall[idx], recall1[idx], recall0[idx], f1macro[idx], f1micro[idx] = train_(embedding + '_' + split + '_seed_' + str(seed) + '_fraction_' + str(fraction), X1_train, X2_train, y_train, X1_test, X2_test, y_test, split)
        
    mean_auc = mean(auc)
    std_auc = std(auc)
    mean_acc = mean(accuracy)
    std_acc = std(accuracy)
    mean_prec = mean(precision)
    std_prec = std(precision)
    mean_prec1 = mean(precision1)
    std_prec1 = std(precision1)
    mean_prec0 = mean(precision0)
    std_prec0 = std(precision0)
    mean_rec = mean(recall)
    std_rec = std(recall)
    mean_rec1 = mean(recall1)
    std_rec1 = std(recall1)
    mean_rec0 = mean(recall0)
    std_rec0 = std(recall0)
    mean_f1mac = mean(f1macro)
    std_f1mac = std(f1macro)
    mean_f1mic = mean(f1micro)
    std_f1mic = std(f1micro)

    print("Mean of AUC =",str(mean_auc),",STD of AUC =",std_auc)
    print("Mean of accuracy =",str(mean_acc),",STD of accuracy =",std_acc)
    print("Mean of precision  =",mean_prec,",STD of precision =",std_prec)
    print("Mean of precision1  =",mean_prec1,",STD of precision1 =",std_prec1)
    print("Mean of precision0 =",mean_prec0,",STD of precision0 =",std_prec0)
    print("Mean of recall =",mean_rec,",STD of recall =",std_rec)
    print("Mean of recall1 =",mean_rec1,",STD of recall1 =",std_rec1)
    print("Mean of recall0 = ",mean_rec0,",STD of recall0 =",std_rec0)
    print("Mean of f1macro =",mean_f1mac,",STD of f1macro =",std_f1mac)
    print("Mean of f1micro =",mean_f1mic,",STD of f1micro =",std_f1mic)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--embedding', type=str,help='elmo or blosum62', default='catelmo')
    parser.add_argument('--split', type=str,help='random, tcr or epi', default='epi')
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--fraction', type=float, default=1.0) 
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    main(args.embedding, args.split, args.fraction, args.seed, args.gpu)
