"""Train a cldnn on the task4 of DCASE2017 dataset.
GPU run command with Theano backend (with TensorFlow, the GPU is automatically used):
    THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatx=float32 python train*.py

Author: Yong XU
Creat date: 03/04/2017
"""
from __future__ import print_function 
import sys
import cPickle
import numpy as np
import argparse
import glob
import time
import os

import keras
from keras import backend as K
from keras.models import Sequential,Model, load_model
from keras.layers.core import Dense, Dropout, Activation, Flatten, Reshape, Permute,Lambda, RepeatVector
from keras.layers.convolutional import ZeroPadding2D, AveragePooling2D, Conv2D,MaxPooling2D, Convolution1D,MaxPooling1D
from keras.layers.pooling import GlobalMaxPooling2D
from keras.layers import Merge, Input, merge
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, LearningRateScheduler
from keras.layers import LSTM, SimpleRNN, GRU, TimeDistributed, Bidirectional
from keras.layers.normalization import BatchNormalization
import h5py
from keras.layers.merge import Multiply
from sklearn import preprocessing
import random

import config as cfg
from prepare_data import create_folder, load_hdf5_data, do_scale
from data_generator import RatioDataGenerator
from evaluation import io_task4, evaluate
from main_crnn_at import block, slice1, slice2, slice1_output_shape, slice2_output_shape, run_func, recognize, get_stat

def outfunc(vects):
    x,y=vects
    y = K.clip(y, 1e-7, 1.)     # clip to avoid numerical underflow
    y = K.sum(y, axis=1)
    x = K.sum(x, axis=1)
    return x / y

def train(args):
    num_classes = cfg.num_classes
    
    # Load training & testing data
    (tr_x, tr_y, tr_na_list) = load_hdf5_data(args.tr_hdf5_path, verbose=1)
    # (tr_x, tr_y, tr_na_list) = load_hdf5(args.te_hdf5_path, verbose=1)
    (te_x, te_y, te_na_list) = load_hdf5_data(args.te_hdf5_path, verbose=1)
    print("")

    # Scale data
    tr_x = do_scale(tr_x, args.scaler_path, verbose=1)
    te_x = do_scale(te_x, args.scaler_path, verbose=1)
    
    # Build model
    (_, n_time, n_freq) = tr_x.shape
    input_logmel = Input(shape=(n_time, n_freq), name='in_layer')
    a1 = Reshape((n_time, n_freq, 1))(input_logmel)
    
    cnn1 = block(a1)
    cnn1 = block(cnn1)
    cnn1 = MaxPooling2D(pool_size=(1, 2))(cnn1)
    
    cnn1 = block(cnn1)
    cnn1 = block(cnn1)
    cnn1 = MaxPooling2D(pool_size=(1, 2))(cnn1)
    
    cnn1 = block(cnn1)
    cnn1 = block(cnn1)
    cnn1 = MaxPooling2D(pool_size=(1, 2))(cnn1)
    
    cnn1 = block(cnn1)
    cnn3 = block(cnn1)
    cnn3 = MaxPooling2D(pool_size=(1, 2))(cnn1)
    
    cnnout = Conv2D(256, (3, 3), padding="same", activation="relu", use_bias=True)(cnn3)
    cnnout = MaxPooling2D(pool_size=(1, 4))(cnnout)
    
    cnnout = Reshape((240, 256))(cnnout)   # Time step is downsampled to 30. 
    
    rnnout = Bidirectional(GRU(128, activation='linear', return_sequences=True))(cnnout)
    rnnout_gate = Bidirectional(GRU(128, activation='sigmoid', return_sequences=True))(cnnout)
    out = Multiply()([rnnout, rnnout_gate])
    
    out = TimeDistributed(Dense(num_classes, activation='sigmoid'), name='localization_layer')(out)
    det =TimeDistributed(Dense(num_classes, activation='softmax'))(out)
    out=Multiply()([out,det])
    out=Lambda(outfunc, output_shape=(num_classes,))([out, det])
    
    model = Model(input_logmel, out)
    model.summary()
    
    # Compile model
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    
    # Save model callback
    filepath = os.path.join(args.out_model_dir, "gatedAct_rationBal44_lr0.001_normalization_at_cnnRNN_64newMel_240fr.{epoch:02d}-{val_acc:.4f}.hdf5")
    create_folder(os.path.dirname(filepath))
    save_model = ModelCheckpoint(filepath=filepath,
                                 monitor='val_acc', 
                                 verbose=0,
                                 save_best_only=False,
                                 save_weights_only=False,
                                 mode='auto',
                                 period=1)  

    # Data generator
    gen = RatioDataGenerator(batch_size=44, type='train')

    # Train
    model.fit_generator(generator=gen.generate([tr_x], [tr_y]), 
                        steps_per_epoch=100,    # 100 iters is called an 'epoch'
                        epochs=2000000010,      # Maximum 'epoch' to train
                        verbose=1, 
                        callbacks=[save_model], 
                        validation_data=(te_x, te_y))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="")
    subparsers = parser.add_subparsers(dest='mode')
    
    parser_train = subparsers.add_parser('train')
    parser_train.add_argument('--tr_hdf5_path', type=str)
    parser_train.add_argument('--te_hdf5_path', type=str)
    parser_train.add_argument('--scaler_path', type=str)
    parser_train.add_argument('--out_model_dir', type=str)
    
    parser_recognize = subparsers.add_parser('recognize')
    parser_recognize.add_argument('--te_hdf5_path', type=str)
    parser_recognize.add_argument('--scaler_path', type=str)
    parser_recognize.add_argument('--model_dir', type=str)
    parser_recognize.add_argument('--out_dir', type=str)
    
    parser_get_stat = subparsers.add_parser('get_stat')
    parser_get_stat.add_argument('--pred_dir', type=str)
    parser_get_stat.add_argument('--stat_dir', type=str)
    parser_get_stat.add_argument('--submission_dir', type=str)
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        train(args)
    elif args.mode == 'recognize':
        recognize(args, at_bool=True, sed_bool=True)
    elif args.mode == 'get_stat':
        get_stat(args, at_bool=True, sed_bool=True)
    else:
        raise Exception("Incorrect argument!")