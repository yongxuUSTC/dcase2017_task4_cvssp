"""
Summary:  Train a CNN-RNN audio tagging classifier on the task4 of DCASE2017 
          dataset. 
Author:   Yong XU, Qiuqiang Kong
Created:  03/04/2017
Modified: 24/09/2017
--------------------------------------
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
from keras.layers.core import (Dense, Dropout, Activation, Flatten, Reshape, 
                            Permute,Lambda, RepeatVector)
from keras.layers.convolutional import (ZeroPadding2D, AveragePooling2D, 
                                Conv2D,MaxPooling2D, Convolution1D,MaxPooling1D)
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

def scheduler(epoch):
    initial_lrate = float(0.001)
    cur_lr=initial_lrate*pow(float(0.5),(epoch//5))
    print("learning rate: %f" % cur_lr)
    return cur_lr

def block(input):
    cnn = Conv2D(128, (3, 3), padding="same", activation="linear", use_bias=False)(input)
    cnn = BatchNormalization(axis=-1)(cnn)

    cnn1 = Lambda(slice1, output_shape=slice1_output_shape)(cnn)
    cnn2 = Lambda(slice2, output_shape=slice2_output_shape)(cnn)

    cnn1 = Activation('linear')(cnn1)
    cnn2 = Activation('sigmoid')(cnn2)

    out = Multiply()([cnn1, cnn2])
    return out

def slice1(x):
    return x[:, :, :, 0:64]

def slice2(x):
    return x[:, :, :, 64:128]

def slice1_output_shape(input_shape):
    return tuple([input_shape[0],input_shape[1],input_shape[2],64])

def slice2_output_shape(input_shape):
    return tuple([input_shape[0],input_shape[1],input_shape[2],64])

def train(args):
    num_classes = cfg.num_classes
    
    # Load training & testing data
    (tr_x, tr_y, tr_na_list) = load_hdf5_data(args.tr_hdf5_path, verbose=1)
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
    cnn1 = MaxPooling2D(pool_size=(2, 2))(cnn1)
    
    cnn1 = block(cnn1)
    cnn1 = block(cnn1)
    cnn1 = MaxPooling2D(pool_size=(2, 2))(cnn1)
    
    cnn1 = block(cnn1)
    cnn1 = block(cnn1)
    cnn1 = MaxPooling2D(pool_size=(2, 2))(cnn1)
    
    cnn1 = block(cnn1)
    cnn3 = block(cnn1)
    cnn3 = MaxPooling2D(pool_size=(1, 2))(cnn1)
    
    cnnout = Conv2D(256, (3, 3), padding="same", activation="relu", use_bias=True)(cnn3)
    cnnout = MaxPooling2D(pool_size=(1, 4))(cnnout)
    
    cnnout = Reshape((30, 256))(cnnout)   # Time step is downsampled to 30. 
    
    rnnout = Bidirectional(GRU(128, activation='linear', return_sequences=True))(cnnout)
    rnnout_gate = Bidirectional(GRU(128, activation='sigmoid', return_sequences=True))(cnnout)
    out = Multiply()([rnnout, rnnout_gate])
    
    out = TimeDistributed(Dense(num_classes, activation='sigmoid'), name='localization_layer')(out)
    out = Lambda(lambda x: K.mean(x, axis=1),output_shape=(num_classes,))(out)
    
    model = Model(input_logmel, out)
    model.summary()
    
    # Compile model
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    
    # Save model callback
    filepath = os.path.join(args.out_model_dir, 
        "gatedAct_rationBal44_lr0.001_normalization_at_cnnRNN_64mel_240fr.{epoch:02d}-{val_acc:.4f}.hdf5")
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

def run_func(func, x, batch_size):
    pred_all = []
    for ptr in xrange(0, len(x), batch_size):
        batch_x = x[ptr : ptr + batch_size]
        [pred] = func([batch_x, 0.])
        pred_all.append(pred)
    pred_all = np.concatenate(pred_all, axis=0)
    return pred_all

def recognize(args, at_bool, sed_bool):
    (te_x, te_y, te_na_list) = load_hdf5_data(args.te_hdf5_path, verbose=1)
    x = te_x
    y = te_y
    na_list = te_na_list
    
    x = do_scale(x, args.scaler_path, verbose=1)
    
    fusion_at_list = []
    fusion_sed_list = []
    # for epoch in range(19, 81, 10):
    for epoch in range(20, 30, 1):
        t1 = time.time()
        [model_path] = glob.glob(os.path.join(args.model_dir, 
            "*.%02d-0.*.hdf5" % epoch))
        model = load_model(model_path)
        
        # Audio tagging
        if at_bool:
            pred = model.predict(x)
            fusion_at_list.append(pred)
        
        # Sound event detection
        if sed_bool:
            in_layer = model.get_layer('in_layer')
            loc_layer = model.get_layer('localization_layer')
            func_loc_output = K.function([in_layer.input, K.learning_phase()], 
                                         [loc_layer.output])
            pred3d = run_func(func_loc_output, x, batch_size=64)
            fusion_sed_list.append(pred3d)
        
        print("Prediction time: %s" % (time.time() - t1,))
    
    # Write out AT probabilities
    if at_bool:
        fusion_at = np.mean(np.array(fusion_at_list), axis=0)
        print("AT shape: %s" % (fusion_at.shape,))
        io_task4.at_write_prob_mat_to_csv(
            na_list=na_list, 
            prob_mat=fusion_at, 
            out_path=os.path.join(args.out_dir, "at_prob_mat.csv.gz"))
    
    # Write out SED probabilites
    if sed_bool:
        fusion_sed = np.mean(np.array(fusion_sed_list), axis=0)
        print("SED shape:%s" % (fusion_sed.shape,))
        io_task4.sed_write_prob_mat_list_to_csv(
            na_list=na_list, 
            prob_mat_list=fusion_sed, 
            out_path=os.path.join(args.out_dir, "sed_prob_mat_list.csv.gz"))
            
    print("Prediction finished!")

def get_stat(args, at_bool, sed_bool):
    lbs = cfg.lbs
    step_time_in_sec = cfg.step_time_in_sec
    max_len = cfg.max_len
    thres_ary = [0.3] * len(lbs)

    # Calculate AT stat
    if at_bool:
        pd_prob_mat_csv_path = os.path.join(args.pred_dir, "at_prob_mat.csv.gz")
        at_stat_path = os.path.join(args.stat_dir, "at_stat.csv")
        at_submission_path = os.path.join(args.submission_dir, "at_submission.csv")
        
        at_evaluator = evaluate.AudioTaggingEvaluate(
            weak_gt_csv="meta_data/groundtruth_weak_label_testing_set.csv", 
            lbs=lbs)
        
        at_stat = at_evaluator.get_stats_from_prob_mat_csv(
                        pd_prob_mat_csv=pd_prob_mat_csv_path, 
                        thres_ary=thres_ary)
                        
        # Write out & print AT stat
        at_evaluator.write_stat_to_csv(stat=at_stat, 
                                       stat_path=at_stat_path)
        at_evaluator.print_stat(stat_path=at_stat_path)
        
        # Write AT to submission format
        io_task4.at_write_prob_mat_csv_to_submission_csv(
            at_prob_mat_path=pd_prob_mat_csv_path, 
            lbs=lbs, 
            thres_ary=at_stat['thres_ary'], 
            out_path=at_submission_path)
               
    # Calculate SED stat
    if sed_bool:
        sed_prob_mat_list_path = os.path.join(args.pred_dir, "sed_prob_mat_list.csv.gz")
        sed_stat_path = os.path.join(args.stat_dir, "sed_stat.csv")
        sed_submission_path = os.path.join(args.submission_dir, "sed_submission.csv")
        
        sed_evaluator = evaluate.SoundEventDetectionEvaluate(
            strong_gt_csv="meta_data/groundtruth_strong_label_testing_set.csv", 
            lbs=lbs, 
            step_sec=step_time_in_sec, 
            max_len=max_len)
                            
        # Write out & print SED stat
        sed_stat = sed_evaluator.get_stats_from_prob_mat_list_csv(
                    pd_prob_mat_list_csv=sed_prob_mat_list_path, 
                    thres_ary=thres_ary)
                    
        # Write SED to submission format
        sed_evaluator.write_stat_to_csv(stat=sed_stat, 
                                        stat_path=sed_stat_path)                     
        sed_evaluator.print_stat(stat_path=sed_stat_path)
        
        # Write SED to submission format
        io_task4.sed_write_prob_mat_list_csv_to_submission_csv(
            sed_prob_mat_list_path=sed_prob_mat_list_path, 
            lbs=lbs, 
            thres_ary=thres_ary, 
            step_sec=step_time_in_sec, 
            out_path=sed_submission_path)
                                                        
    print("Calculating stat finished!")

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
        recognize(args, at_bool=True, sed_bool=False)
    elif args.mode == 'get_stat':
        get_stat(args, at_bool=True, sed_bool=False)
    else:
        raise Exception("Incorrect argument!")