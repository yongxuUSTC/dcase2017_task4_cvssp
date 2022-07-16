'''Train a cldnn on the task4 of DCASE2017 dataset.
GPU run command with Theano backend (with TensorFlow, the GPU is automatically used):
    THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatx=float32 python train*.py

Author: Yong XU
Creat date: 03/04/2017
'''

#from __future__ import print_function 

import keras
from keras import backend as K
import sys
import cPickle
import numpy as np
from keras.models import Sequential,Model
#from keras.layers.core import Dense, Dropout, Activation, Flatten, Flatten_last2d, Reshape,Permute,Lambda, RepeatVector
from keras.layers.core import Dense, Dropout, Activation, Flatten, Reshape,Permute,Lambda, RepeatVector
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

def scheduler(epoch):
    initial_lrate = float(0.001)
    cur_lr=initial_lrate*pow(float(0.5),(epoch//5))
    print "learning rate:", cur_lr
    return cur_lr

class RatioDataGenerator(object):
    def __init__(self, batch_size, type, te_max_iter=100):
        assert type in ['train', 'test']
        self._batch_size_ = batch_size
        self._type_ = type
        self._te_max_iter_ = te_max_iter
            
    def _get_lb_list(self, n_samples_list):
        lb_list = []
        for idx in xrange(len(n_samples_list)):
            n_samples = n_samples_list[idx]
            if n_samples < 1000:
                lb_list += [idx]
            elif n_samples < 2000:
                lb_list += [idx] * 2
            elif n_samples < 3000:
                lb_list += [idx] * 3
            elif n_samples < 4000:
                lb_list += [idx] * 4
            else:
                lb_list += [idx] * 5
        return lb_list
        
    def generate(self, xs, ys):
        batch_size = self._batch_size_
        x = xs[0]
        y = ys[0]
        (n_samples, n_labs) = y.shape
        
        n_samples_list = np.sum(y, axis=0)
        lb_list = self._get_lb_list(n_samples_list)
        
        print "n_samples_list:", n_samples_list
        print "lb_list:", lb_list
        print "len(lb_list):", len(lb_list)
        
        
        index_list = []
        for i1 in xrange(n_labs):
            index_list.append(np.where(y[:, i1] == 1)[0])
            
        for i1 in xrange(n_labs):
            np.random.shuffle(index_list[i1])
        
        queue = []
        pointer_list = [0] * n_labs
        len_list = [len(e) for e in index_list]
        iter = 0
        while True:
            if (self._type_) == 'test' and (iter == self._te_max_iter_):
                break
            iter += 1
            batch_x = []
            batch_y = []
            
            while len(queue) < batch_size:
                random.shuffle(lb_list)
                queue += lb_list
                
            batch_idx = queue[0 : batch_size]
            queue[0 : batch_size] = []
            
            n_per_class_list = [batch_idx.count(idx) for idx in xrange(n_labs)]
            
            for i1 in xrange(n_labs):
                if pointer_list[i1] >= len_list[i1]:
                    pointer_list[i1] = 0
                    np.random.shuffle(index_list[i1])
                
                per_class_batch_idx = index_list[i1][pointer_list[i1] : min(pointer_list[i1] + n_per_class_list[i1], len_list[i1])]
                batch_x.append(x[per_class_batch_idx])
                batch_y.append(y[per_class_batch_idx])
                pointer_list[i1] += n_per_class_list[i1]
            batch_x = np.concatenate(batch_x, axis=0)
            batch_y = np.concatenate(batch_y, axis=0)
            yield batch_x, batch_y

class BalanceDataGenerator(object):
    def __init__(self, batch_size, type, max_iter=100):
        assert type in ['train', 'test']
        self._batch_size_ = batch_size
        self._type_ = type
        self._max_iter_ = max_iter
        
    def generate(self, xs, ys):
        batch_size = self._batch_size_
        x = xs[0]
        y = ys[0]
        #(n_samples, n_features) = x.shape ### yong xu commented
        (n_samples, n_labs) = y.shape
        n_each = batch_size // n_labs   
        
        index_list = []
        for i1 in xrange(n_labs):
            index_list.append(np.where(y[:, i1] == 1)[0])
            
        for i1 in xrange(n_labs):
            np.random.shuffle(index_list[i1])
        
        pointer_list = [0] * n_labs
        iter = 0
        while True:
            if (self._type_) == 'test' and (iter == self._max_iter_):
                break
            iter += 1
            batch_x = []
            batch_y = []
            for i1 in xrange(n_labs):
                idx_num = len(index_list[i1])
                if pointer_list[i1] >= idx_num:
                    pointer_list[i1] = 0
                    np.random.shuffle(index_list[i1])
                
                batch_idx = index_list[i1][pointer_list[i1] : min(pointer_list[i1] + n_each, idx_num)]
                batch_x.append(x[batch_idx])
                batch_y.append(y[batch_idx])
                pointer_list[i1] += n_each
            batch_x = np.concatenate(batch_x, axis=0)
            batch_y = np.concatenate(batch_y, axis=0)
            yield batch_x, batch_y

# resize data for fit into CNN. size: (batch_num*color_maps*height*weight)
def reshapeX1( X ):
    N = len(X)
    return X.reshape( (N, t_delay, feadim, 1, 1) )
    
# resize data for fit into CNN. size: (batch_num*color_maps*height*weight)
def reshapeX2( X ):
    N = len(X)
    return X.reshape( (N, t_delay, feadim) )

def reshapeX3( X ):
    N = len(X)
    return X.reshape( (N, t_delay, feadim, 1) )

def reshapeX4( X ):
    N = len(X)
    return X.reshape( (N*t_delay, feadim) )

def reshapeX5( X , sample_num):
    N = len(X)
    return X.reshape( (sample_num, t_delay, feadim, 1) )

def outfunc(vects):
    x,y=vects
    #y=K.sum( y, axis=1 )
    y = K.clip( y, 1.0e-9, 1 )     # clip to avoid numerical underflow
    #z=Lambda(lambda x: K.sum(x, axis=1),output_shape=(8,))(y)
    y = K.sum(y, axis=1)
    #y = K.sum(y, axis=1)
    #z = RepeatVector(249)(z)
    #z=Permute((2,1))(z)
    #return K.sum( x / z, axis=1 )
    #x = K.sum( x, axis=(1,2) )
    x = K.sum( x, axis=1 )
    #x = K.sum( x, axis=1 ) 
    return x / y

def slice1(x):
    return x[:,:,:, 0:64]

def slice2(x):
    return x[:,:,:, 64:128]

def slice1_output_shape(input_shape):
    return tuple([input_shape[0],input_shape[1],input_shape[2],64])

def slice2_output_shape(input_shape):
    return tuple([input_shape[0],input_shape[1],input_shape[2],64])

#parameters:
num_classes=17
feadim=64
t_delay=240 # the len of Utterance is 300
model_out_path="/vol/vssp/msos/yx/t4_d2017/models_val"

# train and test sets:
with h5py.File("/vol/vssp/msos/Audioset/task4_dcase2017_features/packed_features/logmel/training_pack.h5", 'r') as hf:
    tr_x = np.array(hf.get('x'))
    tr_y = np.array(hf.get('y'))
print tr_x.shape
print tr_y.shape



with h5py.File("/vol/vssp/msos/Audioset/task4_dcase2017_features/packed_features/logmel/testing_pack.h5", 'r') as hf:
    va_x = np.array(hf.get('x'))
    va_y = np.array(hf.get('y'))
print va_x.shape
print va_y.shape

###########normalization training and test set
tr_x2=reshapeX4(tr_x)
va_x2=reshapeX4(va_x)
scaler = preprocessing.StandardScaler().fit(tr_x2)
print scaler.mean_, scaler.scale_
tr_x2 = scaler.transform( tr_x2 )
va_x2 = scaler.transform( va_x2 )
tr_x=reshapeX5( tr_x2 , len(tr_x))
va_x=reshapeX5( va_x2 , len(va_x))
####################################
#tr_x=reshapeX3(tr_x)
print tr_x.shape
#va_x=reshapeX3(va_x)
print va_x.shape

def block(input):
    cnn1=Conv2D(128, (3, 3), padding="same", activation="linear", use_bias=False)(input)
    cnn1=BatchNormalization(axis=-1)(cnn1)

    cnn11 = Lambda(slice1,output_shape=slice1_output_shape)(cnn1)
    cnn12 = Lambda(slice2,output_shape=slice2_output_shape)(cnn1)

    cnn11=Activation('linear')(cnn11)
    cnn12=Activation('sigmoid')(cnn12)

    cnn1=Multiply()([cnn11,cnn12])
    return cnn1

###build model by keras

input_audio=Input(shape=(t_delay, feadim, 1))

#input_flat=TimeDistributed(Flatten())(input_audio)

###detection factor for each tag (7 meaningful tags + 1 silence tag = 8 tags)
#det =TimeDistributed(Dense(17,activation='softmax'))(input_flat) # The posterior sum of each tag is 1.0, now the dims of det are 33 frs * 8 tags

cnn1 = block(input_audio)
cnn1 = block(cnn1)
cnn1=MaxPooling2D(pool_size=(2, 2))(cnn1)

cnn1 = block(cnn1)
cnn1 = block(cnn1)
cnn1=MaxPooling2D(pool_size=(2, 2))(cnn1)

cnn1 = block(cnn1)
cnn1 = block(cnn1)
cnn1=MaxPooling2D(pool_size=(2, 2))(cnn1)

cnn1 = block(cnn1)
cnn3 = block(cnn1)
cnn3=MaxPooling2D(pool_size=(1, 3))(cnn1)

cnnout=Conv2D(256, (3, 3), padding="same", activation="relu", use_bias=True)(cnn3)
cnnout=MaxPooling2D(pool_size=(1, 2))(cnnout)

#cnnout=Flatten_last2d()(cnnout)
cnnout=Reshape((30,256))(cnnout)

rnnout=Bidirectional(GRU(128, activation='linear', return_sequences=True))(cnnout)
rnnout_gate=Bidirectional(GRU(128, activation='sigmoid', return_sequences=True))(cnnout)
out=Multiply()([rnnout,rnnout_gate])

out=TimeDistributed(Dense(17,activation='sigmoid'))(out)
#det =TimeDistributed(Dense(17,activation='softmax'))(out)
#out=Multiply()([out,det])
#out=Lambda(outfunc,output_shape=(17,))([out,det])
out=Lambda(lambda x: K.mean(x, axis=1),output_shape=(17,))(out)

allmodel=Model(input_audio, out)
allmodel.summary()

# Let's train the model using RMSprop
allmodel.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

dump_fd=model_out_path+'/gatedAct_rationBal44_lr0.001_normalization_at_cnnRNN_64newMel_240fr.{epoch:02d}-{val_acc:.4f}.hdf5'

eachmodel=ModelCheckpoint(dump_fd,monitor='val_acc',verbose=0,save_best_only=False,save_weights_only=False,mode='auto',period=10)  

#reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
#              patience=5, min_lr=0.00000001)

#reduce_lr = LearningRateScheduler(scheduler)

gen = RatioDataGenerator(batch_size=44, type='train')
#gen = BalanceDataGenerator(batch_size=52, type='train')
#for (batch_x, batch_y) in gen.generate([x], [y]):
#    train_on_batch(batch_x, batch_y, class_weight=None, sample_weight=None)

steps_per_epoch=100
allmodel.fit_generator(gen.generate([tr_x], [tr_y]), steps_per_epoch, epochs=2000000010, verbose=1, callbacks=[eachmodel], validation_data=(va_x, va_y))
#allmodel.fit(tr_x, tr_y, batch_size=100, epochs=31,
#              verbose=1, validation_data=(va_x, va_y), callbacks=[eachmodel]) #, callbacks=[best_model]) 
