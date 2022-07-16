'''Train a cldnn on the task4 of DCASE2017 dataset.
GPU run command with Theano backend (with TensorFlow, the GPU is automatically used):
    THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatx=float32 python train*.py

Author: Yong XU
Creat date: 03/04/2017
'''

#from __future__ import print_function

import keras
from keras import backend as K
from keras.models import load_model
import sys
import cPickle as pickle
import numpy as np
from keras.models import Sequential,Model
from keras.layers.core import Dense, Dropout, Activation, Flatten, Reshape,Permute,Lambda, RepeatVector
from keras.layers.convolutional import ZeroPadding2D, AveragePooling2D, Conv2D,MaxPooling2D, Convolution1D,MaxPooling1D
from keras.layers import Merge, Input, merge
from keras.callbacks import ModelCheckpoint
from keras.layers import LSTM, SimpleRNN, GRU, TimeDistributed, Bidirectional
import h5py
import os
import shutil
from sklearn import preprocessing
import gzip
import glob

# resize data for fit into CNN. size: (batch_num*color_maps*height*weight)
def reshapeX1( X ):
    N = len(X)
    return X.reshape( (1, N, feadim, 1, 1) )
    
# resize data for fit into CNN. size: (batch_num*color_maps*height*weight)
def reshapeX2( X ):
    N = len(X)
    return X.reshape( (N, t_delay, feadim) )

def reshapeX3( X ):
    N = len(X)
    return X.reshape( (1, N, feadim) )

def reshapeX6( X ):
    N = len(X)
    return X.reshape( (1, N, feadim, 1) )

def reshapeX4( X ):
    N = len(X)
    return X.reshape( (N*t_delay, feadim) )

def reshapeX5( X , sample_num):
    N = len(X)
    return X.reshape( (sample_num, t_delay, feadim, 1) )

#parameters:
num_classes=17
feadim=64
t_delay=240 # the len of Utterance is 300

## train sets:
#with h5py.File("/vol/vssp/msos/Audioset/task4_dcase2017_features/packed_features/logmel/training_pack.h5", 'r') as hf:
#    tr_x = np.array(hf.get('x'))
#    tr_y = np.array(hf.get('y'))
#print tr_x.shape
#print tr_y.shape

#test sets:
with h5py.File("/vol/vssp/msos/Audioset/task4_dcase2017_features/packed_features/logmel/testing_pack.h5", 'r') as hf:
    va_x = np.array(hf.get('x'))
    va_y = np.array(hf.get('y'))
    va_id = np.array(hf.get('na_list'))

print va_id.shape
print va_x.shape
print va_y.shape

###########normalization training and test set
#tr_x2=reshapeX4(tr_x)
va_x2=reshapeX4(va_x)
#scaler = preprocessing.StandardScaler().fit(tr_x2)
#print scaler.mean_, scaler.scale_
#with open('tr_norm.pickle', 'wb') as handle:
#    pickle.dump(scaler, handle, protocol=pickle.HIGHEST_PROTOCOL)


with open('tr_norm.pickle', 'rb') as handle:
    scaler = pickle.load(handle)
va_x2 = scaler.transform( va_x2 )
va_x=reshapeX5( va_x2 , len(va_x))
####################################
f=gzip.open('/vol/vssp/msos/yx/t4_d2017/dcase2017_task4_evaluation_code_20170712/data/at_prob_mat_icassp2018_crnn_baseline.csv.gz','w')
#f2=gzip.open('/vol/vssp/msos/yx/t4_d2017/dcase2017_task4_evaluation_code_20170712/data/sed_prob_mat_list.csv.gz','w')
pred_fusion=[[0 for x in range(17)] for y in range(488)] 
print len(pred_fusion)
model_num=0
#for epoch in range(20,110,10):
for epoch in range(19,81,10):
#19-61: f1=56,59
#19-71: f1=56,58.5
#19-81: f1=0.567,0.596
#19-91: f1=0.565,0.584
    print "epoch:", epoch
    model_num=model_num+1
    print "model num:", model_num
    
    path='/vol/vssp/msos/yx/t4_d2017/models_val/icassp_baseline_crnn_rationBal44_lr0.001_norm_64newMel_240fr.%d-0.*.hdf5'%epoch
    for model_f in glob.glob(path):
        print model_f
        md=load_model(model_f)
        #md.summary()
    
#         def recognize():
            
        p_y_pred = md.predict( va_x )
        pred_fusion=pred_fusion+p_y_pred
            
            #shutil.copy('at_prediction.csv', '/vol/vssp/msos/yx/t4_d2017/dcase2017_task4_evaluation_code_20170616/examples/.')
            #os.wait()
            #shutil.copy('prob_mat.csv', '/vol/vssp/msos/yx/t4_d2017/dcase2017_task4_evaluation_code_20170616/examples/.')
            #os.wait()
        
            #path="/vol/vssp/msos/yx/t4_d2017/dcase2017_task4_evaluation_code_20170616"
            #os.chdir(path)   
            #cmd="python runme.py"
            #os.system(cmd)

# do fusion
p_y_pred= pred_fusion/model_num
print p_y_pred.shape
p_y_pred=list(p_y_pred)
#print p_y_pred
for i in range(len(va_x)):
    #print i
    p_y_pred_s = p_y_pred[i]
    f.write("%s.wav" % (va_id[i]))
    for e in p_y_pred_s:
#        f.write("    %s" % e)
        f.write("	%s" % e)
    f.write("\n")
    #sys.exit()
    # copy the audio tagging results to form the sed results, simple mode
    #for j in range(240):
    #   #print i
    #   f2.write("%s.wav" % (va_id[i]))
    #   for e in p_y_pred_s:
    #       f2.write("	%s" % e)
    #   f2.write("\n")     

f.close
#f2.close

# if __name__ == '__main__':
#     recognize()
