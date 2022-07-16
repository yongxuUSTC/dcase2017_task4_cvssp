#!/bin/bash

TEST_WAV_DIR="/vol/vssp/AP_datasets/audio/audioset/task4_dcase2017_audio/official_downloads/testing"
TRAIN_WAV_DIR="/vol/vssp/AP_datasets/audio/audioset/task4_dcase2017_audio/official_downloads/training"
EVALUATION_WAV_DIR="/vol/vssp/datasets/audio/audioset/task4_dcase2017_audio/official_downloads/evaluation"

WORKSPACE="/vol/vssp/msos/qk/workspaces/ICASSP2018_dcase"

# Extract features
python prepare_data.py extract_features --wav_dir=$TEST_WAV_DIR --out_dir=$WORKSPACE"/features/logmel/testing" --recompute=True
python prepare_data.py extract_features --wav_dir=$TRAIN_WAV_DIR --out_dir=$WORKSPACE"/features/logmel/training" --recompute=True
python prepare_data.py extract_features --wav_dir=$EVALUATION_WAV_DIR --out_dir=$WORKSPACE"/features/logmel/evaluation" --recompute=True

# Pack features
python prepare_data.py pack_features --fe_dir=$WORKSPACE"/features/logmel/testing" --csv_path="meta_data/testing_set.csv" --out_path=$WORKSPACE"/packed_features/logmel/testing.h5"
python prepare_data.py pack_features --fe_dir=$WORKSPACE"/features/logmel/training" --csv_path="meta_data/training_set.csv" --out_path=$WORKSPACE"/packed_features/logmel/training.h5"
python prepare_data.py pack_features --fe_dir=$WORKSPACE"/features/logmel/evaluation" --csv_path="" --out_path=$WORKSPACE"/packed_features/logmel/evaluation.h5"

# Calculate scaler
python prepare_data.py calculate_scaler --hdf5_path=$WORKSPACE"/packed_features/logmel/training.h5" --out_path=$WORKSPACE"/scalers/logmel/training.scaler"

# Train AT
THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python main_crnn_at.py train --tr_hdf5_path=$WORKSPACE"/packed_features/logmel/training.h5" --te_hdf5_path=$WORKSPACE"/packed_features/logmel/testing.h5" --scaler_path=$WORKSPACE"/scalers/logmel/training.scaler" --out_model_dir=$WORKSPACE"/models/crnn_at"

# Recognize AT
THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python main_crnn_at.py recognize --te_hdf5_path=$WORKSPACE"/packed_features/logmel/testing.h5" --scaler_path=$WORKSPACE"/scalers/logmel/training.scaler" --model_dir=$WORKSPACE"/models/crnn_at" --out_dir=$WORKSPACE"/preds/crnn_at"

# Get stat of AT
THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python main_crnn_at.py get_stat --pred_dir=$WORKSPACE"/preds/crnn_at" --stat_dir=$WORKSPACE"/stats/crnn_at" --submission_dir=$WORKSPACE"/submissions/crnn_at"

# Train SED
THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python main_crnn_sed.py train --tr_hdf5_path=$WORKSPACE"/packed_features/logmel/training.h5" --te_hdf5_path=$WORKSPACE"/packed_features/logmel/testing.h5" --scaler_path=$WORKSPACE"/scalers/logmel/training.scaler" --out_model_dir=$WORKSPACE"/models/crnn_sed"

# Recognize SED
THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python main_crnn_at.py recognize --te_hdf5_path=$WORKSPACE"/packed_features/logmel/testing.h5" --scaler_path=$WORKSPACE"/scalers/logmel/training.scaler" --model_dir=$WORKSPACE"/models/crnn_sed" --out_dir=$WORKSPACE"/preds/crnn_sed"

# Get stat of SED
THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python main_crnn_at.py get_stat --pred_dir=$WORKSPACE"/preds/crnn_sed" --stat_dir=$WORKSPACE"/stats/crnn_sed" --submission_dir=$WORKSPACE"/submissions/crnn_sed"
