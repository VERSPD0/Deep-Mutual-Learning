#!/bin/bash
#
# This script performs the following operations:
# Training 2 MobileNets with DML on Market-1501
#
# Usage:
# cd Deep-Mutual-Learning
# ./scripts/train_dml_mobilenet_on_market.sh


# Where the TFRecords are saved to.
DATASET_DIR=/home/dingyf/lwy/Deep-Mutual-Learning-master/data/cifar100-tfrecord
# DATASET_DIR=/home/dingyf/lwy/Deep-Mutual-Learning-master/data/cifar-100-tfrecord

# Where the checkpoint and logs will be saved to.
DATASET_NAME=cifar100
# NET_NAME=mobilenet_v1
NET_NAME=resnet44
SAVE_NAME=cifar100_dml_${NET_NAME}
CKPT_DIR=${SAVE_NAME}/checkpoint
LOG_DIR=${SAVE_NAME}/logs

# Model setting
MODEL_NAME=${NET_NAME},${NET_NAME}
SPLIT_NAME=train

# Run training.
CUDA_VISIBLE_DEVICES=1 python train_my_classifier.py \
    --dataset_name=${DATASET_NAME}\
    --split_name=${SPLIT_NAME} \
    --dataset_dir=${DATASET_DIR} \
    --checkpoint_dir=${CKPT_DIR} \
    --log_dir=${LOG_DIR} \
    --model_name=${MODEL_NAME} \
    --preprocessing_name=${NET_NAME} \
    --max_number_of_steps=180000 \
    --ckpt_steps=5000 \
    --batch_size=128 \
    --num_classes=100 \
    --optimizer=adam \
    --learning_rate=0.0016 \
    --adam_beta1=0.5 \
    --opt_epsilon=1e-8 \
    --label_smoothing=0.1 \
    --num_networks=2
