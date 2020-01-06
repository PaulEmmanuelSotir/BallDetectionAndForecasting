#!/bin/bash

DATASETS_DIR='datasets'
MINIBALLS_DATASET='mini_balls'
MINIBALLS_SEQ_DATASET='mini_balls_seq'

set -euxo pipefail

echo "############# DOWNLOADING DATASET(S) #############"

mkdir -p ./$DATASETS_DIR/$MINIBALLS_DATASET

# Download task 1 dataset (miniballs)
echo "> #1: Downloading counterfactual ball detection dataset..."
curl https://perso.liris.cnrs.fr/christian.wolf/teaching/deeplearning/data/train.tgz -o ./$DATASETS_DIR/$MINIBALLS_DATASET/data.tgz
tar -zxvf ./$DATASETS_DIR/$MINIBALLS_DATASET/data.tgz -C ./$DATASETS_DIR/$MINIBALLS_DATASET/
rm ./$DATASETS_DIR/$MINIBALLS_DATASET/data.tgz
mv ./$DATASETS_DIR/$MINIBALLS_DATASET/train/* ./$DATASETS_DIR/$MINIBALLS_DATASET/
rm -r ./$DATASETS_DIR/$MINIBALLS_DATASET/train/

# Download task 2 dataset (miniballs_seq)
echo "> #2: Downloading counterfactual ball position forecasting dataset..."
curl https://perso.liris.cnrs.fr/christian.wolf/teaching/deeplearning/data/train_seq.tgz -o ./$DATASETS_DIR/$MINIBALLS_SEQ_DATASET/data.tgz
tar -zxvf ./$DATASETS_DIR/$MINIBALLS_SEQ_DATASET/train.tgz -C ./$DATASETS_DIR/$MINIBALLS_SEQ_DATASET/
rm ./$DATASETS_DIR/$MINIBALLS_SEQ_DATASET/data.tgz
mv ./$DATASETS_DIR/$MINIBALLS_SEQ_DATASET/train/* ./$DATASETS_DIR/$MINIBALLS_SEQ_DATASET/
rm -r ./$DATASETS_DIR/$MINIBALLS_SEQ_DATASET/train/

echo "############# DONE #############"
