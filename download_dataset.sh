#!/bin/bash

DATASETS_DIR='datasets'
MINIBALLS_DATASET='mini_balls'
MINIBALLS_SEQ_DATASET='mini_balls_seq'

set -euo pipefail

echo "############# DOWNLOADING DATASET(S) #############"


# Download task 1 dataset (miniballs)
mkdir -p ./$DATASETS_DIR/$MINIBALLS_DATASET
cd ./$DATASETS_DIR/$MINIBALLS_DATASET
echo "> #1: Downloading counterfactual ball detection dataset..."
curl https://perso.liris.cnrs.fr/christian.wolf/teaching/deeplearning/data/train.tgz --output ./data.tgz
tar -zxvf ./data.tgz -C ./
rm ./data.tgz
mv ./train/* ./
rm -r ./train/
cd -

# Download task 2 dataset (miniballs_seq)
echo "> #2: Downloading counterfactual ball position forecasting dataset..."
mkdir -p ./$DATASETS_DIR/$MINIBALLS_SEQ_DATASET
cd ./$DATASETS_DIR/$MINIBALLS_SEQ_DATASET
curl https://perso.liris.cnrs.fr/christian.wolf/teaching/deeplearning/data/train_seq.tgz --output ./data.tgz
tar -zxvf ./data.tgz -C ./
rm ./data.tgz
cd -

echo "############# DONE #############"
