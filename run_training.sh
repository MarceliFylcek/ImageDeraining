#!/bin/bash

: '
FOLDER STRUCTURE

A_path/
    train/
    test/
    ...
B_path
    train/
    test/
    ...
AB_output_path(aligned images)/
    train/
    test/
    ...
'

# PATHS
model_path="."
A_path="/home/marceli/Documents/Vident-Lens/pix2pix/dataset/A"
B_path="/home/marceli/Documents/Vident-Lens/pix2pix/dataset/B"
AB_output_folder="/home/marceli/Documents/Vident-Lens/pix2pix/dataset_merged"
model_name="first_model"
venv_path="./venv"

# OPTIONS
build_dataset=false
train=false
test=true

source venv/bin/activate
echo "Environment activated"

#Make side-by-side (aligned) images
if [ "$build_dataset" = true ]
then
    echo "Building dataset"
    python $model_path/datasets/combine_A_and_B.py \
        --fold_A $A_path \
        --fold_B $B_path \
        --fold_AB $AB_output_folder \
        #--use_AB

fi

if [ "$train" = true ]
then
python $model_path/train.py \
    --dataroot $AB_output_folder \
    --lambda_A 0.1 \
    --dataset_mode aligned \
    --name $model_name \
    --direction BtoA \
    --display_id -1 \
    --n_epochs 20 \
    --n_epochs_decay 5 \
    --preprocess scale_width \
    --load_size 512 \
    --dataset_mode aligned \
    --checkpoints_dir $model_path/checkpoints
fi


if [ "$test" = true ]
then
    python test.py  \
    --dataroot $AB_output_folder \
    --name $model_name  \
    --dataset_mode aligned \
    --load_size 512 \
    --direction BtoA \
    --preprocess scale_width
fi