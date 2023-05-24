#!/bin/bash

ROOT_DIR="/home/zhenhuanliu/Desktop/CVproj_ngp_chroma/ngp_chroma_project/data/nerf_synthetic"



for std in 0.05 0.1 0.2 0.3 0.4
do
for data_name in "lego"  "drums" "ficus" "hotdog" "chair" "materials" "mic" "ship"
do
python  train.py --adjust_view_appearance 1 \
    --root_dir $ROOT_DIR/$data_name \
    --exp_name $data_name  \
    --num_epochs 50 --batch_size 16384 --lr 1e-2 --eval_lpips \
    --chroma_std $std \
     2>&1 | tee -a $ROOT_DIR/$data_name.log

python  train.py --adjust_view_appearance 0 \
    --root_dir $ROOT_DIR/$data_name \
    --exp_name $data_name  \
    --num_epochs 50 --batch_size 16384 --lr 1e-2 --eval_lpips \
    --chroma_std $std \
     2>&1 | tee -a $ROOT_DIR/$data_name.log

done
done