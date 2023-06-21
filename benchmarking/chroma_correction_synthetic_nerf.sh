#!/bin/bash

ROOT_DIR="/home/zhenhuanliu/Desktop/CVproj_ngp_chroma/ngp_chroma_project/data/nerf_synthetic"


dash="__"

prefix="synthetic_nerf_"

#python gpu_wait.py

for std in 0.1 0.05 0.01
do
#for data_name in "lego" # "drums" "ficus" "hotdog" "chair" "materials" "mic" "ship"
for data_name in "lego"  "drums" "ficus" "hotdog" "chair" "materials" "mic" "ship"

do


for adjust_ in 1 0
do
python  train.py --adjust_view_appearance $adjust_ \
    --root_dir $ROOT_DIR/$data_name \
    --exp_name $prefix$data_name  \
    --num_epochs 20 --batch_size 4096 --lr 1e-2 --eval_lpips \
    --chroma_std $std \
     2>&1 | tee -a $prefix$data_name$std$dash$adjust_.log
done


done
done