#!/bin/bash




python train.py --root_dir "data/nerf_synthetic/lego" --exp_name lego_demo \
 --adjust_view_appearance 1 \
 --chroma_std 0.1


python train.py --root_dir "data/nerf_synthetic/lego" --exp_name lego_demo \
 --adjust_view_appearance 0 \
 --chroma_std 0.1