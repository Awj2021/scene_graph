#!/bin/bash
set -e
echo "Start Training"
# Runing with the format likes VidVRD.
# CUDA_VISIBLE_DEVICES=1 python tools/train_net_step_rel.py --dataset chaos --cfg configs/chaos/chaos_res101xi3d50_all_boxes_sample_train_flip_dc5_2d_new.yaml --nw 8 --use_tfboard --disp_interval 2 --o SGD --lr 0.025
 
# Running with the format likes AG.
CUDA_VISIBLE_DEVICES=1 python tools/train_net_step_rel.py --dataset chaos --cfg configs/chaos/chaos_res101xi3d50_dc5_2d.yaml --nw 8 --use_tfboard --disp_interval 20 --o SGD --lr 0.01
