#!bin/bash
# Dataset of VidVRD.
CUDA_VISIBLE_DEVICES=1 python tools/train_net_step_rel.py --dataset vidvrd --cfg configs/vidvrd/vidvrd_res101xi3d50_all_boxes_sample_train_flip_dc5_2d_new.yaml --nw 8 --use_tfboard --disp_interval 20 --o SGD --lr 0.025