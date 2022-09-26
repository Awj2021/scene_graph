#!/bin/bash
# Dataset of VidVRD.
# CUDA_VISIBLE_DEVICES=1 python tools/train_net_step_rel.py --dataset vidvrd --cfg configs/vidvrd/vidvrd_res101xi3d50_all_boxes_sample_train_flip_dc5_2d_new.yaml --nw 8 --use_tfboard --disp_interval 20 --o SGD --lr 0.025


######     Data Processing. Dataset: Chaos

# dataset rename.
# python tools/rename_chaos_anno.py
# python tools/dump_frames.py --ignore_editlist --video_dir data/chaos/videos --frame_dir data/chaos/sampled_frames --frame_list_file val_fname_list.json,train_fname_list.json --annotation_dir data/chaos/annotations --frames_store_type jpg --high_quality --sampled_frames --st_id 0

# Dump the frames.（only for the training dataset which includes only one json file）
# python tools/dump_frames.py --ignore_editlist --video_dir data/chaos/videos --frame_dir data/chaos/frames --frame_list_file train_fname_list.json --annotation_dir data/chaos/annotations --st_id 0

# Runing
CUDA_VISIBLE_DEVICES=1 python tools/train_net_step_rel.py --dataset chaos --cfg configs/chaos/chaos_res101xi3d50_all_boxes_sample_train_flip_dc5_2d_new.yaml --nw 8 --use_tfboard --disp_interval 20 --o SGD --lr 0.025
 
# CUDA_VISIBLE_DEVICES=1 python tools/train_net_step_rel.py --dataset vidvrd --cfg configs/vidvrd/vidvrd_res101xi3d50_all_boxes_sample_train_flip_dc5_2d_new.yaml --nw 8 --use_tfboard --disp_interval 20 --o SGD --lr 0.025
