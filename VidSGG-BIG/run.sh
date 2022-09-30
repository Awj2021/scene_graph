#!/bin/bash

# Dataset VidVRD. the VidVRD has no grounding stage as the dataset have more background environment.
# Have finished. 
CUDA_VISIBLE_DEVICES=1 python tools/train_vidvrd.py --cfg_path experiments/exp3/config_.py --save_tag retrain
