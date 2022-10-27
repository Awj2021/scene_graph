#!/bin/bash
set -e
source ~/anaconda3/bin/activate stt4

if [ $? -eq 0 ];then
    echo "Activate the conda environment of STT4 successfully."
else
    echo "Cannot activate the environment of Conda."
fi

# Delete the .pyc file
find ./ -name *.pyc -exec rm -rf {} \;
# TODO: Motify the $DATAPATH.
# For PredCLS:
CUDA_VISIBLE_DEVICES=1 CUDA_LAUNCH_BLOCKING=1 python train.py -mode predcls -datasize mini --cudnn True -bce_loss
# For SGCLS:
# python train.py -mode sgcls -datasize large -data_path $DATAPATH 
# For SGDET:
# python train.py -mode sgdet -datasize large -data_path $DATAPATH
