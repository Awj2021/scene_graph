#!/bin/bash
set -e
source ~/anaconda3/bin/activate trace1

if [ $? -eq 0 ];then
    echo "Activate the conda environment of TRACE successfully."
else
    echo "Cannot activate the environment of Conda."
fi

