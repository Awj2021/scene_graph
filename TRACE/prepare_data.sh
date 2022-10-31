#!/bin/bash
set -e
# Data Processing. Dataset: Chaos
# dataset rename.
# The dir of json file.
path_json=/home/chaos/data/Chaos/dataset/annotation/AG_vidvrd_format
echo "======== Start Precessing the dataset!!!"
cd ./data/chaos/annotations
# firstly delete the existing files
# cp the dict_classes into the corresponding dir.
cp $path_json/dict_AG.txt ./predicate.txt

# TODO: directly cover the train_videos_list and test_videos_list

ls $path_json/annotation/train | awk -F '.' '{print $1}' > train_videos_list.txt
ls $path_json/annotation/test | awk -F '.' '{print $1}' > test_videos_list.txt
ls $path_json/annotation/val | awk -F '.' '{print $1}' > val_videos_list.txt

rm -rf detection* new_annotations* train_fname* val_fname* 

cd ../
rm -rf frames/* sampled_frames/*

cd ../../

# rename annotations.
# change the dir of annotations.

python tools/rename_chaos_anno.py --path_json $path_json/annotation


# Dump the frames.Dump frames is time-consuming, so if the process is right, there is no need for repeating this process.
echo "***Dump sampld frames***"
python tools/dump_frames.py --ignore_editlist --video_dir /data/chaos/videos_320x180 --frame_dir data/chaos/sampled_frames --frame_list_file train_fname_list.json,val_fname_list.json --annotation_dir data/chaos/annotations --frames_store_type jpg --high_quality --sampled_frames --st_id 0 

# Dump the frames.
echo "***Dump frames***"
python tools/dump_frames.py --ignore_editlist --video_dir /data/chaos/videos_320x180 --frame_dir data/chaos/frames --frame_list_file train_fname_list.json,val_fname_list.json --annotation_dir data/chaos/annotations --st_id 0

echo "===== Dataset Processing Finished !!!"