# The number of cards for eval should be same as the testing.
set -e
source ~/anaconda3/bin/activate trace1

if [ $? -eq 0 ];then
    echo "Activate the conda environment of TRACE successfully."
else
    echo "Cannot activate the environment of Conda."
fi

# TODO: Testing has 3 steps. So don't forget the every step.

# evaluation for detected boxes, Recalls (SGDet)
CUDA_VISIBLE_DEVICES=0 python tools/test_net_rel.py --dataset chaos --cfg configs/chaos/chaos_res101xi3d50_all_boxes_sample_train_flip_dc5_2d_new.yaml --load_ckpt Outputs/chaos_res101xi3d50_all_boxes_sample_train_flip_dc5_2d_new/Oct18-18-50-55_sutd-AS-5014A-TT_step_with_prd_cls_v3/ckpt/model_step8667.pth --output_dir Outputs/chaos_new101 --do_val
