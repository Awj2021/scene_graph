# The number of cards for eval should be same as the testing.
#!/bin/bash
set -e
source ~/anaconda3/bin/activate trace1

if [ $? -eq 0 ];then
    echo "Activate the conda environment of TRACE successfully."
else
    echo "Cannot activate the environment of Conda."
fi

# evaluation for detected boxes, Recalls (SGDet)
echo "============  Evaluating the SGDet  =================="
# CUDA_VISIBLE_DEVICES=0 python tools/test_net_rel.py --dataset chaos --cfg configs/chaos/chaos_res101xi3d50_dc5_2d.yaml --load_ckpt Outputs/chaos_res101xi3d50_dc5_2d/Oct19-20-08-31_sutd-AS-5014A-TT_step_with_prd_cls_v3/ckpt/model_step120369.pth --output_dir Outputs/chaos_new101 --do_val
# CUDA_VISIBLE_DEVICES=1 python tools/test_net_rel.py --dataset chaos --cfg configs/chaos/chaos_res101xi3d50_dc5_2d.yaml \
# --load_ckpt Outputs/chaos_res101xi3d50_dc5_2d/Oct23-23-59-58_sutd-AS-5014A-TT_step_with_prd_cls_v3/ckpt/model_step47485.pth --output_dir Outputs/chaos_new101 --do_val

# 10.30 Using the pretrained Detector.
# CUDA_VISIBLE_DEVICES=0 python tools/test_net_rel.py --dataset chaos --cfg configs/chaos/chaos_res101xi3d50_dc5_2d.yaml \
# --load_ckpt Outputs/chaos_res101xi3d50_dc5_2d/Oct30-10-32-05_sutd-AS-5014A-TT_step_with_prd_cls_v3/ckpt/model_step53759.pth --output_dir Outputs/chaos_new101 --do_val

# #  PredCLS
# echo "============  Evaluating the SGCLS  =================="
# CUDA_VISIBLE_DEVICES=0 python tools/test_net_rel.py --dataset chaos --cfg configs/chaos/chaos_res101xi3d50_dc5_2d.yaml \
# --load_ckpt Outputs/chaos_res101xi3d50_dc5_2d/Oct30-10-32-05_sutd-AS-5014A-TT_step_with_prd_cls_v3/ckpt/model_step53759.pth --do_val --use_gt_boxes

# 10.31 Using the pretrained Detector. And the last one Trained Model.
CUDA_VISIBLE_DEVICES=0 python tools/test_net_rel.py --dataset chaos --cfg configs/chaos/chaos_res101xi3d50_dc5_2d.yaml \
--load_ckpt Outputs/chaos_res101xi3d50_dc5_2d/Oct30-10-32-05_sutd-AS-5014A-TT_step_with_prd_cls_v3/ckpt/model_step107521.pth --output_dir Outputs/chaos_new101 --do_val

#  PredCLS
echo "============  Evaluating the SGCLS  =================="
CUDA_VISIBLE_DEVICES=0 python tools/test_net_rel.py --dataset chaos --cfg configs/chaos/chaos_res101xi3d50_dc5_2d.yaml \
--load_ckpt Outputs/chaos_res101xi3d50_dc5_2d/Oct30-10-32-05_sutd-AS-5014A-TT_step_with_prd_cls_v3/ckpt/model_step107521.pth --do_val --use_gt_boxes