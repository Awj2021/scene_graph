# The number of cards for eval should be same as the testing.
# CUDA_VISIBLE_DEVICES=0 python tools/test_net_rel.py --dataset chaos --cfg configs/chaos/chaos_res101xi3d50_all_boxes_sample_train_flip_dc5_2d_new.yaml --load_ckpt Outputs/chaos_res101xi3d50_all_boxes_sample_train_flip_dc5_2d_new/Oct10-14-07-44_sutd-AS-5014A-TT_step_with_prd_cls_v3/ckpt/model_step2289.pth --output_dir Outputs/chaos_new101 --do_val

# python tools/transform_vidvrd_results.py --input_dir Outputs/chaos_new101 --output_dir Outputs/chaos_new101 --is_gt_traj

python tools/test_chaos.py --prediction Outputs/chaos_new101/baseline_relation_prediction.json --groundtruth data/chaos/annotations/test_gt.json
