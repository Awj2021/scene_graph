from collections import defaultdict
import numpy as np
import json
import pickle
from tqdm import tqdm
import copy
import os

import _init_paths
from vidvrd_utils.common import voc_ap, viou
import ipdb

def get_relation_insts(anno_dir, vid, tet, no_traj=False):
    """
    get the visual relation instances labeled in a video,
    no_traj=True will not include trajectories, which is
    faster.
    """
    
    with open(os.path.join(anno_dir, tet, vid), 'r') as f:
        print(os.path.join(anno_dir, tet, vid))
        anno = json.load(f)
        f.close()
    sub_objs = dict()
    for so in anno['subject/objects']:
        sub_objs[so['tid']] = so['category']
    # trajs used for link all the trajectories, but if we have the dict, it's easier.
    if not no_traj:
        trajs = {}
        frame_ids = set(sorted([frame['frame'] for frame in anno['trajectories']])) #
        for frame in list(frame_ids): 
            bboxes = dict()
            # 先找出所有的帧
            info = [frame_info for frame_info in anno['trajectories'] if frame_info['frame']==frame]
            for bbox in info:
                # ipdb.set_trace()
                bboxes[bbox['tid']] = (bbox['bbox']['xmin'],
                                    bbox['bbox']['ymin'],
                                    bbox['bbox']['xmax'],
                                    bbox['bbox']['ymax'])
            trajs[frame] = bboxes   # 将每一帧，tid对应的bounding box保存起来
    relation_insts = []
    for anno_inst in anno['relationship_instances']:
        inst = dict()
        inst['triplet'] = (sub_objs[anno_inst['subject_tid']],
                        anno_inst['predicate'],
                        sub_objs[anno_inst['object_tid']])
        inst['subject_tid'] = anno_inst['subject_tid']
        inst['object_tid'] = anno_inst['object_tid']
        inst['duration'] = (anno_inst['begin_fid'], anno_inst['end_fid'])
        # ipdb.set_trace()
        if not no_traj: 
            # anno_inst['subject_tid']对应的segment的长度
            # inst: subject & object's trajectories. all the bounding boxes.
            # ipdb.set_trace()
            inst['sub_traj'] = [trajs[id][anno_inst['subject_tid']] for id in range(inst['duration'][0], inst['duration'][1])]
            inst['obj_traj'] = [trajs[id][anno_inst['object_tid']] for id in range(inst['duration'][0], inst['duration'][1])]

            # inst['sub_traj'] = [bboxes[anno_inst['subject_tid']] for bboxes in
            #         trajs[inst['duration'][0]: inst['duration'][1]]]
            # inst['obj_traj'] = [bboxes[anno_inst['object_tid']] for bboxes in
            #         trajs[inst['duration'][0]: inst['duration'][1]]]
        # 转换bounding boxes..

        relation_insts.append(inst)
    return relation_insts


def eval_detection_scores(gt_relations, pred_relations, viou_threshold):
    pred_relations = sorted(pred_relations, key=lambda x: x['score'], reverse=True)
    gt_detected = np.zeros((len(gt_relations),), dtype=bool)
    hit_scores = np.ones((len(pred_relations))) * -np.inf
    for pred_idx, pred_relation in enumerate(pred_relations):
        ov_max = -float('Inf')
        k_max = -1
        for gt_idx, gt_relation in enumerate(gt_relations):
            if not gt_detected[gt_idx]\
                    and tuple(pred_relation['triplet']) == tuple(gt_relation['triplet']):
                s_iou = viou(pred_relation['sub_traj'], pred_relation['duration'],
                        gt_relation['sub_traj'], gt_relation['duration'])
                o_iou = viou(pred_relation['obj_traj'], pred_relation['duration'],
                        gt_relation['obj_traj'], gt_relation['duration'])
                ov = min(s_iou, o_iou)
                if ov >= viou_threshold and ov > ov_max:
                    ov_max = ov
                    k_max = gt_idx
                    #print(pred_idx, gt_idx, ov)
        if k_max >= 0:
            hit_scores[pred_idx] = pred_relation['score']
            gt_detected[k_max] = True
    tp = np.isfinite(hit_scores)
    fp = ~tp
    cum_tp = np.cumsum(tp).astype(np.float32)
    cum_fp = np.cumsum(fp).astype(np.float32)
    rec = cum_tp / np.maximum(len(gt_relations), np.finfo(np.float32).eps)
    prec = cum_tp / np.maximum(cum_tp + cum_fp, np.finfo(np.float32).eps)
    return prec, rec, hit_scores


def eval_tagging_scores(gt_relations, pred_relations):
    pred_relations = sorted(pred_relations, key=lambda x: x['score'], reverse=True)
    # ignore trajectories
    gt_triplets = set(tuple(r['triplet']) for r in gt_relations)
    pred_triplets = []
    hit_scores = []
    for r in pred_relations:
        triplet = tuple(r['triplet'])
        if not triplet in pred_triplets:
            pred_triplets.append(triplet)
            hit_scores.append(r['score'])
    hit_scores = np.asarray(hit_scores)
    for i, t in enumerate(pred_triplets):
        if not t in gt_triplets:
            hit_scores[i] = -np.inf
    tp = np.isfinite(hit_scores)
    fp = ~tp
    cum_tp = np.cumsum(tp).astype(np.float32)
    cum_fp = np.cumsum(fp).astype(np.float32)
    rec = cum_tp / np.maximum(len(gt_triplets), np.finfo(np.float32).eps)
    prec = cum_tp / np.maximum(cum_tp + cum_fp, np.finfo(np.float32).eps)
    return prec, rec, hit_scores


def evaluate(groundtruth, prediction, viou_threshold=0.5,
        det_nreturns=[50, 100], tag_nreturns=[1, 5, 10]):
    """ evaluate visual relation detection and visual 
    relation tagging.
    """
    video_ap = dict()
    tot_scores = defaultdict(list)
    tot_tp = defaultdict(list)
    prec_at_n = defaultdict(list)
    tot_gt_relations = 0
    print('Computing average precision AP over {} videos...'.format(len(groundtruth)))
    for vid, gt_relations in groundtruth.items():
        if len(gt_relations)==0:
            continue
        tot_gt_relations += len(gt_relations)
        predict_relations = prediction.get(vid, [])
        # compute average precision and recalls in detection setting
        det_prec, det_rec, det_scores = eval_detection_scores(
                gt_relations, predict_relations, viou_threshold)
        video_ap[vid] = voc_ap(det_rec, det_prec)
        tp = np.isfinite(det_scores)
        for nre in det_nreturns:
            cut_off = min(nre, det_scores.size)
            tot_scores[nre].append(det_scores[:cut_off])
            tot_tp[nre].append(tp[:cut_off])
        # compute precisions in tagging setting
        tag_prec, _, _ = eval_tagging_scores(gt_relations, predict_relations)
        for nre in tag_nreturns:
            cut_off = min(nre, tag_prec.size)
            if cut_off > 0:
                prec_at_n[nre].append(tag_prec[cut_off - 1])
            else:
                prec_at_n[nre].append(0.)
    # calculate mean ap for detection
    mean_ap = np.mean(list(video_ap.values()))
    # calculate recall for detection
    rec_at_n = dict()
    for nre in det_nreturns:
        scores = np.concatenate(tot_scores[nre])
        tps = np.concatenate(tot_tp[nre])
        sort_indices = np.argsort(scores)[::-1]
        tps = tps[sort_indices]
        cum_tp = np.cumsum(tps).astype(np.float32)
        rec = cum_tp / np.maximum(tot_gt_relations, np.finfo(np.float32).eps)
        rec_at_n[nre] = rec[-1]
    # calculate mean precision for tagging
    mprec_at_n = dict()
    for nre in tag_nreturns:
        mprec_at_n[nre] = np.mean(prec_at_n[nre])
    # print scores
    print('detection mean AP (used in challenge): {}'.format(mean_ap))
    print('detection recall@50: {}'.format(rec_at_n[50]))
    print('detection recall@100: {}'.format(rec_at_n[100]))
    print('tagging precision@1: {}'.format(mprec_at_n[1]))
    print('tagging precision@5: {}'.format(mprec_at_n[5]))
    print('tagging precision@10: {}'.format(mprec_at_n[10]))
    return mean_ap, rec_at_n, mprec_at_n


if __name__ == "__main__":
    """
    You can directly run this script from the parent directory, e.g.,
    python -m evaluation.visual_relation_detection val_relation_groundtruth.json val_relation_prediction.json
    """
    import json
    from argparse import ArgumentParser

    parser = ArgumentParser(description='Video visual relation detection evaluation.')
    parser.add_argument('--groundtruth', 
            default="data/chaos/annotations/test_gt.json", 
            help='A ground truth JSON file generated by yourself')
    parser.add_argument('--prediction', 
            default="Outputs/ag_val_50_vidvrd/baseline_relation_prediction.json", 
            help='A prediction file')
    parser.add_argument('--is_train', 
            action='store_true', 
            help='is_test')
    parser.add_argument('--anno_dir', 
            default='/home/chaos/data/Chaos/dataset/annotation/AG_vidvrd_format/annotation',
            type=str, help='The dir of annotation which contains the single json.')
    args = parser.parse_args()
    
    if os.path.exists(args.groundtruth):
        print('Loading ground truth from {}'.format(args.groundtruth))
        with open(args.groundtruth, 'r') as fp:
            gt = json.load(fp)
            fp.close()
        print('Number of videos in ground truth: {}'.format(len(gt)))
    else:
        gt = dict()
        tet = 'train' if args.is_train else 'test'
        # TODO: the dir of videos. Need to change.
        # anno_dir = '/home/chaos/data/Chaos/dataset/annotation/activity_graph/vidvrd_format/annotations'
        init_path = os.path.join(args.anno_dir, tet)
        video_anno_list = os.listdir(init_path)  
        for i in tqdm(video_anno_list):
            vid = i.split('.')[0]
            gt[vid] = get_relation_insts(args.anno_dir, i, tet)
            # ipdb.set_trace()
        with open(args.groundtruth, 'w') as fp:
            json.dump(gt, fp)
            fp.close()

    print('Loading prediction from {}'.format(args.prediction))
    # TODO: 这里应该先要生成baseline_relation_prediction.json文件
    with open(args.prediction, 'r') as fp: 
        pred = json.load(fp)
        fp.close()
    print('Number of videos in prediction: {}'.format(len(pred['results'])))
    
    
    #res = pred['results']
    #for vid_name, l in res.items():
    #    if vid_name != 'ILSVRC2015_train_00098007': continue
    #    l = sorted(l, key=lambda x: x['score'], reverse=True)
    #    for i in l:
    #        print(i['triplet'], i['score'], i['duration'], i['sub_traj'][0], i['obj_traj'][0])
    #    print()
    #    print(len(l), vid_name)
    #    assert False
    ipdb.set_trace()
    mean_ap, rec_at_n, mprec_at_n = evaluate(gt, pred['results'])
