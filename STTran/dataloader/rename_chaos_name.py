""" This Code is inspired by the TRACE. """
import os
import h5py
import pickle
import json
import numpy as np
import math
from tqdm import tqdm
import copy
from shutil import copyfile
from collections import OrderedDict
import ipdb
import argparse

def get_name_mapping(video_id, frame_id, cnt):
    mapped_name = '{:012d}'.format(cnt) + '.png'
    return mapped_name

def get_class_id(class_list):
    ans_dict = {}
    for i, c in enumerate(class_list):
        ans_dict[c] = i
    return ans_dict
    
def obj_class2int(s, obj_dict):
    ans = obj_dict[s]
    return ans

def pred_class2int(s, pred_dict):
    ans = pred_dict[s]
    return ans

# [ymin, ymax, xmin, xmax] to [x, y, w, h]
def box_transform(box):
    x = box[2]
    y = box[0]
    w = box[3] - box[2] + 1
    h = box[1] - box[0] + 1
    return [x, y, w, h]


def txt2json(path, txt_path, json_path):
    # with open(txt_path, 'r') as f:
    #     s = f.read().split()
    #     f.close()
    # because not every predicate has only one word.
    with open(txt_path, 'r') as f:
        s = f.readlines()
        s = [x.strip() for x in s]
        f.close()
    # TODO: 如果要加入background的话，需要提前加入
    if '__background__' not in s:
        s.append('__background__')

    with open(json_path, 'w') as fj:
        fj.write(json.dumps(s))
        fj.close()
    print(f'=== number of predicates: {len(s)} \n', s)
    return s

def get_box_from_track(tracklet_clip, obj_list, tid, obj_class_list, video_id):
    flg = False
    if tid >= len(obj_list) or tid != obj_list[tid]['tid']:
        #print('tid != obj_list[tid][\'tid\'] ! {} != {}'.format(tid, obj_list[tid]['tid']))
        for i in obj_list:
            if tid == i['tid']:
                category = i['category']
                flg = True
                break
    else:
        flg = True
        category = obj_list[tid]['category']
    if flg is not True:
        assert False
    x1 = int(tracklet_clip['bbox']['xmin'])
    y1 = int(tracklet_clip['bbox']['ymin'])
    x2 = int(tracklet_clip['bbox']['xmax'])
    y2 = int(tracklet_clip['bbox']['ymax'])
    ans = {"category": obj_class2int(category, obj_class_list), 
            "bbox": [y1, y2, x1, x2]}
    return ans

def filter_files(out_split):
    """ 
    Filter some json file that have some error
    These videos have serious error that cannot be processed. 
    """
    file_list_train = ['AZGKQCZJ.json', 'AZGKQCZJ.json']
    file_list_val = ['AZGKQCZJ.mp4','BQSEQOVM.mp4','DATDLDDI.mp4','LEZNSMQH.mp4',
        'NEJWHHON.mp4','PVJUWOUG.mp4','UFFQBBPJ.mp4','UUCINGKC.mp4']
    
    file_list = file_list_train if out_split == 'train' else file_list_val

    files = [x.split('.')[0] + '.json' for x in file_list]
    return files


def check_data_files(anno_dir = '', out_split='train', check_result = '../check_info.json'):
    """Before transferring the json file, do some check for quicker process"""
    # TODO: Check the keywords of every json file.
    anno_keys = set(['video_id', 'video_id_original', 'frame_count', 
        'fps', 'height', 'width', 'trajectories',
        'subject/objects', 'relationship_instances'])
    # anno_dir = 'data/chaos/annotations/'
    init_path = os.path.join(anno_dir, out_split)

    check_info = []
    for file in os.listdir(init_path):
        with open(os.path.join(init_path, file), 'r') as f:
            anno_data = json.load(f)
        if set(list(anno_data.keys())) != anno_keys:
            if len(list(anno_data.keys())) < len(list(anno_keys)):
                info = f'Missing keys {anno_keys - set(list(anno_data.keys()))}'
            else:
                info = f'Keys difference: {set(list(anno_data.keys())).difference(anno_keys)}'
            print(info)
            check_info.append({f'{file}': info})
    
        # TODO: Check the length of trajectories with the frame count.len(trajectory) == frame_count.
        if anno_data['frame_count'] != len(anno_data['trajectories']):
            info = 'frame_count : {}, length_of_trajectories: {}'.format(anno_data['frame_count'], len(anno_data['trajectories']))
            check_info.append({f'{file}': f'{info}'})
        
        # TODO: check the trajectory that cannot be None.
        content_trajectory = [x for x in anno_data['trajectories'] if x != []]
        if len(content_trajectory) == 0:
            info = f'Trajectory is None'
            check_info.append({f'{file}': f'{info}'})

        # assert len(content_trajectory) == 0, (
        #     'Trajectories cannot be None in {}'.format(file))

    with open(check_result, 'w') as f:
        json.dump(check_info, f)

    # assert len(check_info) == 0, (
    #     'Please check the annotations!!! at {}'.format(check_result)
    # )

def modify(json_file):
    """
    Args:
        json_file (str): the dir of json file. e.g. dir/train/
    return:
        the correct json file.
    """
    with open(json_file, 'r') as f:
        data = json.load(f)
    f.close()
    traj_new = []
    for item in data['trajectories']:
        if type(item) == list:
            if item != [] and type(item[0]) != dict:
                traj_new.append(item[0])
                print(json_file.split('/')[-1])
            else:
                traj_new.append(item)
        elif type(item) == dict:
            traj_new.append([item])
        else:
            raise TypeError(f'Unknow type {type(item)}')
    assert len(traj_new) == len(data['trajectories']), (
        print('Different length!'))
    data['trajectories']=traj_new
    
    # with open(json_file, 'w') as f:
    #     json.dump(data, f)


def process_vrd_split(pred_class_list, obj_class_list, out_split='train', anno_dir = ''):
    # anno_dir = 'data/chaos/annotations/'
    # filter_file = filter_files(out_split)
    init_path = os.path.join(anno_dir, out_split)
    # video_anno_list = [x for x in os.listdir(init_path) if x not in filter_file]
    video_anno_list = os.listdir(init_path)

    pred_class_list = get_class_id(pred_class_list)  # arrage the index of every item.
    obj_class_list = get_class_id(obj_class_list)    
    
    cnt = 1
    name_map = {}
    name_list = [' ', ]
    
    new_anns = dict()
    size_dict = dict()

    # print('Start Checking the Annotation!')
    # for video in tqdm(video_anno_list):
    #     modify(os.path.join(init_path, video))

    for video_anno_file in tqdm(video_anno_list):       # loop for every .json file
        # print(video_anno_file)
        # video_pure_id = video_anno_file.split('_')[-1].split('.')[0]

        with open(os.path.join(init_path, video_anno_file), 'r') as f:
            relation_anns = json.load(f)
            f.close()
        video_id = relation_anns['video_id']
        # frame_count = relation_anns['frame_count']
        # fps = relation_anns['fps']
        w = relation_anns['width']
        h = relation_anns['height']
        obj_list = relation_anns['subject/objects']
        tracks_list = relation_anns['trajectories']  # [[{}, {}, {}]]
        rel_list = relation_anns['relationship_instances']

        sgg = OrderedDict()
        for rel in rel_list:   # loop for relationship
            s_tid = int(rel['subject_tid'])
            o_tid = int(rel['object_tid'])
            pred = pred_class2int(rel['predicate'], pred_class_list)
            st = rel['begin_fid']
            ed = rel['end_fid']
            for t in range(st, ed):  
                # ipdb.set_trace()
                triplet_info = dict(predicate=pred)
                # print(t)
                # cur_frame_tracks_list = tracks_list[t]  # choose the corresponding frame.
                
                # frame == t; then choose together.
                cur_frame_tracks_list = [x for x in tracks_list if int(x['frame'])==int(t)]
                
                assert cur_frame_tracks_list != [], (
                    f'Please Check the annotation which has no matching frame!!!'
                )
                # ipdb.set_trace()
                for tracklet_clip in cur_frame_tracks_list: # for every frame.
                    assert type(tracklet_clip) == dict, (
                        print(f'Checking the format of {tracklet_clip} (Should be dict), (Now: {type(tracklet_clip)})')
                    ) 
                    if tracklet_clip['tid'] == s_tid:
                        triplet_info['subject'] = get_box_from_track(tracklet_clip, 
                                                    obj_list, s_tid, obj_class_list, video_id)
                    if tracklet_clip['tid'] == o_tid:
                        triplet_info['object'] = get_box_from_track(tracklet_clip, 
                                                    obj_list, o_tid, obj_class_list, video_id)
                if t in sgg:
                    sgg[t].append(triplet_info)
                else:
                    sgg[t] = [triplet_info, ]

        for t, val in sgg.items():
            mapped_name = get_name_mapping(video_id, t, cnt) # cnt用在编码new_annotations_.json的key中；
            # print(mapped_name)
            # new_anns[mapped_name] = val
            f_frames_path = video_id + '.mp4' + '/' + '{:06d}'.format(t) + '.png' # t is the frame number.
            new_anns[f_frames_path] = val
            size_dict[mapped_name] = (h, w)
            

            name_map[f_frames_path] = cnt   # 按顺序进行编号, mapped_name包含了cnt.
            name_list.append(f_frames_path)
            cnt += 1
    
    
    if out_split == 'test': out_split = 'val'
    
    with open('/home/aiwenjie/code/STTran/data/annotations/new_annotations_' + out_split + '.json', 'w') as outfile:
        json.dump(new_anns, outfile)
        outfile.close()
    
    name_map_fname = '/home/aiwenjie/code/STTran/data/annotations/%s_fname_mapping.json' %(out_split)
    with open(name_map_fname, 'w') as f:
        json.dump(name_map, f, sort_keys=True, indent=4)
        f.close()
    name_list_fname = '/home/aiwenjie/code/STTran/data/annotations/%s_fname_list.json' %(out_split)
    with open(name_list_fname, 'w') as f:
        json.dump(name_list, f, sort_keys=True, indent=4)
        f.close()

    # STT中不需要这部分；
    # convert_anno(out_split, new_anns, obj_class_list, size_dict)
    
def convert_anno(split, vrd_anns, obj_dict, size_dict):    
    print(len(vrd_anns)) #val: 45315; train: 75345
    new_imgs = []
    new_anns = []
    ann_id = 1
    for f, anns in tqdm(vrd_anns.items()): 
        im_h, im_w = size_dict[f]
        
        image_id = int(f.split('.')[0])  
        new_imgs.append(dict(file_name=f, height=im_h, width=im_w, id=image_id))
        # used for duplicate checking
        bbox_set = set()
        for ann in anns:
            # "area" in COCO is the area of segmentation mask, while here it's the area of bbox
            # also need to fake a 'iscrowd' which is always 0
            # print(ann)
            
            s_box = ann['subject']['bbox']
            bbox = box_transform(s_box)
            if not tuple(bbox) in bbox_set:
                bbox_set.add(tuple(bbox))
                area = bbox[2] * bbox[3]
                cat = ann['subject']['category']
                new_anns.append(dict(area=area, bbox=bbox, category_id=cat, id=ann_id, image_id=image_id, iscrowd=0))
                ann_id += 1
            o_box = ann['object']['bbox']

            bbox = box_transform(o_box)
            if not tuple(bbox) in bbox_set:
                bbox_set.add(tuple(bbox))
                area = bbox[2] * bbox[3]
                cat = ann['object']['category']
                new_anns.append(dict(area=area, bbox=bbox, category_id=cat, id=ann_id, image_id=image_id, iscrowd=0))
                ann_id += 1

    new_objs = []
    for obj, i in obj_dict.items():
        new_objs.append(dict(id=i, name=obj, supercategory=obj))


    new_data = dict(images=new_imgs, annotations=new_anns, categories=new_objs)
    ipdb.set_trace()
    with open('/home/aiwenjie/code/STTran/data/annotations/detections_' + split + '.json', 'w') as outfile:
        json.dump(new_data, outfile)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser("Set Data of STT", add_help=False)
    parser.add_argument("--path_anno", default="/home/aiwenjie/code/STTran/data/annotations", type=str)
    parser.add_argument("--path_json", default="/home/aiwenjie/dataset/chaos/annotations", type=str)
    args = parser.parse_args()

    path = args.path_anno
    path_json = args.path_json
    obj_txt_path = os.path.join(path, 'object.txt')
    obj_json_path = os.path.join(path, 'objects.json')
    pred_txt_path = os.path.join(path, 'predicate.txt')
    pred_json_path = os.path.join(path, 'predicates.json')
    if not os.path.exists(obj_json_path):
        obj_class_list = txt2json(path, obj_txt_path, obj_json_path) #35
    else:
        with open(obj_json_path, 'r') as f:
            obj_class_list = json.load(f)
            f.close()
        
    if not os.path.exists(pred_json_path):
        pred_class_list = txt2json(path, pred_txt_path, pred_json_path) #132
    else:
        with open(pred_json_path, 'r') as f:
            pred_class_list = json.load(f)
            f.close()
    # TODO: ====   Check the info in advance.
    check_data_files(anno_dir=path_json, out_split='train', check_result='check_info.json')
    rel_train_new_anno_json_path = os.path.join(path, 'new_annotations_train.json')
    if not os.path.exists(rel_train_new_anno_json_path):
        process_vrd_split(pred_class_list, obj_class_list, out_split='train', anno_dir = path_json)
    
    rel_val_new_anno_json_path = os.path.join(path, 'new_annotations_val.json')
    if not os.path.exists(rel_val_new_anno_json_path):
        process_vrd_split(pred_class_list, obj_class_list, out_split='val', anno_dir = path_json)

    rel_test_new_anno_json_path = os.path.join(path, 'new_annotations_test.json')
    if not os.path.exists(rel_test_new_anno_json_path):
        process_vrd_split(pred_class_list, obj_class_list, out_split='test', anno_dir = path_json) 
