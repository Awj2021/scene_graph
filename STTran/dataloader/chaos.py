import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import Resize, Compose, ToTensor, Normalize
import random
from scipy.misc import imread
import numpy as np
import pickle
import os
from fasterRCNN.lib.model.utils.blob import prep_im_for_blob, im_list_to_blob
import ipdb
import json
from collections import defaultdict

# TODO: 该文件中用到的.json文件都由TRACE提供的rename_chaos.py文件生成的；
class CHAOS(Dataset):

    def __init__(self, out_split, datasize, frames_path=None, annt_path=None):
        """
        subject_object_pairs: 
        videos_list;
        the frames that corresponding to the videos_list;
        """
        self.out_split = out_split    # ['train', 'test', 'val']
        self.annt_path = annt_path
        self.frames_path = frames_path
        self.object_classes = ['__background__']
        self.relationship_classes = []

        with open(os.path.join(annt_path, 'objects.json'), 'r') as f:
            self.object_classes.extend(json.load(f))
            f.close()

        with open(os.path.join(annt_path, 'predicates.json'), 'r') as f:
            self.relationship_classes = json.load(f)
            f.close()

        # self.attention_relationships = self.relationship_classes[-2:]   # watching, waving hands at.
        self.contacting_relationships = self.relationship_classes

        self.video_frames_pair = self.load_video_frames_pairs()
        self.video_list = list(self.video_frames_pair.keys())

        self.video_size = [320, 180] # (w,h)                  # TODO: 因为所有的视频大小都是固定的；
        # TODO: key: 'IXMTVXXQ.mp4/000546.png', value: 当前帧的所有的bbox info.
        gt_annotation_frames = self.load_sub_obj_pairs()  # 将原来的方式转换成字典索引的方式
        self.gt_annotation_videos = {}
        for video_id in self.video_list:
            # TODO: 等间隔抽样：抽30帧（内存经常爆炸，因此先搞10帧做测试）
            frames_list = [{key: value} for key, value in gt_annotation_frames.items() if key.split('/')[0]==video_id]
            if len(frames_list) <= 10:
                self.gt_annotation_videos[video_id] = frames_list
                continue
            else:
                indexs = np.linspace(0, len(frames_list)-1, 10, endpoint=False)
                frames_extract_list = [frames_list[int(round(index, 0))] for index in indexs]  # 四舍五入进行采样。间隔肯定大于1了。
                self.gt_annotation_videos[video_id] = frames_extract_list
        # ipdb.set_trace()
        print('x' * 60)
        print('There are {} videos and {} valid frames'.format(len(self.video_list), len(gt_annotation_frames)))
        print('x' * 60)


    def __getitem__(self, index):

        video_id = self.video_list[index]                  # 这是一个视频中的所有frames.
        frame_names = self.video_frames_pair[video_id]     # 加载出一个视频对应的所有frames.
        
        processed_ims = []
        im_scales = []

        for idx, name in enumerate(frame_names):  # 单个视频
            im = imread(os.path.join(self.frames_path, name)) # frame_path 就是/frames/
            im = im[:, :, ::-1] # rgb -> bgr
            # TODO: cfg.PIXEL_MEANS可能需要重新计算
            im, im_scale = prep_im_for_blob(im, [[[102.9801, 115.9465, 122.7717]]], 600, 1000) #cfg.PIXEL_MEANS, target_size, cfg.TRAIN.MAX_SIZE
            im_scales.append(im_scale)
            processed_ims.append(im)

        blob = im_list_to_blob(processed_ims)
        im_info = np.array([[blob.shape[1], blob.shape[2], im_scales[0]]],dtype=np.float32)
        im_info = torch.from_numpy(im_info).repeat(blob.shape[0], 1)
        img_tensor = torch.from_numpy(blob)
        img_tensor = img_tensor.permute(0, 3, 1, 2)

        gt_boxes = torch.zeros([img_tensor.shape[0], 1, 5])
        num_boxes = torch.zeros([img_tensor.shape[0]], dtype=torch.int64)
        return img_tensor, im_info, gt_boxes, num_boxes, video_id

    def __len__(self):
        return len(self.video_list)
    
    def load_video_frames_pairs(self):
        with open(os.path.join(self.annt_path, '{}_fname_list.json'.format(self.out_split)), 'r') as f:
            frames = json.load(f)
        video_frames_pair = defaultdict(list)
        for video_frame in frames:
            if len(video_frame) < 5:
                continue
            video, _  = video_frame.split('/')
            video_frames_pair[video].append(video_frame) # 不止一个video_frame.
        return video_frames_pair

    def load_sub_obj_pairs(self): 
        """
        return:  "CYIVQRLR.mp4/000559.png": 
        [{"predicate": 5, "object": {"category": 0, "bbox": [63, 173, 74, 134]}, "subject": {"category": 0, "bbox": [92, 178, 115, 187]}}]
        """
        pairs_sub_obj = {}
        with open(os.path.join(self.annt_path, 'new_annotations_{}.json'.format(self.out_split)), 'r') as f:
            images_info = json.load(f)
            f.close()
        
        with open(os.path.join(self.annt_path, '{}_fname_mapping.json'.format(self.out_split)), 'r') as f:
            key_value = json.load(f)
            f.close()
        
        new_key_value = {key:'{:012d}.png'.format(int(value)) for key, value in key_value.items()}
        # TODO: 提前进行数据结构的转换
        for key, value in new_key_value.items():
            info_one_frame = images_info[value]
            info_one_frame_trans = []
            for one_pair in info_one_frame:
                one_pair['subject']['bbox'] = np.array(one_pair['subject']['bbox'])
                one_pair['object']['bbox'] = np.array(one_pair['object']['bbox'])
                one_pair['predicate'] = torch.tensor([one_pair['predicate']], dtype=torch.long)
                info_one_frame_trans.append(one_pair)
            pairs_sub_obj[key] = info_one_frame_trans

        return pairs_sub_obj


def cuda_collate_fn_chaos(batch):
    """
    don't need to zip the tensor

    """
    return batch[0]
