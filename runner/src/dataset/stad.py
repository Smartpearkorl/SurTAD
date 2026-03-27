import os
import json
import numpy as np
import torch
import platform
from random import randint
from copy import deepcopy
from PIL import Image
from torch.utils.data import Dataset
import torch.nn.functional as F
from torch.utils.data import DataLoader
from runner.src.dataset.data_utils import RandAugmentor, PlainAugmentor

# 1. 重新定义 STAD 的异常类别
stad_anomalies = [
    "Collision:car2car", "Collision:car2bike", "Collision:car2person",
    "Collision:car2large", "Collision:large2large", "Collision:large2vru",
    "Collision:bike2bike", "Collision:bike2person", "Collision:obstacle",
    "Rollover", "Collision:others", "Unknown"
]

# 建立 accident_name 到数值 id 的映射字典
# normal 为 0, 其他异常类别为 1-12
STAD_ANOMALY_TO_ID = {name: i + 1 for i, name in enumerate(stad_anomalies)}
STAD_ANOMALY_TO_ID['normal'] = 0

def read_file(path, type='rgb'):
    if type == 'rgb':
        return np.asarray(Image.open(path))
    elif type == 'npy':
        return np.load(path)
    else:
        raise Exception(f'unsupported file type {type}')

def has_objects(ann):
    # 如果 STAD 完全没有 objects，这里会稳定返回 False(0)
    return sum([len(labels.get('objects', [])) for labels in ann['labels']]) != 0

class AnomalySubBatch(object):
    def __init__(self, dataset, index):
        key = dataset.keys[index]
        num_frames = dataset.metadata[key]['num_frames']
        self.begin, self.end = dataset._get_random_subbatch(num_frames, index)
        # negative case
        if self.end >= dataset.metadata[key]['anomaly_start'] and \
                self.begin <= dataset.metadata[key]['anomaly_end']:
            self.label = 1
            self.a_start = max(
                0, dataset.metadata[key]['anomaly_start'] - self.begin
            )
            self.a_end = min(
                dataset.metadata[key]['anomaly_end'] - self.begin,
                self.end - self.begin
            )
        else:
            self.label = -1
            self.a_start = -1
            self.a_end = -1

class Stad(Dataset):
    def __init__(
            self, root_path, phase, 
            pre_process_type='rgb',
            trans_cfg={},
            VCL=None,
            local_rank=0,
            sorted_num_frames=False,
            data_type='',
            **arg, 
            ):
        self.root_path = root_path
        self.phase = phase  
        self.pre_process_type = pre_process_type  
        
        self.aug_type = trans_cfg.get('aug_type','plain')
        if self.aug_type == 'plain': 
            self.trans = PlainAugmentor(**trans_cfg)
        elif self.aug_type == 'randaug':
            self.trans = RandAugmentor(**trans_cfg)  
        else:
            raise Exception(f'unsupported data augmentation type {self.aug_type}') 

        self.fps = 10
        self.VCL = VCL
        self.local_rank = local_rank
        self.sorted_num_frames = sorted_num_frames 
        self.data_type = data_type 
        self.get_data_list()
        self.video_clip_begin = [0] * len(self.metadata)

    def split_meta_data(self, choose_chunk):
        chunks = 2
        items = list(self.metadata.items())
        chunk_size = len(items) // chunks
        slpit_metadata = [dict(items[i:i + chunk_size]) for i in range(0, len(items), chunk_size)]
        if len(slpit_metadata) == chunks + 1: 
            slpit_metadata[chunks-1].update(slpit_metadata[chunks])
        self.metadata = slpit_metadata[choose_chunk]
        self._load_anns()

    def get_data_list(self):
        list_file = os.path.join(
            self.root_path, 'metadata', '{}metadata_{}.json'.format(self.data_type, self.phase))
        assert os.path.exists(list_file), "File does not exist! %s" % (list_file)
        
        with open(list_file, 'r') as f:
            self.metadata = json.load(f)
            
        if self.sorted_num_frames:
            self.metadata = dict(sorted(self.metadata.items(), key=lambda item: item[1]['num_frames'], reverse=True))

        self._load_anns()
        self._filter_wrong_metadata()
        self._load_anns()

    def _cut_off_data(self, keep_size=10):
        keep_scenes = list(self.metadata.keys())[:keep_size]
        self.metadata = {x:self.metadata[x] for x in keep_scenes}

    def _load_anns(self):
        self.keys = list(self.metadata.keys())
        self.annotations = []
        for key in self.keys:
            self.annotations.append(self._load_ann(key))
    
    def _filter_wrong_metadata(self):
        metadata = deepcopy(self.metadata)
        for index in range(len(self.metadata)):
            ann = self.annotations[index]
            video_file = self.keys[index]
            # 注意：根据你的 json，图像可能存放在 'frames/000001/' 下，而不是 'frames/000001/images/'
            # 这里为了兼容，如果是原 dota 的 /images 层级，请自行加上 'images'
            frames_dir = os.path.join(self.root_path, 'frames', video_file)
            if not os.path.exists(frames_dir): 
                frames_dir = os.path.join(self.root_path, 'frames', video_file, 'images')

            if os.path.exists(frames_dir):
                count_files = len([f for f in os.listdir(frames_dir) if f.endswith('.jpg') or f.endswith('.png')])
            else:
                count_files = 0
                
            count_ann = len(ann['labels'])
            count_meta = self.metadata[video_file]['num_frames']

            if count_ann != count_files or count_files != count_meta or count_ann != count_meta or (self.VCL!=None and count_meta<self.VCL):
                del metadata[video_file]
        print('removed {} videos'.format(len(self.metadata) - len(metadata)))
        self.metadata = metadata

    def _load_ann(self, key):
        ann_file = os.path.join(self.root_path, 'annotations', '{}.json'.format(key))
        with open(ann_file, 'r') as f:
            ann = json.load(f)
        return ann

    def __len__(self):
        return len(self.metadata)

    def _get_random_subbatch(self, count, index):
        if self.VCL is None:
            return 0, count
        else:
            if count <= self.VCL:
                return 0, count
            max_ = count - self.VCL
            begin = randint(0, max_)
            end = begin + self.VCL
            return begin, end

    def _add_video_filler(self, frames):
        try:
            filler_count = self.VCL - len(frames)
        except TypeError:
            return frames
        if filler_count > 0:
            filler = np.full((filler_count,) + frames.shape[1:], 0)
            frames = np.concatenate((frames, filler), axis=0)
        return frames

    def _add_box_filer(self, boxes):
        try:
            box_count = self.VCL - len(boxes)
        except TypeError:
            return boxes
        if box_count > 0:
            boxes.extend([np.empty((0,4)) for _ in range(box_count)])
        return boxes
    
    # 修改 1：跳过 yolo 文件的读取，直接返回空框适配网络
    def get_yolo_boxes(self, video_name, sub_batch):
        yolo_boxes = [np.empty((0,4)) for _ in range(sub_batch.begin, sub_batch.end)]
        yolo_boxes = self._add_box_filer(yolo_boxes)      
        return yolo_boxes

    # 修改 2：跳过 json 中 objects 解析（因为是空的），直接返回空框
    def get_gt_boxes(self, index, sub_batch):
        frames_boxes = [np.empty((0,4)) for _ in range(sub_batch.begin, sub_batch.end)]
        frames_boxes = self._add_box_filer(frames_boxes)    
        return frames_boxes
    
    def load_video_data(self, index, sub_batch):
        video_name = self.keys[index]
        ann = self.annotations[index]

        if self.pre_process_type == 'rgb':
            # 直接使用 json 中自带的 image_path 更加安全和稳定
            names = [
                os.path.join(self.root_path, 'frames', ann['labels'][i]['image_path']) 
                for i in range(sub_batch.begin, sub_batch.end)
            ]
            frames = np.array(list(map(read_file, names)))
            video_len_orig = len(frames)
            frames = self._add_video_filler(frames)
            return frames, video_len_orig
        else:
            raise Exception(f'unsupported type {self.pre_process_type=}')

    # 修改 3：映射 accident_name 为 accident_id
    def gather_info(self, index, sub_batch, video_len_orig):
        ann = self.annotations[index]
        label = sub_batch.label
        
        # 寻找视频对应的异常类别 (遍历 labels 找非 normal 的 label)
        accident_name = 'normal'
        for frame_lbl in ann['labels']:
            current_label = frame_lbl.get('accident_name', 'normal')
            if current_label != 'normal':
                accident_name = current_label
                break
                
        # 映射为数值ID
        accident_id = STAD_ANOMALY_TO_ID.get(accident_name, 0)
        
        # 如果原始标注中缺失这俩字段，给一个默认值 0
        ego_involve = ann.get('ego_involve', 0)
        night = ann.get('night', 0)

        return np.array([
            video_len_orig,
            self.keys.index(ann['video_name']),
            sub_batch.a_start,
            sub_batch.a_end,
            label,
            sub_batch.begin,
            sub_batch.end,
            accident_id,         # 映射后的类别 ID
            int(ego_involve),    # 默认 0
            int(night),          # 默认 0
            int(has_objects(ann)), # 默认 0 (如果没有BBox)
        ]).astype('float'), ann['video_name']

    def __getitem__(self, index):
        sub_batch = AnomalySubBatch(self, index)
        video_data, video_len_orig = self.load_video_data(index, sub_batch)
        data_info, video_name = self.gather_info(index, sub_batch, video_len_orig)
        frames_boxes = self.get_gt_boxes(index, sub_batch)
        yolo_boxes = self.get_yolo_boxes(video_name, sub_batch)
        
        # 即使 boxes 是 empty，只要你的 PlainAugmentor 不强依赖坐标，依然可以正常进行
        video_data, yolo_boxes = self.trans(video_data, yolo_boxes)

        return video_data, data_info, yolo_boxes, frames_boxes, video_name
