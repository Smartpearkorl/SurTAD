import os
import json
import numpy as np
import torch
import platform

from random import randint
from copy import deepcopy
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from pytorchvideo import transforms as T
import torch.nn.functional as F
from torch.utils.data import DataLoader
import re
from runner.src.dataset.data_utils import RandAugmentor , PlainAugmentor

anomalies = [
    'start_stop_or_stationary',  # 1 ST
    'moving_ahead_or_waiting',  # 2 AH
    'lateral',  # 3 LA
    'oncoming',  # 4 OC
    'turning',  # 5 TC
    'pedestrian',  # 6 VP
    'obstacle',  # 7 VO
    'leave_to_right',  # 8 OO-r
    'leave_to_left',  # 9 OO-l
    'unknown',  # 10 UK
    'out_of_control', # 11 OO Combine from OO-r and OO-l
]


def read_file(path ,type = 'rgb'):
    if type == 'rgb':
        return np.asarray(Image.open(path))
    elif type == 'npy':
        return np.load(path)
    else:
        raise Exception(f'unsupported file type {type}')


def has_objects(ann):
    return sum([len(labels['objects']) for labels in ann['labels']]) != 0


class AnomalySubBatch(object):
    def __init__(self, dota, index):
        key = dota.keys[index]
        num_frames = dota.metadata[key]['num_frames']
        self.begin, self.end = dota._get_random_subbatch(num_frames,index)
        # negative case
        if self.end >= dota.metadata[key]['anomaly_start'] and \
                self.begin <= dota.metadata[key]['anomaly_end']:
            self.label = 1
            self.a_start = max(
                0, dota.metadata[key]['anomaly_start'] - self.begin
            )
            self.a_end = min(
                dota.metadata[key]['anomaly_end'] - self.begin,
                self.end - self.begin
            )
        else:
            self.label = -1
            self.a_start = -1
            self.a_end = -1

def pad_collate(batch):
    video_data, data_info, yolo_boxes, frames_boxes , video_name = zip(*batch)
    max_length = max([video.shape[0] for video in video_data])
    padded_videos = []
    for i, video in enumerate(video_data):
        pad_size = max_length - video.shape[0]
        if pad_size!=0:
            # Padding video with zeros along the time dimension (T)
            padded_video = F.pad(video, (0, 0, 0, 0, 0, 0, 0, pad_size))
            padded_videos.append(padded_video)
            # Padding yolo boxes
            yolo_boxes[i].extend([np.empty((0,4)) for _ in range(pad_size)])
            frames_boxes[i].extend([np.empty((0,4)) for _ in range(pad_size)])
        else:
            padded_videos.append(video)
    # Stack the padded videos into a single tensor
    batch_videos = torch.stack(padded_videos)
    data_info = [torch.tensor(info) for info in data_info]
    data_info = torch.stack(data_info)
    return batch_videos, data_info, yolo_boxes, frames_boxes, video_name

class Dota(Dataset):
    def __init__(
            self, root_path, phase, 
            pre_process_type = 'rgb', # rgb or sam_emb or dino_emb
            trans_cfg = {},
            VCL=None,
            local_rank=0,
            sorted_num_frames=False,
            data_type='',
            **arg, #多余参数冗余
            ):
        self.root_path = root_path
        self.phase = phase  # 'train', 'test', 'play'
        self.pre_process_type = pre_process_type  # 'rgb' , 'emb'
        # data augmentation
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
        self.sorted_num_frames=sorted_num_frames # when testing, close num_frames is good for padding
        self.data_type = data_type # size of data: '', 'mini_','sub_' ,  
        self.get_data_list()
        # self.split_meta_data(1)
        self.video_clip_begin = [0] * len(self.metadata)

    # split meta data into chunks for multiple gpus
    def split_meta_data(self, choose_chunk):
        chunks = 2
        items = list(self.metadata.items())
        chunk_size = len(items) // chunks
        slpit_metadata = [dict(items[i:i + chunk_size]) for i in range(0, len(items), chunk_size)]
        if len(slpit_metadata) == chunks + 1: # append last 
            slpit_metadata[chunks-1].update(slpit_metadata[chunks])
        self.metadata = slpit_metadata[choose_chunk]
        self._load_anns()


    def get_data_list(self):
        list_file = os.path.join(
            self.root_path, 'metadata', '{}metadata_{}.json'.format(self.data_type , self.phase))
        assert os.path.exists(list_file), \
            "File does not exist! %s" % (list_file)
        with open(list_file, 'r') as f:
            self.metadata = json.load(f)
        if self.sorted_num_frames:
            # 按照 num_frames 对应的值排序
            self.metadata = dict(sorted(self.metadata.items(), key=lambda item: item[1]['num_frames'],reverse=True))

        # load annotations
        self._load_anns()
        # clean wrong metadata and reload
        self._filter_wrong_metadata()
        self._load_anns()
        # self._cut_off_data(10)

    # debug
    def _cut_off_data(self, keep_size = 10):
        keep_scenes = list(self.metadata.keys())[:keep_size]
        self.metadata = {x:self.metadata[x] for x in keep_scenes}

    def _load_anns(self):
        self.keys = list(self.metadata.keys())
        self.annotations = []
        for key in self.keys:
            self.annotations.append(self._load_ann(key))
    
    def _filter_wrong_metadata(self):
        # check if #ann != #images
        metadata = deepcopy(self.metadata)
        for index in range(len(self.metadata)):
            ann = self.annotations[index]
            video_file = self.keys[index]
            frames_dir = os.path.join(
                self.root_path, 'frames', video_file, 'images')
            count_files = len(os.listdir(frames_dir))
            count_ann = len(ann['labels'])
            count_meta = self.metadata[video_file]['num_frames']
            if count_ann != count_files or count_files != count_meta \
                    or count_ann != count_meta:
                # remove wrong!
                del metadata[video_file]
                #  del self.annotations[index]
        print('removed {} videos'.format(len(self.metadata) - len(metadata)))
        self.metadata = metadata

    def _load_ann(self, key):
        ann_file = os.path.join(
            self.root_path, 'annotations', '{}.json'.format(key))
        with open(ann_file, 'r') as f:
            ann = json.load(f)
        return ann

    def __len__(self):
        return len(self.metadata)

    def _get_random_subbatch(self, count, index):
        if self.VCL is None:
            return 0, count
        else:
            # if video is small then VCL, return full video
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
    
    # get yolov9 boxes
    def get_yolo_boxes(self, video_name, sub_batch):
        boxes_dir = os.path.join(self.root_path, "yolov9",video_name+'.json')
        with open(boxes_dir, 'r', encoding='utf-8') as file:
            yolov9 = json.load(file)
        yolo_boxes = []
        for frame_data in yolov9['lables'][sub_batch.begin:sub_batch.end]:
            # 如果当前帧没有物体返回的是空array([],dtype=float64)
            boxes = [ obj['bbox'] for obj in frame_data['objects'] ]
            yolo_boxes.append(np.array(boxes))  
        yolo_boxes = self._add_box_filer(yolo_boxes)      
        return yolo_boxes

    def get_gt_boxes(self, index, sub_batch):
        ann = self.annotations[index]
        gt_labels = ann['labels']
        # gt_box_count = [len(gt_labels[i]['objects']) for i in range(frame_begin,frame_end)]
        frames_boxes = []
        for frame_data in gt_labels[sub_batch.begin:sub_batch.end]:
            # 如果当前帧没有物体返回的是空array([],dtype=float64)
            boxes = [ obj['bbox'] for obj in frame_data['objects'] ]
            frames_boxes.append(np.array(boxes))
        frames_boxes = self._add_box_filer(frames_boxes)    
        return frames_boxes
    

    def load_video_data(self, index, sub_batch):
        video_name = self.keys[index]
        ano_begin, ano_end = self.metadata[video_name]['anomaly_start'] ,self.metadata[video_name]['anomaly_end']
        label = sub_batch.label

        """ read rgb  or vit embedding  """
        if self.pre_process_type == 'rgb':
            frames_dir = os.path.join(self.root_path, 'frames', video_name, 'images')
            frames = sorted(os.listdir(frames_dir))
            names = [os.path.join(frames_dir, name) for name in frames[sub_batch.begin:sub_batch.end]]
            frames = np.array(list(map(read_file, names)))
            video_len_orig = len(frames)
            # in case video count less than VCL
            frames = self._add_video_filler(frames)
            return frames, video_len_orig
            # return frames.astype('float32'), video_len_orig
        else:
            raise Exception(f'unsupported type {self.pre_process_type=}')

    def gather_info(self, index, sub_batch, video_len_orig):
        ann = self.annotations[index]
        label = sub_batch.label
        return np.array([
            video_len_orig,
            self.keys.index(ann['video_name']),
            sub_batch.a_start,
            sub_batch.a_end,
            label,
            # min value because of reward formula
            sub_batch.begin,
            sub_batch.end,
            ann['accident_id'],
            int(ann['ego_involve']),
            int(ann['night']),
            int(has_objects(ann)),
        ]).astype('float'),ann['video_name']

    def __getitem__(self, index):
        # init sub_batch
        sub_batch = AnomalySubBatch(self, index)

        # load video data : rgb or sam-vit embedding
        video_data, video_len_orig = self.load_video_data(index, sub_batch)
        
        # gather info
        data_info , video_name = self.gather_info(index, sub_batch, video_len_orig)

        # get gt boxes
        frames_boxes = self.get_gt_boxes(index, sub_batch)

        # get yolov9 boxes
        yolo_boxes = self.get_yolo_boxes(video_name,sub_batch)

        video_data, yolo_boxes = self.trans(video_data,yolo_boxes)

        return video_data, data_info, yolo_boxes, frames_boxes , video_name

