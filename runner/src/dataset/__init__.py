import torch
import numpy as np
from torch.utils.data import Dataset
import torch.nn.functional as F
from torch.utils.data import DataLoader
from runner.src.dataset.dota import Dota, anomalies 
from runner.src.dataset.stad import Stad, stad_anomalies
from runner.src.dataset.data_utils import gt_cls_target

SUB_ANOMALIES = ['normal'] + stad_anomalies

def pad_collate(batch):
    # 保持不变，可以直接复用原 Dota 的 pad_collate
    video_data, data_info, yolo_boxes, frames_boxes , video_name = zip(*batch)
    max_length = max([video.shape[0] for video in video_data])
    padded_videos = []
    for i, video in enumerate(video_data):
        pad_size = max_length - video.shape[0]
        if pad_size!=0:
            padded_video = F.pad(video, (0, 0, 0, 0, 0, 0, 0, pad_size))
            padded_videos.append(padded_video)
            yolo_boxes[i].extend([np.empty((0,4)) for _ in range(pad_size)])
            frames_boxes[i].extend([np.empty((0,4)) for _ in range(pad_size)])
        else:
            padded_videos.append(video)
    batch_videos = torch.stack(padded_videos)
    data_info = [torch.tensor(info) for info in data_info]
    data_info = torch.stack(data_info)
    return batch_videos, data_info, yolo_boxes, frames_boxes, video_name


def prepare_dataset(cfg, train_data=None, test_data=None):
    train_sampler, test_sampler = None, None 
    train_shuffle, test_shuffle = True, False
    traindata_loader, testdata_loader = None, None

    # --- 这里新增对 stad 的解析 ---
    if train_data is not None:
        if train_data.name == 'dota':
            train_data = Dota(**train_data)
        elif train_data.name == 'stad':
            train_data = Stad(**train_data)
        else:
            raise Exception(f'unsupported dataset {train_data.name}')

    if test_data is not None:
        if test_data.name == 'dota':
            test_data = Dota(**test_data)
        elif test_data.name == 'stad':
            test_data = Stad(**test_data)
        else:
            raise Exception(f'unsupported dataset {test_data.name}')

    # training dataset
    if cfg.phase == 'train' and train_data is not None:
        if cfg.distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_data, shuffle=train_shuffle)
            train_shuffle = False 

        traindata_loader = DataLoader(
            dataset=train_data, batch_size=cfg.batch_size, sampler=train_sampler,
            shuffle=train_shuffle, drop_last=True, num_workers=cfg.num_workers, collate_fn=pad_collate, #cfg.num_workers
            pin_memory=False)
        
        print("distributed is {} and train set: {}".format(cfg.distributed, len(train_data)))

    # testing dataset 
    if test_data is not None:
        if cfg.distributed:
            test_sampler = torch.utils.data.distributed.DistributedSampler(test_data, shuffle=test_shuffle)
            test_shuffle = False

        testdata_loader = DataLoader(
            dataset=test_data, batch_size=cfg.test_batch_size, shuffle=test_shuffle, sampler=test_sampler, collate_fn=pad_collate, 
            drop_last=False, num_workers=cfg.num_workers,
            pin_memory=False, prefetch_factor=1)
        
        print("distributed is {} and test set: {}".format(cfg.distributed, len(test_data)))
    
    return train_sampler, test_sampler, traindata_loader, testdata_loader