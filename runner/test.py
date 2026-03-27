
import os
from tqdm import tqdm
import datetime
import pickle

import torch
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import copy

from runner.src.tools import CEloss
from runner.src.dataset import gt_cls_target
from models.componets import HungarianMatcher

class FrameState():
    def __init__(self) -> None:
        self.t = 0
        self.begin_t = 0
        self.T = 0
        pass

def  pama_test(cfg, model, test_sampler, testdata_loader, epoch, filename):
    '''
    when DDP test, every node makes a single .pkl file, then rank_0 node load and combine all .pkl data  
    '''
    filename_rank = filename
    # DDP sampler
    if dist.is_initialized():
        filename_rank = filename_rank.replace('.pkl', f'_rank{str(dist.get_rank())}.pkl')
    
    # matcher
    apply_ins_loss = cfg.get('apply_ins_loss',False)
    if apply_ins_loss:
        matcher = HungarianMatcher()

    apply_sub_class = cfg.get('apply_sub_class',False)
    if apply_sub_class:
        sub_cls_num = cfg.get('sub_class_num',0)
        if sub_cls_num == 0:
            raise ValueError("sub_class_num must be greater than 0 when apply_sub_class is True")

    # whole test dataset data
    targets_all , outputs_all , bbox_all= [] , [] , []
    obj_targets_all , obj_outputs_all , frame_outputs_all = [] , [] , []
    toas_all, teas_all,  idxs_all , info_all = [] , [] , [] , [] 
    frames_counter = []
    video_name_all = []
    
    # --- 新增：多分类结果收集 ---
    sub_targets_all, sub_outputs_all = [], []

    # NF for vst
    fb = cfg.get('NF',0)
    # clip model use two image in every iteration 
    clip_model =  'clip' in cfg.get('model_type')

    index_frame = 0
    celoss = CEloss(cfg)

    # tqdm
    if cfg.local_rank == 0:
        pbar = tqdm(total=len(testdata_loader),desc='Test Epoch: %d / %d' % (epoch, cfg.total_epoch))

    model.eval()
    for j, (video_data, data_info, yolo_boxes, frames_boxes, video_name) in enumerate(testdata_loader):
        # prepare data for model and loss func
        video_data = video_data.to(cfg.device, non_blocking=True) # [B,T,C,H,W]  emb is float16->float32
        data_info = data_info.to(cfg.device, non_blocking=True)
        # yolo_boxes : list B x list T x nparray(N_obj, 4)
        yolo_boxes = np.array(yolo_boxes,dtype=object)
        # matcher between yolo and gt
        if apply_ins_loss:
            match_index = [matcher(yolo,gt) for yolo , gt in zip(yolo_boxes,frames_boxes)]

        # record whole video data
        B,T = video_data.shape[:2]
        t_shape = (B,T-fb) 

        # video data
        targets = torch.full(t_shape, -100).to(video_data.device)
        outputs = torch.full(t_shape, -100, dtype=float).to(video_data.device)  

        # --- 新增：多分类 Tensor ---
        if apply_sub_class:
            sub_targets = torch.full(t_shape, -100, dtype=torch.long).to(video_data.device)
            sub_outputs = torch.full((*t_shape, sub_cls_num), -100, dtype=float).to(video_data.device)
            subcls_target = data_info[:, 7] # 获取视频级子类标签

        idx_batch,toa_batch,tea_batch = data_info[:, 1] , data_info[:, 2] , data_info[:, 3]
        info_batch = data_info[:, 7:11] # accident info: accident_id ego_involve

        batch_bbox = []
        batch_obj_targets = []
        batch_obj_outputs = []
        frame_outputs = torch.full(t_shape, -100, dtype=float).to(video_data.device)

        # loop in video frames
        rnn_state , frame_state = None , FrameState()
        for i in range(fb + clip_model , T):
            # preparation
            frame_state.t = i-fb
            frame_state.begin_t = 1 if clip_model else 0
            frame_state.T = T-1-fb
            #clip 
            if clip_model:
                target = gt_cls_target(i, toa_batch, tea_batch).long()
                batch_image_data , batch_boxes = video_data[:,i-1:i+1] , yolo_boxes[:,i]
            # poma
            elif fb: 
                target = gt_cls_target(i-1, toa_batch, tea_batch).long()          
                batch_image_data , batch_boxes = video_data[:,i-fb:i] , yolo_boxes[:,i-1]
            # pama  
            else:
                target = gt_cls_target(i, toa_batch, tea_batch).long()
                batch_image_data , batch_boxes = video_data[:,i] , yolo_boxes[:,i]

         
            with torch.cuda.amp.autocast(enabled=cfg.fp16):
                if clip_model:
                    logits_per_image, logits_per_text = model(batch_image_data, batch_boxes, frame_state)
                    output = logits_per_image
                    # 注: 如果 clip_model 也用子类，需要在此处添加逻辑，暂时假设以 pama 为主
                else:
                    ret = model(batch_image_data, batch_boxes, rnn_state, frame_state)
                    output_dict , rnn_state, outputs_ins_anormal= ret['output'] , ret['rnn_state'] , ret['ins_anomaly']
                    # --- 修改：解析字典输出 ---
                    output = output_dict['frame_out'].softmax(dim=-1)
                    if apply_sub_class:
                        sub_output = output_dict['sub_class_out'].softmax(dim=-1)

                # instance anomaly loss (此处省略未改动部分，直接保持原样)
                if apply_ins_loss:
                    if fb: 
                        now_match_index = [ x[i-1] for x in match_index ]
                    else:
                        now_match_index = [ x[i] for x in match_index ]
                    outputs_ins_anormal = [x.softmax(dim=-1) for x in outputs_ins_anormal]
                    gt_objects = [torch.zeros(single.shape[0]).type(torch.int64).to(video_data.device) for single in outputs_ins_anormal]                        
                    for ind,single in enumerate(gt_objects):
                        single[now_match_index[ind][0]]=1
                    split_size = [ x.shape[0] for x in gt_objects]
                    src_objects = list(torch.cat(outputs_ins_anormal, dim=0)[:,1].split(split_size))
                    gt_objects = [x.cpu().numpy() for x in gt_objects]  
                    src_objects = [x.cpu().numpy() for x in src_objects]  
                    batch_obj_targets.append(gt_objects)
                    batch_obj_outputs.append(src_objects)

            targets[:, i-fb] = target.clone()
            outputs[:, i-fb] = output[:, 1].clone()

            # --- 新增：记录子类 ---
            if apply_sub_class and not clip_model:
                now_subcls = torch.where(target > 0, subcls_target, torch.zeros_like(subcls_target)).long()
                sub_targets[:, i-fb] = now_subcls.clone()
                sub_outputs[:, i-fb] = sub_output.clone()
        
        if clip_model: #filter first index
            targets = targets[:, 1:]
            outputs = outputs[:, 1:]
            if apply_sub_class:
                sub_targets = sub_targets[:, 1:]
                sub_outputs = sub_outputs[:, 1:]

        # concate bbox in batch into video
        video_bbox = [[] for _ in range(video_data.shape[0])]
        for sub_bbox in batch_bbox:  
            for ind,bbox in enumerate(sub_bbox):  
                video_bbox[ind].append(bbox)

        # concate obj score
        video_obj_targets = [[] for _ in range(video_data.shape[0])]
        video_obj_outputs = [[] for _ in range(video_data.shape[0])]
        for sub_target, sub_output in zip(batch_obj_targets,batch_obj_outputs): 
            for ind,(tar,out) in enumerate(zip(sub_target,sub_output)): 
                video_obj_targets[ind].append(tar)
                video_obj_outputs[ind].append(out)
        
        # collect results for each video
        video_len_batch = data_info[:, 0].int()-(fb+clip_model)  
        for i in range(targets.shape[0]):
            targets_all.append(targets[i][:video_len_batch[i]].view(-1).tolist())
            outputs_all.append(outputs[i][:video_len_batch[i]].view(-1).tolist())  
            frames_counter.append(video_len_batch[i].tolist())
            video_name_all.append(video_name[i])
            bbox_all.append(video_bbox[i][:video_len_batch[i]])
            obj_targets_all.append(video_obj_targets[i][:video_len_batch[i]])
            obj_outputs_all.append(video_obj_outputs[i][:video_len_batch[i]])
            frame_outputs_all.append(frame_outputs[i][:video_len_batch[i]].view(-1).tolist())
            
            # --- 新增：提取有效的子类序列 ---
            if apply_sub_class:
                sub_targets_all.append(sub_targets[i][:video_len_batch[i]].view(-1).tolist())
                # sub_outputs_all 存的是 [seq_len, 13] 的二维列表
                sub_outputs_all.append(sub_outputs[i][:video_len_batch[i]].tolist())

        toas_all.append(toa_batch.tolist())
        teas_all.append(tea_batch.tolist())
        idxs_all.append(idx_batch.tolist())
        info_all.append(info_batch.tolist())

        if cfg.local_rank == 0:  
            pbar.update(1)
    
    toas_all = np.array([item for sublist in toas_all for item in sublist]).reshape(-1)
    teas_all = np.array([item for sublist in teas_all for item in sublist]).reshape(-1)
    idxs_all = np.array([item for sublist in idxs_all for item in sublist]).reshape(-1)
    info_all = np.array([item for sublist in info_all for item in sublist]).reshape(-1, 4)
    frames_counter = np.array(frames_counter).reshape(-1)
    video_name_all = np.array(video_name_all).reshape(-1)

    print(f'save file {filename_rank}')
    
    # 构造保存的字典
    save_dict = {
        'targets': targets_all, 'outputs': outputs_all, 'bbox_all':bbox_all,
        'obj_targets':obj_targets_all, 'obj_outputs':obj_outputs_all,
        'fra_outputs':frame_outputs_all, 'toas': toas_all, 'teas': teas_all,
        'idxs': idxs_all, 'info': info_all, 'frames_counter': frames_counter,
        'video_name':video_name_all,
    }
    if apply_sub_class:
        save_dict['sub_targets'] = sub_targets_all
        save_dict['sub_outputs'] = sub_outputs_all

    with open(filename_rank, 'wb') as f:
        pickle.dump(save_dict, f)

    if dist.is_initialized():
        dist.barrier()
    
    # combine all .pkl data 
    if cfg.local_rank == 0 and dist.is_initialized():
        device = torch.cuda.device_count()
        contents = []
        for i in range(device):
            with open(filename.replace('.pkl', f'_rank{str(i)}.pkl'), 'rb') as f:
                contents.append(pickle.load(f))
            os.remove(filename.replace('.pkl', f'_rank{str(i)}.pkl'))

        targets_all, outputs_all, bbox_all = [], [], []
        obj_targets_all, obj_outputs_all, frame_outputs_all = [], [], []
        toas_all, teas_all, idxs_all, info_all = [], [], [], []
        frames_counter, video_name_all = [], []
        sub_targets_all, sub_outputs_all = [], []

        for i in range(device):
            targets_all.extend(contents[i]['targets'])
            outputs_all.extend(contents[i]['outputs'])
            bbox_all.extend(contents[i]['bbox_all'])
            obj_targets_all.extend(contents[i]['obj_targets'])
            obj_outputs_all.extend(contents[i]['obj_outputs'])
            frame_outputs_all.extend(contents[i]['fra_outputs'])
            toas_all.append(contents[i]['toas'])
            teas_all.append(contents[i]['teas'])
            idxs_all.append(contents[i]['idxs'])
            info_all.append(contents[i]['info'])
            frames_counter.append(contents[i]['frames_counter'])
            video_name_all.append(contents[i]['video_name'])
            
            if apply_sub_class:
                sub_targets_all.extend(contents[i]['sub_targets'])
                sub_outputs_all.extend(contents[i]['sub_outputs'])
        
        toas_all = np.concatenate(toas_all).reshape(-1)
        teas_all = np.concatenate(teas_all).reshape(-1)
        idxs_all = np.concatenate(idxs_all).reshape(-1)
        info_all = np.concatenate(info_all).reshape(-1, 4)
        frames_counter = np.concatenate(frames_counter).reshape(-1)
        video_name_all = np.array(video_name_all).reshape(-1)
        
        final_save_dict = {
            'targets': targets_all, 'outputs': outputs_all, 'bbox_all':bbox_all,
            'obj_targets':obj_targets_all, 'obj_outputs':obj_outputs_all,
            'fra_outputs':frame_outputs_all, 'toas': toas_all, 'teas': teas_all,
            'idxs': idxs_all, 'info': info_all, 'frames_counter': frames_counter,
            'video_name':video_name_all,
        }
        if apply_sub_class:
            final_save_dict['sub_targets'] = sub_targets_all
            final_save_dict['sub_outputs'] = sub_outputs_all

        print(f'save file {filename}')
        with open(filename, 'wb') as f:
            pickle.dump(final_save_dict, f)