import torch
import math
from collections import OrderedDict
from torch import nn
from torch.nn import functional as F
from torch.nn.init import trunc_normal_
from torch.nn.utils.rnn import pad_sequence
import functools
import operator
from models.TTHF import open_clip_local as clip
from models.prompt_models.multiscale_vst import register_vst_model
from models.prompt_models.vst_fpn import Standard_FPN
from models.prompt_models.aggregation import Instance_Encoder , Instance_Decoder_V2
from models.Transformer import *
from models.componets import *
from models.dinov2.dinov2.models import register_vit_model
from models.prompt_models.video_vits import register_video_vit_model
from models.positional_encoding import PositionEmbeddingSine, PositionIndexAwareEmbedding

# Initialize Policy weights
def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)
    if isinstance(m, nn.LSTMCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                torch.nn.init.orthogonal_(param.data)
            else:
                torch.nn.init.normal_(param.data)
    if isinstance(m, nn.Conv3d):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)

class poma(nn.Module):
    def __init__(
        self,
        # vst
        vst_cfg: dict = {},
        fpn_cfg: dict = {},
        # vit
        vit_cfg: dict = {},
        ins_encoder_cfg: dict ={},  
        ins_decoder_cfg = {}, 
        ano_decoder_cfg: dict = {},
        proxy_task_cfg: dict = {},
        ):
        super().__init__()
        empy_dict = {}
        # backbone type
        configs = { 'vst': vst_cfg, 'vit': vit_cfg, }
        self.vit_type = next((key for key, cfg in configs.items() if cfg != empy_dict), None)

        if self.vit_type == 'vst':
            self.vst_model = register_vst_model(**vst_cfg)
            self.use_fpn = fpn_cfg.dimen_reduce_type != empy_dict
            if self.use_fpn:
                self.fpn = Standard_FPN(**fpn_cfg)     
            else:
                self.use_linear = fpn_cfg.get('apply_linar',False) # 兼容之前的cfg
                if self.use_linear:
                    self.linear = nn.Sequential(nn.Linear(fpn_cfg.linear.vst_dim , fpn_cfg.linear.target_dim),nn.ReLU())

        if self.vit_type == 'vit':
            self.vit_model = register_video_vit_model(**vit_cfg)
            self.use_linear = fpn_cfg.get('apply_linar',False) # 兼容之前的cfg
            if self.use_linear:
                self.linear = nn.Sequential(nn.Linear(fpn_cfg.linear.vit_dim , fpn_cfg.linear.target_dim),nn.ReLU())
                self.norm = nn.LayerNorm(fpn_cfg.linear.target_dim)

        # build Instance Encoder-Decoder
        self.use_ins_encoder = ins_encoder_cfg != empy_dict   
        if self.use_ins_encoder :
            self.instance_encoder = Instance_Encoder(ins_encoder_cfg.pmt_encoder, ins_encoder_cfg.resize,ins_encoder_cfg.pmt_decoder )

            # way of augmenting instance-aware embedding: bottleneck or average
            self.agg_type = ins_encoder_cfg.aggregation_type # 'bottle_neck' or 'mean'
            if self.agg_type == 'bottle_neck': 
                bottle_aug_cfg = ins_encoder_cfg.bottle_aug
                self.bottle_aug = SelfAttentionBlock(TransformerLayer(bottle_aug_cfg.hid_dim, MultiHeadAttention(bottle_aug_cfg.nhead, bottle_aug_cfg.hid_dim),
                                                                    PositionwiseFeedForward(bottle_aug_cfg.hid_dim, bottle_aug_cfg.ffn_dim), bottle_aug_cfg.dropout))
                self.bottle_regressor = nn.Sequential(nn.Linear(bottle_aug_cfg.hid_dim, 64), nn.ReLU(),
                                                    nn.Dropout(0.3),nn.Linear(64, 32), nn.Dropout(0.3),
                                                    nn.Linear(32, 1))                                     

        self.use_ins_decoder = ins_decoder_cfg != empy_dict
        if self.use_ins_decoder :    
            # instance_decoder ablation experiment 
            if ins_decoder_cfg == 'mean':
                self.instance_decoder = 'mean' 
                self.ins_reducer = Reducer(**ano_decoder_cfg.reducer) # for object-aug image embs              
            else:
                self.instance_query = nn.Parameter(torch.zeros(1, ins_decoder_cfg.num_query_token, ins_decoder_cfg.hid_dim))
                self.instance_query.data.normal_(mean=0.0, std=ins_decoder_cfg.initializer_range)
                self.instance_decoder = Instance_Decoder_V2(ins_decoder_cfg.block_depth, ins_decoder_cfg.block)

        # build Anomaly Decoder:  'plain' , 'rnn' or 'memory bank'
        self.anomaly_decoder_type =  ano_decoder_cfg.type
        if ano_decoder_cfg.type == 'plain':
            self.anomaly_regressor = Plain_regressor(**ano_decoder_cfg.regressor)

        elif ano_decoder_cfg.type == 'rnn':
            self.reducer = Reducer(**ano_decoder_cfg.reducer)
            self.anomaly_regressor = Rnn_regressor(**ano_decoder_cfg.regressor)
            
        elif ano_decoder_cfg.type == 'memory':
            self.init_memory_states()
            self.stride = ano_decoder_cfg.stride 
            # query type
            self.ano_query_type = ano_decoder_cfg.query_type # 'learnable_tokens' or 'vit_tokens'
            if self.ano_query_type == 'learnable_tokens':
                self.query_tokens = nn.Parameter(torch.zeros(1, ano_decoder_cfg.num_query_token, ano_decoder_cfg.hid_dim))   
                self.query_tokens.data.normal_(mean=0.0, std=ano_decoder_cfg.initializer_range)
            elif self.ano_query_type == 'vit_tokens':
                pass
            
            # frame emb apply pool
            self.apply_pool = ano_decoder_cfg.apply_pool
            if self.apply_pool:
                self.vis_pool = Vis_pool(**ano_decoder_cfg.vis_pool)  
            
            self.memory_bank_length = ano_decoder_cfg.memory_bank_length
            self.compress_size = ano_decoder_cfg.get('compress_size', -1)
            self.apply_vis_mb = ano_decoder_cfg.apply_vis_mb
            self.apply_obj_mb = ano_decoder_cfg.apply_obj_mb

            self.memory_tpos_enc = torch.nn.Parameter(torch.zeros(self.memory_bank_length, 1, 1, ano_decoder_cfg.hid_dim))   
            trunc_normal_(self.memory_tpos_enc, std=0.02)
            self.pe_2d = PositionEmbeddingSine(ano_decoder_cfg.hid_dim)
            # self.pe_1d = PositionIndexAwareEmbedding(ano_decoder_cfg.hid_dim)
            self.anomaly_regressor = Memoery_regressor(**ano_decoder_cfg.regressor)

        # proxy task
        # self.use_proxy_task = proxy_task_cfg != empy_dict
        # if self.use_proxy_task:
        #     self.proxy_tasks = proxy_task_cfg.task_names
        #     # instance anomaly detection
        #     if proxy_task_cfg.use_ins_anomaly:
        #         assert self.use_ins_decoder, 'instance anomaly detection requires instance decoder'
        #         if proxy_task_cfg.ins_anomaly_type == 'ffn':
        #             self.ins_anomaly_type = 'ffn'
        #             self.ins_anomaly_regressor = MLP(**proxy_task_cfg.ins_anomaly_ffn)
                         
        # self.apply(weights_init_)
        # init weights before attach the vision transformer 
        # self.vst_model = register_vst_model(**vst_cfg)
    
    def get_num_layers(self,):
        if self.vit_type == 'vst':
            raise NotImplementedError(f'Unsupported get_num_layers for vst backbone')
            # return self.vst_model.get_num_layers()
            
        elif self.vit_type == 'vit':
            return self.vit_model.get_num_layers()
            
    def pad_and_mask(self, seq_list):
        from torch.nn.utils.rnn import pad_sequence
        padded = pad_sequence(seq_list, batch_first=True)
        mask = torch.zeros(padded.shape[:2], dtype=torch.bool, device=padded.device)
        for i, s in enumerate(seq_list):
            mask[i, :s.shape[0]] = 1
        return padded, mask

    @staticmethod
    def pad_to_max_len(tensors, max_len):
        B = len(tensors)
        shapes = tensors[0].shape
        if len(shapes) == 2: # [N,D]
            padded = pad_sequence(tensors, batch_first=True)
            return padded
        elif len(shapes) == 3: # [N,H,D]
            H, D = tensors[0].shape[1], tensors[0].shape[2]
            padded = tensors[0].new_zeros((B, H, max_len, D))
            for i, t in enumerate(tensors):
                padded[i, :, :t.shape[0], :] = t.transpose(0, 1)
            return padded

    def init_memory_states(self,):
        self.memory_states = OrderedDict()


    def reset_memory_states(self,):
        self.memory_states.clear()

    @staticmethod
    def concat_varlen_time_pad(a_list, b_list, pad_value=0.0):
        """
        将多个时间步的变长张量拼接并pad成统一长度。

        Args:
            a_list: list of T tensors, each [B, N_max_t, C]
            b_list: list of T tensors/lists, each [B]
            pad_value: float, padding值 (默认0)

        Returns:
            padded: [B, max_total_len, C]
            lengths: [B] 每个样本拼接后的有效长度
        """
        T = len(a_list)
        B = a_list[0].size(0)
        out_list = []
        lengths = []

        for i in range(B):
            seqs_i = []
            total_len = 0
            for t in range(T):
                bi = int(b_list[t][i])
                seqs_i.append(a_list[t][i, :bi])
                total_len += bi
            out_i = torch.cat(seqs_i, dim=0)  # [sum_t b[t][i], C]
            out_list.append(out_i)
            lengths.append(total_len)

        padded = pad_sequence(out_list, batch_first=True, padding_value=pad_value)
        # lengths = torch.tensor(lengths, device=padded.device)
        return padded, lengths


    def select_memory_feature(self, frame_idx, device, track_in_reverse=False):
        stride = self.stride 
        t_pos_and_prevs = []
        img_embs_memory , agg_ins_embs_memory , spatem_2d_pe, ins_tokens_memory, tem_1d_pe, ins_tokens_memory_len  = [], [], [], [], [], []
        for t_pos in range(1, self.memory_bank_length):
            t_rel = self.memory_bank_length - t_pos  # how many frames before current frame
            if t_rel == 1:
                # for t_rel == 1, we take the last frame (regardless of r)
                if not track_in_reverse:
                    # the frame immediately before this frame (i.e. frame_idx - 1)
                    prev_frame_idx = frame_idx - t_rel
                else:
                    # the frame immediately after this frame (i.e. frame_idx + 1)
                    prev_frame_idx = frame_idx + t_rel
            else:
                # for t_rel >= 2, we take the memory frame from every r-th frames
                if not track_in_reverse:
                    # first find the nearest frame among every r-th frames before this frame
                    # for r=1, this would be (frame_idx - 2)
                    prev_frame_idx = ((frame_idx - 2) // stride) * stride
                    # then seek further among every r-th frames
                    prev_frame_idx = prev_frame_idx - (t_rel - 2) * stride
                else:
                    # first find the nearest frame among every r-th frames after this frame
                    # for r=1, this would be (frame_idx + 2)
                    prev_frame_idx = -(-(frame_idx + 2) // stride) * stride
                    # then seek further among every r-th frames
                    prev_frame_idx = prev_frame_idx + (t_rel - 2) * stride
            out = self.memory_states.get(prev_frame_idx, None)
            t_pos_and_prevs.append((t_pos, out))
        # add current frame as memory
        t_pos_and_prevs.append((0, self.memory_states.get(frame_idx, None)))

        eff_mem_len = 0
        for t_pos, prev in t_pos_and_prevs:
            if prev is None:
                continue  # skip padding frames
            # temp_img_embs temp_agg_ins_embs temp_ins_tokens
            eff_mem_len +=1
            feats = prev["temp_img_embs"].to(device, non_blocking=True)
            img_embs_memory.append(feats.flatten(2).permute(0, 2, 1)) # [B, C, H, W] -> [B, C, HW] -> [B, HW, C]
            # Spatial positional encoding  
            spa_2d_pe = self.pe_2d(feats).to(device)
            spa_2d_pe = spa_2d_pe.flatten(2).permute(0, 2, 1)
            # Temporal positional encoding
            spa_2d_pe = (
                spa_2d_pe + self.memory_tpos_enc[self.memory_bank_length - t_pos - 1]
            )
            spatem_2d_pe.append(spa_2d_pe)
            # temp_ins_tokens with Temporal positional encoding
            if self.use_ins_encoder:
                tmp_feats = prev["temp_agg_ins_embs"].to(device, non_blocking=True)
                agg_ins_embs_memory.append(tmp_feats.flatten(2).permute(0, 2, 1)) # [B, C, H, W] -> [B, C, HW] -> [B, HW, C]
                ins_feats = prev["temp_ins_tokens"].to(device, non_blocking=True)
                ins_feats = ins_feats  
                B, N_max,C = ins_feats.shape
                tem_1d_pe.append(self.memory_tpos_enc[self.memory_bank_length - t_pos - 1].expand(B,N_max,C))
                ins_tokens_memory.append(ins_feats)
                ins_tokens_memory_len.append(prev['temp_ins_tokens_len'])

        padded_ins_tokens_memory, ins_tokens_memory_lengths = None , None
        if self.use_ins_encoder:
            padded_ins_tokens_memory, ins_tokens_memory_lengths = self.concat_varlen_time_pad(ins_tokens_memory, ins_tokens_memory_len)
            tem_1d_pe , _ = self.concat_varlen_time_pad(tem_1d_pe, ins_tokens_memory_len)

        # compress membank
        ins_embs_pe = None
        if self.compress_size !=-1 and eff_mem_len>self.compress_size:
            img_embs_memory = torch.stack(img_embs_memory, dim=1)  # [B, T, N, C]
            img_embs_pe = torch.stack(spatem_2d_pe, dim=1)  # [B, T, N, C]
            B,T,N,C = img_embs_memory.shape
            compress_step = eff_mem_len-self.compress_size
            img_size = torch.ones(B, T, N).to(device) # [B, T, N]  
            for _ in range(compress_step): 
                img_embs_memory, img_size, img_embs_pe = memory_bank_compress(img_embs_memory, img_size, img_embs_pe)
            img_embs_memory = img_embs_memory.flatten(1, 2)
            img_embs_pe = img_embs_pe.flatten(1, 2)
            
            if self.use_ins_encoder:
                agg_ins_embs_memory = torch.stack(agg_ins_embs_memory, dim=1)  # [B, T, N, C]
                ins_embs_pe = torch.stack(spatem_2d_pe, dim=1)  # [B, T, N, C]
                B,T,N,C = agg_ins_embs_memory.shape
                compress_step = eff_mem_len-self.compress_size
                ins_size = torch.ones(B, T, N).to(device) # [B, T, N]   
                for _ in range(compress_step): 
                    agg_ins_embs_memory, ins_size , ins_embs_pe = memory_bank_compress(agg_ins_embs_memory, ins_size, ins_embs_pe)
                agg_ins_embs_memory = agg_ins_embs_memory.flatten(1, 2)
                ins_embs_pe = ins_embs_pe.flatten(1, 2)
        else:
            img_embs_memory = torch.concat(img_embs_memory, dim=1)  # [B, TxN, C]
            img_embs_pe = torch.concat(spatem_2d_pe, dim=1)  # [B, TxN, C]
            if self.use_ins_encoder:
                agg_ins_embs_memory = torch.concat(agg_ins_embs_memory, dim=1)  # [B, TxN, C]
                ins_embs_pe = torch.concat(spatem_2d_pe, dim=1)  # [B, TxN, C]

        return img_embs_memory , img_embs_pe, agg_ins_embs_memory , ins_embs_pe, padded_ins_tokens_memory, ins_tokens_memory_lengths, tem_1d_pe

    def forward(self, imgs , boxes , rnn_state = None, frame_state = None):
        if self.vit_type == 'vst':
            # imgs: B, T, C, W, H -> B, C, T, W, H
            vst_img_emb = self.vst_model(imgs.permute(0,2,1,3,4)) # list[]
            if self.use_fpn:
                img_embs = self.fpn(vst_img_emb) # [B, C, H, W]
            else:
                img_embs = vst_img_emb[-1] # [B, C, T, H, W]
                if self.use_linear:
                    B, C, T, H, W = img_embs.shape
                    img_embs = img_embs.flatten(2).transpose(1, 2).contiguous()
                    img_embs = self.linear(img_embs)
                    img_embs = img_embs.permute(0,2,1).view(B,-1,T,H,W).contiguous()
                    img_embs = torch.mean(img_embs,dim=2)
        
        if self.vit_type == 'vit':
            # imgs: [B, T, C, H, W] -> [B, C, T, H, W]
            img_embs = self.vit_model(imgs.permute(0,2,1,3,4)) # [B, C, H, W]
            if self.use_linear:    
                B, _, H, W = img_embs.shape
                img_embs = img_embs.flatten(2).transpose(1, 2).contiguous()
                img_embs = self.norm(self.linear(img_embs))
                img_embs = img_embs.permute(0,2,1).view(B,-1,H,W).contiguous()
        
        img_cls_token = None # prompt decoder is mean
        object_embs = None    
        B,C,H,W = img_embs.shape  
        if self.use_ins_encoder :
            # image to instance : similar to SAM-Mask Encoder 
            instance_tokens , instacne_embs = self.instance_encoder( img_embs , boxes)      
            # aggregate instance-aware frame embedding
            if self.agg_type == 'bottle_neck' :
                agg_ins_embs = []
                bottleneck_weight = []
                for ins_tokens, ins_embs in zip(instance_tokens,instacne_embs):    
                    bottle_weight = self.bottle_aug(ins_tokens.unsqueeze(dim=0))
                    bottle_weight = self.bottle_regressor(bottle_weight)
                    bottleneck_weight.append(bottle_weight.squeeze(dim=0))
                    bottle_weight = torch.sigmoid(bottle_weight)
                    ins_embs = bottle_weight.view(-1,1,1,1)*ins_embs
                    ins_embs = torch.sum(ins_embs, dim=0, keepdim=False)
                    agg_ins_embs.append(ins_embs)
                agg_ins_embs = torch.stack(agg_ins_embs,dim=0)
            else: # mean
                agg_ins_embs = torch.stack([ torch.mean(x,dim=0) for x in instacne_embs],dim=0)        

            if self.use_ins_decoder:
                if self.instance_decoder == 'mean':
                    instance_tokens_mean = torch.concat([torch.mean(x,dim=0,keepdim=True) for x in instance_tokens],dim=0)
                    img_cls_token = self.ins_reducer(agg_ins_embs, object_embs=None, cls_tokens=instance_tokens_mean)
                    batch_object_tokens = instance_tokens 
                    pass  
                else:
                    instance_query = self.instance_query.expand(B,-1,-1)
                    # instance to ins_query: loop in batch  
                    batch_object_embs = []
                    batch_object_tokens = []                   
                    for ins_tokens, ins_embs, ins_query in zip(instance_tokens,instacne_embs,instance_query):
                        # ins_tokens tensor[1,N_obj,C]   instance_query tensor[1,N_query,C]
                        ins_tokens , ins_query = ins_tokens.unsqueeze(dim=0) , ins_query.unsqueeze(dim=0)
                        # augment intance-aware embedding
                        ins_embs = torch.mean(ins_embs,dim=0,keepdim=True)
                        # BxCxHxW -> BxHWxC == B x N_image_tokens x C
                        ins_embs = ins_embs.flatten(2).permute(0, 2, 1)
                        # add image position encoding: [1, C, H, W] -> [1, C, HW] -> [1,HW,C]
                        embs_pe = self.instance_encoder.prompt_encoder.get_dense_pe().flatten(2).permute(0,2,1)
                        object_embs, object_tokens =  self.instance_decoder(ins_query, ins_embs, ins_tokens, img_pe = embs_pe)
                        batch_object_tokens.append(object_tokens.squeeze(dim=0))
                        batch_object_embs.append(object_embs)
                    # object_embs : [B, 1, num_query, C] -> [B, num_query, C]
                    object_embs = torch.stack(batch_object_embs,dim=0).squeeze(dim=1)
        
        '''
        return :

        '''
        ret = {'output':None, 'rnn_state':None , 'ins_anomaly':None, 'bottleneck_weight':None}
        if self.use_ins_encoder and self.agg_type == 'bottle_neck':
            ret['bottleneck_weight'] = bottleneck_weight

        if self.anomaly_decoder_type == 'plain':
            output = self.anomaly_regressor(img_embs)      
            ret['output'] = output 
            
        elif self.anomaly_decoder_type == 'rnn':
            x = self.reducer(img_embs, object_embs, cls_tokens = img_cls_token) 
            # apply instance-level anomaly detection
            if self.anomaly_regressor.apply_ins_cls: 
                batch_ins_tokens = batch_object_tokens
            else:
                batch_ins_tokens = None
            output, rnn_state, ins_output = self.anomaly_regressor(x, rnn_state , batch_ins_tokens)
            ret['output'], ret['rnn_state'], ret['ins_anomaly'] = output , rnn_state, ins_output
 
        elif self.anomaly_decoder_type == 'memory':
            # if self.vit_type == 'vst':
            #     # only use the last vst emb : [B , C , 15 , 20]       
            #     # img_embs = vst_img_emb[-1].flatten(2).permute(0, 2, 1) # [B, N=HxW, C]
            #     # ave-pooling
            #     img_embs = self.vis_pool(img_embs) # [B , C , 60 , 80] -> [B , C , 6 , 6] -> [B , 36, C]    
            # elif self.vit_type == 'vit':
            #     img_embs = self.vis_pool(img_embs)

            # init memory_states
            if frame_state.t == frame_state.begin_t:
                self.reset_memory_states()

            if self.apply_pool:
                img_embs = self.vis_pool(img_embs)
            
            # padding instance tokens to max length in the batch
            ins_len, q_max_len = [], 0  
            if self.use_ins_encoder:
                ins_len = [q.shape[0] for q in instance_tokens] 
                q_max_len = max(ins_len)   
                padded_ins_tokens = self.pad_to_max_len(instance_tokens, q_max_len) 


            is_calflops = True
            if is_calflops:
                '''
                计算复杂度：直接构造memory bank feature
                '''
                cal_membank = 8
                for tmp_i in range(cal_membank): 
                    self.memory_states[tmp_i]= {
                        'temp_img_embs': img_embs.detach(),
                        'temp_agg_ins_embs': agg_ins_embs.detach() if self.use_ins_encoder else None, 
                        'temp_ins_tokens': padded_ins_tokens.detach() if self.use_ins_encoder else None,
                        'temp_ins_tokens_len': ins_len if self.use_ins_encoder else None,
                    }
                frame_state.t = cal_membank - 1
                
            else:
                self.memory_states[frame_state.t]= {
                    'temp_img_embs': img_embs.detach(),
                    'temp_agg_ins_embs': agg_ins_embs.detach() if self.use_ins_encoder else None, 
                    'temp_ins_tokens': padded_ins_tokens.detach() if self.use_ins_encoder else None,
                    'temp_ins_tokens_len': ins_len if self.use_ins_encoder else None,
                }

   
            (   img_embs_memory , img_embs_pe,
                agg_ins_embs_memory , ins_embs_pe,
                padded_ins_tokens_memory,  ins_tokens_memory_lengths, tem_1d_pe         
            ) = self.select_memory_feature(frame_state.t, img_embs.device, track_in_reverse=False)

            # prepare query tokens : [query_tokens, instance_tokens]
            query_tokens, query_pe = None, None
            if self.ano_query_type == 'learnable_tokens':
                query_tokens = self.query_tokens.expand(B, -1, -1) # [B, N_query, C]
                query_pe = query_tokens
            elif self.ano_query_type == 'vit_tokens':
                query_tokens = img_embs.flatten(2).permute(0,2,1) # [B, N_img, C]
                query_pe = self.pe_2d(img_embs).to(img_embs.device).flatten(2).permute(0,2,1)

            B, Q_len, C = query_tokens.shape
            # query_tokens = [query_tokens, instance_tokens]
            q_lengths = None
            if self.use_ins_encoder :
                query_tokens = torch.concat([query_tokens, padded_ins_tokens],dim=1)  # 
                query_pe = torch.concat([query_pe, self.memory_tpos_enc[0].expand(B, q_max_len, C)],dim=1)
                q_lengths = [Q_len + l for l in ins_len]
  
            # img_hidden_states , obj_hidden_states = None , None
            # if frame_state.t == frame_state.begin_t:
            #     if self.apply_vis_mb:
            #         img_hidden_states  = img_embs # [B, 1, N, C]
            #         self.img_size_constant = torch.ones(B, 1, N_img).to(img_embs.device) # [B, 1, N]
            #     if self.apply_obj_mb:
            #         obj_hidden_states  = object_embs # [B, 1, obj_query, C]
            #         self.obj_size_constant = torch.ones(B, 1, N_obj).to(object_embs.device) # [B, 1, obj_query]
            # else:
            #     if self.apply_vis_mb:
            #         img_hidden_states = torch.cat([self.visual_memory_bank, img_embs], dim=1) # [B, (t+1), N, C]
            #     if self.apply_obj_mb:
            #         obj_hidden_states = torch.cat([self.object_memory_bank, object_embs], dim=1) # [B, (t+1), obj_query, C]
            
            # prepare frame memory features
            img_hidden_states , obj_hidden_states = None , None
            img_hidden_states_pe, obj_hidden_states_pe = None , None
            if self.apply_vis_mb:
                img_hidden_states = img_embs_memory # [B, N, C]
                img_hidden_states_pe = img_embs_pe  # [B, N, C]

            k_max_len = 0
            if self.apply_obj_mb:
                obj_hidden_states = agg_ins_embs_memory  # [B, N, C]
                obj_hidden_states_pe = ins_embs_pe  # [B, N, C]
                # add obj
                _, K_len , _ = obj_hidden_states.shape
                obj_hidden_states = torch.concat([obj_hidden_states, padded_ins_tokens_memory],dim=1)
                obj_hidden_states_pe = torch.concat([obj_hidden_states_pe, tem_1d_pe ],dim=1)
                k_max_len = max(ins_tokens_memory_lengths)
                k_lengths = [K_len + l for l in ins_tokens_memory_lengths]
            
            kwds = {
                    "emb_H_W":(H, W),
                    "ins_len": ins_len,
                    "q_exclude_rope": q_max_len,
                    "img_q_lengths": q_lengths,
                    "img_kv_lengths":None,
                    "obj_q_lengths": q_lengths,
                    "obj_kv_lengths":k_lengths if self.apply_obj_mb else None,
                    "k_exclude_rope": k_max_len,
                }

            frame_output, ins_output = self.anomaly_regressor(query_tokens , query_pe, img_hidden_states, img_hidden_states_pe,
                                            obj_hidden_states, obj_hidden_states_pe, frame_state, **kwds)
            
            ret['output'] = frame_output 
            ret['ins_anomaly'] = ins_output
            # '''
            # ffn for instance anomaly detection 
            # '''
            # if self.use_proxy_task and 'instance' in self.proxy_tasks and self.ins_anomaly_type == 'ffn': 
            #     ret['ins_anomaly'] = []
            #     for frame_instance_tokens in batch_object_tokens: # batch_object_tokens = instance_tokens when self.instance_decoder == 'mean' 
            #         ins_anomaly_output = self.ins_anomaly_regressor(frame_instance_tokens)
            #         ret['ins_anomaly'].append(ins_anomaly_output)
      
        return ret
