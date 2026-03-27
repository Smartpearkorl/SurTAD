from torch import nn ,Tensor
import torch
import numpy as np

from models.prompt_models.points_transform import ResizeCoordinates
from models.prompt_models.prompt_modules import *
from models.Transformer import *

'''
                 Instance-wise Aggregation
target:
    transform object bounding box to instance-aware feature embedding with image feature embedding
                 
module included:
       1. Object-promopt Encoder: encode bounding box coordinate into feature embedding
       2. Instance-wise Aggregation: aggregate instance-aware feature embedding 
'''
class Instance_Encoder(nn.Module):
    def __init__(self,    
        prompt_encoder_cfg:dict,
        resize_cfg:dict,
        prompt_decoder_cfg:dict,
        ):
        super().__init__()
        self.prompt_encoder = PromptEncoder(**prompt_encoder_cfg)
        self.transform = ResizeCoordinates(**resize_cfg)
        self.prompt_decoder = PromptDecoder(**prompt_decoder_cfg)

    def forward(self, img_emb: Tensor , boxes_batch: np.array ): # points: np.array
        # need to loop in batch for building instnce-aware embedding 
        batch_ins_tokens , batch_ins_embs = [], []
        for frame_emb , frame_boxes  in zip(img_emb,boxes_batch):  
            frame_boxes = self.transform.apply_boxes(frame_boxes)
            box_torch = torch.as_tensor(frame_boxes, dtype=torch.float, device=img_emb.device)
           
            # Embed prompts
            sparse_embeddings, dense_embeddings = self.prompt_encoder(points = None, boxes = box_torch, masks = None)
            instance_tokens, instacne_embs = self.prompt_decoder(
                image_embeddings=frame_emb.unsqueeze(dim=0),
                image_pe=self.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,)

            batch_ins_tokens.append(instance_tokens)
            batch_ins_embs.append(instacne_embs)
        return  batch_ins_tokens , batch_ins_embs

'''
            Memory Decoder Aggregation
target: 
            
''' 
class MemoryAttentionLayer(nn.Module):

    def __init__(
        self,
        activation: str,
        cross_attention: nn.Module,
        d_model: int,
        dim_feedforward: int,
        dropout: float,
        pos_enc_at_attn: bool,
        pos_enc_at_cross_attn_keys: bool,
        pos_enc_at_cross_attn_queries: bool,
        self_attention: nn.Module,
    ):
        super().__init__()
        self.d_model = d_model
        self.dim_feedforward = dim_feedforward
        self.dropout_value = dropout
        self.self_attn = self_attention
        self.cross_attn_image = cross_attention

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation_str = activation
        self.activation = get_activation_fn(activation)

        # Where to add pos enc
        self.pos_enc_at_attn = pos_enc_at_attn
        self.pos_enc_at_cross_attn_queries = pos_enc_at_cross_attn_queries
        self.pos_enc_at_cross_attn_keys = pos_enc_at_cross_attn_keys

    def _forward_sa(self, tgt, query_pos):
        # Self-Attention
        tgt2 = self.norm1(tgt)
        q = k = tgt2 + query_pos if self.pos_enc_at_attn else tgt2
        tgt2 = self.self_attn(q, k, v=tgt2)
        tgt = tgt + self.dropout1(tgt2)
        return tgt

    def _forward_ca(self, tgt, memory, query_pos, pos, num_k_exclude_rope=0):
        kwds = {}
        if num_k_exclude_rope > 0:
            assert isinstance(self.cross_attn_image, RoPEAttention)
            kwds = {"num_k_exclude_rope": num_k_exclude_rope}

        # Cross-Attention
        tgt2 = self.norm2(tgt)
        tgt2 = self.cross_attn_image(
            q=tgt2 + query_pos if self.pos_enc_at_cross_attn_queries else tgt2,
            k=memory + pos if self.pos_enc_at_cross_attn_keys else memory,
            v=memory,
            **kwds,
        )
        tgt = tgt + self.dropout2(tgt2)
        return tgt

    def forward(
        self,
        tgt,
        memory,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
        num_k_exclude_rope: int = 0,
    ) -> torch.Tensor:

        # Self-Attn, Cross-Attn
        tgt = self._forward_sa(tgt, query_pos)
        tgt = self._forward_ca(tgt, memory, query_pos, pos, num_k_exclude_rope)
        # MLP
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt


class MemoryDecoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        pos_enc_at_input: bool,
        # layer: nn.Module,
        num_layers: int,
        batch_first: bool = True,  # Do layers expect batch first input?
    ):
        super().__init__()
        self.d_model = d_model
        self.layers = clones(MemoryAttentionLayer, num_layers)
        self.num_layers = num_layers
        self.norm = nn.LayerNorm(d_model)
        self.pos_enc_at_input = pos_enc_at_input
        self.batch_first = batch_first

    def forward(
        self,
        curr: torch.Tensor,  # self-attention inputs
        memory: torch.Tensor,  # cross-attention inputs
        curr_pos: Optional[Tensor] = None,  # pos_enc for self-attention inputs
        memory_pos: Optional[Tensor] = None,  # pos_enc for cross-attention inputs
        num_obj_ptr_tokens: int = 0,  # number of object pointer *tokens*
    ):
        if isinstance(curr, list):
            assert isinstance(curr_pos, list)
            assert len(curr) == len(curr_pos) == 1
            curr, curr_pos = (
                curr[0],
                curr_pos[0],
            )

        assert (
            curr.shape[1] == memory.shape[1]
        ), "Batch size must be the same for curr and memory"

        output = curr
        if self.pos_enc_at_input and curr_pos is not None:
            output = output + 0.1 * curr_pos

        if self.batch_first:
            # Convert to batch first
            output = output.transpose(0, 1)
            curr_pos = curr_pos.transpose(0, 1)
            memory = memory.transpose(0, 1)
            memory_pos = memory_pos.transpose(0, 1)

        for layer in self.layers:
            kwds = {}
            if isinstance(layer.cross_attn_image, RoPEAttention):
                kwds = {"num_k_exclude_rope": num_obj_ptr_tokens}

            output = layer(
                tgt=output,
                memory=memory,
                pos=memory_pos,
                query_pos=curr_pos,
                **kwds,
            )
        normed_output = self.norm(output)

        if self.batch_first:
            # Convert back to seq first
            normed_output = normed_output.transpose(0, 1)
            curr_pos = curr_pos.transpose(0, 1)

        return normed_output


'''
                Relation-wise Aggregation: [add positional encoding]
target:
    aggregate all instance-aware feature embedding into enhanced frame token
'''
class Instance_block_V2(nn.Module):
    def __init__(self,hid_dim,nhead,ffn_dim,dropout):
        super(Instance_block_V2, self).__init__()
        self.query2object = CrossAttentionBlock(TransformerLayer(hid_dim, MultiHeadAttention(nhead, hid_dim), PositionwiseFeedForward(hid_dim, ffn_dim), dropout))     
        self.query2frame = CrossAttentionBlock(TransformerLayer(hid_dim, MultiHeadAttention(nhead, hid_dim), PositionwiseFeedForward(hid_dim, ffn_dim), dropout))     
        self.object2qurey = CrossAttentionBlock(TransformerLayer(hid_dim, MultiHeadAttention(nhead, hid_dim), PositionwiseFeedForward(hid_dim, ffn_dim), dropout))   

    def forward(self, query: Tensor, img_emb: Tensor , obj_emb: Tensor , query_pe:Tensor , img_pe : Tensor, obj_pe: Tensor):         
        attn_out = self.query2object(query + query_pe, obj_emb + obj_pe, obj_emb )
        query = query + attn_out
        attn_out = self.query2frame(query + query_pe, img_emb + img_pe, img_emb )
        query = query + attn_out
        attn_out = self.object2qurey(obj_emb + obj_pe, query + query_pe, query )
        obj_emb = obj_emb + attn_out

        return query , img_emb , obj_emb 

class Instance_Decoder_V2(nn.Module):
    def __init__(self,block_depth,block_cfg):
        super(Instance_Decoder_V2, self).__init__()
        self.layer = clones(Instance_block_V2(**block_cfg),block_depth)
        self.n_layers = block_depth
    def forward(self, query: Tensor ,img_emb: Tensor , obj_emb: Tensor , img_pe : Tensor):
        for layer_i in range(self.n_layers):
            query, img_emb, obj_emb = self.layer[layer_i](query, img_emb, obj_emb , query_pe = query, img_pe = img_pe , obj_pe = obj_emb )
        return query , obj_emb


