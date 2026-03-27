#Some codes are grabbed from https://github.com/JustinYuu/MACIL_SD/blob/main/Transformer.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math

import torch
import torch.nn as nn
from torch import Tensor
from einops import rearrange

from flash_attn.flash_attn_interface import flash_attn_varlen_qkvpacked_func, flash_attn_func
from flash_attn.bert_padding import unpad_input, pad_input
from flash_attn import flash_attn_varlen_func

from models.positional_encoding import apply_rotary_enc, compute_axial_cis



def get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu, not {activation}.")

'''
VideoMAE Attention Module 
'''
class Attention(nn.Module):
    def __init__(
            self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.,
            proj_drop=0., attn_head_dim=None, use_flash_attn=False, causal=False):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, all_head_dim * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
            self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
        else:
            self.q_bias = None
            self.v_bias = None

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.use_flash_attn = use_flash_attn
        if use_flash_attn:
            self.causal = causal
            self.inner_attn = FlashAttention(attention_dropout=attn_drop)

    def _naive_attn(self, x):
        B, N, C = x.shape
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias))
        # qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    
    def _flash_attn(self, x):
        B, N, _ = x.shape
        qkv_bias = None
        if self.q_bias is not None:
            # Combine biases as in Code B
            qkv_bias = torch.cat((
                self.q_bias, 
                torch.zeros_like(self.v_bias, requires_grad=False), 
                self.v_bias
            ))
        # Compute qkv using the same linear layer as vanilla
        qkv = F.linear(x, weight=self.qkv.weight, bias=qkv_bias)
        # Reshape to [B, N, 3, num_heads, -1]
        qkv = qkv.reshape(B, N, 3, self.num_heads, -1)
        # Alternatively, you could also use rearrange:
        # qkv = rearrange(qkv, "b s (three h d) -> b s three h d", three=3, h=self.num_heads)
        
        # Call flash attention module (flash op expects the qkv in a similar shape)
        context, _ = self.inner_attn(qkv, causal=self.causal)
        # context is expected to be of shape [B, N, num_heads, d]
        x = self.proj(context.view(B, N, -1))
        x = self.proj_drop(x)
        return x
    
    def forward(self, x):
        x = self._naive_attn(x) if not self.use_flash_attn else self._flash_attn(x)
        return x

class FlashAttention(nn.Module):
    """Implement the scaled dot product attention with softmax.
    Arguments
    ---------
        softmax_scale: The temperature to use for the softmax attention.
                      (default: 1/sqrt(d_keys) where d_keys is computed at
                      runtime)
        attention_dropout: The dropout rate to apply to the attention
                           (default: 0.0)
    """

    def __init__(self, softmax_scale=None, attention_dropout=0.0, device=None, dtype=None):
        super().__init__()
        self.softmax_scale = softmax_scale
        self.dropout_p = attention_dropout

    def forward(self, qkv, key_padding_mask=None, causal=False, cu_seqlens=None,
                max_s=None, need_weights=False):
        """Implements the multihead softmax attention.
        Arguments
        ---------
            qkv: The tensor containing the query, key, and value. (B, S, 3, H, D) if key_padding_mask is None
                if unpadded: (nnz, 3, h, d)
            key_padding_mask: a bool tensor of shape (B, S)
        """
        assert not need_weights
        #assert qkv.dtype in [torch.float16, torch.bfloat16]
        assert qkv.is_cuda

        if cu_seqlens is None:
            batch_size = qkv.shape[0]
            seqlen = qkv.shape[1]
            if key_padding_mask is None:
                qkv = rearrange(qkv, 'b s ... -> (b s) ...')
                max_s = seqlen
                cu_seqlens = torch.arange(0, (batch_size + 1) * seqlen, step=seqlen, dtype=torch.int32,
                                          device=qkv.device)
                output = flash_attn_varlen_qkvpacked_func(
                    qkv, cu_seqlens, max_s, self.dropout_p if self.training else 0.0,
                    softmax_scale=self.softmax_scale, causal=causal
                )
                output = rearrange(output, '(b s) ... -> b s ...', b=batch_size)
            else:
                nheads = qkv.shape[-2]
                x = rearrange(qkv, 'b s three h d -> b s (three h d)')
                x_unpad, indices, cu_seqlens, max_s = unpad_input(x, key_padding_mask)
                x_unpad = rearrange(x_unpad, 'nnz (three h d) -> nnz three h d', three=3, h=nheads)
                output_unpad = flash_attn_varlen_qkvpacked_func(
                    x_unpad, cu_seqlens, max_s, self.dropout_p if self.training else 0.0,
                    softmax_scale=self.softmax_scale, causal=causal
                )
                output = rearrange(pad_input(rearrange(output_unpad, 'nnz h d -> nnz (h d)'),
                                             indices, batch_size, seqlen),
                                   'b s (h d) -> b s h d', h=nheads)
        else:
            assert max_s is not None
            output = flash_attn_varlen_qkvpacked_func(
                qkv, cu_seqlens, max_s, self.dropout_p if self.training else 0.0,
                softmax_scale=self.softmax_scale, causal=causal
            )

        return output, None
    
'''
PromoptTAD Attention Module
'''
class SelfAttentionBlock(nn.Module):
    def __init__(self, attention_layer):
        super(SelfAttentionBlock, self).__init__()
        self.layer = attention_layer
        self.size = attention_layer.size

    def forward(self, feature):
        feature_sa = self.layer(feature, feature, feature)
        return feature_sa


class CrossAttentionBlock(nn.Module):
    def __init__(self, attention_layer):
        super(CrossAttentionBlock, self).__init__()
        self.layer = attention_layer
        self.size = attention_layer.size

    def forward(self, q, k, v, **kwargs):
        q = self.layer(q, k, v, **kwargs)
        return q

class MultilayerTransformer(nn.Module):
    def __init__(self, Transformer_layer, n_layers):
        super(MultilayerTransformer, self).__init__()
        self.layer = clones(Transformer_layer, n_layers)
        self.n_layers = n_layers

    def forward(self, feat):
        for layer_i in range(self.n_layers):
            feat = self.layer[layer_i](feat, feat, feat)
        return feat
    
class TransformerLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(TransformerLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size
        self.norm = nn.LayerNorm(size)

    def forward(self, q, k, v, **kwargs):
        q, k, v = self.norm(q), self.norm(k), self.norm(v)
        q = self.sublayer[0](q, lambda q: self.self_attn(q, k, v, **kwargs)[0])
        return self.sublayer[1](q, self.feed_forward)


class SublayerConnection(nn.Module):
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = nn.LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return self.norm(x + self.dropout(sublayer(x)))

def attention(query, key, value,  q_lengths=None, kv_lengths=None, masksize=1, mask = None ,dropout=0.0, use_flash_attn=False):         
    """
    通用注意力函数，支持变长 FlashAttention。
    
    Args:
        query: [B, H, L_q, D]
        key:   [B, H, L_k, D]
        value: [B, H, L_v, D]
        q_lengths: list[int] or tensor[B], 每个样本的 query 有效长度
        kv_lengths: list[int] or tensor[B], 每个样本的 key/value 有效长度
        masksize: 窗口大小，仅用于非 flash 模式
        dropout: float
        use_flash_attn: 是否使用 FlashAttention 变长实现
    """
    B, H, L_q, D = query.shape
    _, _, L_k, _ = key.shape
    device = query.device

    # 分支1：变长 FlashAttention 模式
    if use_flash_attn:
        # 如果没有传长度信息，fallback 到普通 flash_attn_func
        if q_lengths is None and kv_lengths is None:
            # 走普通 flash attention
            out = flash_attn_func(
                query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2),
                dropout_p=dropout
            ).transpose(1, 2)
            return out, None
        
        # 检查长度输入
        # 构造默认长度
        if q_lengths is None:
            q_lengths = [query.shape[2]] * query.shape[0]
        if kv_lengths is None:
            kv_lengths = [key.shape[2]] * key.shape[0]
        assert len(q_lengths) == B and len(kv_lengths) == B, "lengths size mismatch with batch"

        # 把 list 转成 tensor
        q_lengths = torch.as_tensor(q_lengths, dtype=torch.int32, device=device)
        kv_lengths = torch.as_tensor(kv_lengths, dtype=torch.int32, device=device)

        # 拼接有效 token
        q_list, k_list, v_list = [], [], []
        for i in range(B):
            q_list.append(query[i, :, :q_lengths[i], :])
            k_list.append(key[i, :, :kv_lengths[i], :])
            v_list.append(value[i, :, :kv_lengths[i], :])
        q_cat = torch.cat(q_list, dim=1).transpose(0, 1).contiguous()  # [total_q, H, D]
        k_cat = torch.cat(k_list, dim=1).transpose(0, 1).contiguous()
        v_cat = torch.cat(v_list, dim=1).transpose(0, 1).contiguous()

        # 构造 cumulative sequence lengths
        cu_q = torch.cat([torch.tensor([0], device=device),torch.cumsum(q_lengths, dim=0)]).type(torch.int32)
        cu_k = torch.cat([torch.tensor([0], device=device),torch.cumsum(kv_lengths, dim=0)]).type(torch.int32)                  


        # 计算 max seq lens
        max_q, max_k = int(q_lengths.max()), int(kv_lengths.max())

        # 调用变长 FlashAttention 核心函数
        out_cat = flash_attn_varlen_func(
            q_cat, k_cat, v_cat,
            cu_seqlens_q=cu_q,
            cu_seqlens_k=cu_k,
            max_seqlen_q=max_q,
            max_seqlen_k=max_k,
            dropout_p=dropout,
            causal=False
        )  # [total_q, H, D]

        # 还原回 batch
        outs = []
        start = 0
        for l in q_lengths:
            outs.append(out_cat[start:start + l].transpose(0, 1))
            start += l

        # pad 回统一 shape，方便后续层处理
        max_len = int(q_lengths.max())
        padded = query.new_zeros((B, H, max_len, D))
        for i, o in enumerate(outs):
            padded[i, :, :o.shape[1], :] = o
        return padded, None

    #  分支2：普通 Attention（不使用 Flash）
    else:
        if mask is not None:
            assert isinstance(mask, torch.Tensor), f'mask should be bool type'
        elif masksize != 1:
            masksize = int(masksize / 2)
            mask = torch.ones([B, H, L_q, L_k], device=device)
            for i in range(L_q):
                if i - masksize > 0:
                    mask[:, :, i, :i - masksize] = 0
                if i + masksize + 1 < L_k:
                    mask[:, :, i, masksize + i + 1:] = 0
        else:
            mask = None
        out = F.scaled_dot_product_attention(query, key, value, attn_mask=mask, dropout_p=dropout)
        return out, None


def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class MultiHeadAttention(nn.Module):
    def __init__(self, h, d_model, masksize=1, dropout=0.1, use_flash_attn=True, use_rotate_pe=False):
        super(MultiHeadAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.masksize = masksize
        self.dropout = dropout
        self.use_flash_attn = use_flash_attn 
        self.use_rotate_pe = use_rotate_pe
        if self.use_rotate_pe:
            self.compute_cis = partial(
                compute_axial_cis, dim=self.d_k, theta=10000.0
            )
            self.rope_k_repeat = False

# def attention(query, key, value,  q_lengths=None, kv_lengths=None, masksize=1, mask = None ,dropout=0.0, use_flash_attn=False):   
    def forward(self, query, key, value, num_q_exclude_rope = 0, num_k_exclude_rope=0, **kwargs):
        nbatches = query.size(0)
        query, key, value = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2) for l, x in
                             zip(self.linears, (query, key, value))] # [B,L,nHead,C] -> [B,nHead,L,C]
        
        if self.use_rotate_pe:
            # Apply rotary position encoding
            num_q_rope = query.size(-2) - num_q_exclude_rope
            num_k_rope = key.size(-2) - num_k_exclude_rope
            # w = h = math.sqrt(num_q_rope)
            h,w = kwargs.get('emb_H_W',(0,0))
            assert h*w == num_q_rope
            self.freqs_cis = self.compute_cis(end_x=w, end_y=h).to(query.device)

            if num_q_rope != num_k_rope:
                self.rope_k_repeat = True

            query[:, :, :num_q_rope], key[:, :, :num_k_rope] = apply_rotary_enc(
                query[:, :, :num_q_rope],
                key[:, :, :num_k_rope],
                freqs_cis=self.freqs_cis,
                repeat_freqs_k=self.rope_k_repeat,
            )

        # 变长 attn
        kwds = {
                "q_lengths": kwargs.get('q_lengths',None),
                "kv_lengths": kwargs.get('kv_lengths',None),
                "masksize": kwargs.get('masksize',1),
                "use_flash_attn":self.use_flash_attn
               }

        x, self.attn = attention(query, key, value, dropout=self.dropout, **kwds)
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        out = self.linears[-1](x)
        return out, self.attn

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        output = self.w_2(self.dropout(F.relu(self.w_1(x))))
        return output


'''
SAM Attention Module with RoPE and Flash Attention
'''
import contextlib
import warnings
from functools import partial

def get_sdpa_settings():
    if torch.cuda.is_available():
        old_gpu = torch.cuda.get_device_properties(0).major < 7
        # only use Flash Attention on Ampere (8.0) or newer GPUs
        use_flash_attn = torch.cuda.get_device_properties(0).major >= 8
        if not use_flash_attn:
            warnings.warn(
                "Flash Attention is disabled as it requires a GPU with Ampere (8.0) CUDA capability.",
                category=UserWarning,
                stacklevel=2,
            )
        # keep math kernel for PyTorch versions before 2.2 (Flash Attention v2 is only
        # available on PyTorch 2.2+, while Flash Attention v1 cannot handle all cases)
        pytorch_version = tuple(int(v) for v in torch.__version__.split(".")[:2])
        if pytorch_version < (2, 2):
            warnings.warn(
                f"You are using PyTorch {torch.__version__} without Flash Attention v2 support. "
                "Consider upgrading to PyTorch 2.2+ for Flash Attention v2 (which could be faster).",
                category=UserWarning,
                stacklevel=2,
            )
        math_kernel_on = pytorch_version < (2, 2) or not use_flash_attn
    else:
        old_gpu = True
        use_flash_attn = False
        math_kernel_on = True

    return old_gpu, use_flash_attn, math_kernel_on

warnings.simplefilter(action="ignore", category=FutureWarning)
# Check whether Flash Attention is available (and use it by default)
OLD_GPU, USE_FLASH_ATTN, MATH_KERNEL_ON = get_sdpa_settings()
# A fallback setting to allow all available kernels if Flash Attention fails
ALLOW_ALL_KERNELS = False
def sdp_kernel_context(dropout_p):
    """
    Get the context for the attention scaled dot-product kernel. We use Flash Attention
    by default, but fall back to all available kernels if Flash Attention fails.
    """
    if ALLOW_ALL_KERNELS:
        return contextlib.nullcontext()

    return torch.backends.cuda.sdp_kernel(
        enable_flash=USE_FLASH_ATTN,
        # if Flash attention kernel is off, then math kernel needs to be enabled
        enable_math=(OLD_GPU and dropout_p > 0.0) or MATH_KERNEL_ON,
        enable_mem_efficient=OLD_GPU,)
    # return torch.nn.attention.sdpa_kernel(
    #     [SDPBackend.FLASH_ATTENTION,SDPBackend.EFFICIENT_ATTENTION,SDPBackend.MATH]
    # )


class RoPEAttention(nn.Module):
    """Attention with rotary position encoding."""

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        downsample_rate: int = 1,
        rope_theta=10000.0,
        # whether to repeat q rope to match k length
        # this is needed for cross-attention to memories
        rope_k_repeat=False,
        feat_sizes=(32, 32),  # [w, h] for stride 16 feats at 512 resolution
        **kwargs,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.internal_dim = embedding_dim // downsample_rate
        self.num_heads = num_heads
        assert self.internal_dim % num_heads == 0, "num_heads must divide embedding_dim."
        self.q_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.k_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.v_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.out_proj = nn.Linear(self.internal_dim, embedding_dim)

        self.compute_cis = partial(
            compute_axial_cis, dim=self.internal_dim // self.num_heads, theta=rope_theta
        )
        freqs_cis = self.compute_cis(end_x=feat_sizes[0], end_y=feat_sizes[1])
        self.freqs_cis = freqs_cis
        self.rope_k_repeat = rope_k_repeat
    
    def _separate_heads(self, x: Tensor, num_heads: int) -> Tensor:
        b, n, c = x.shape
        x = x.reshape(b, n, num_heads, c // num_heads)
        return x.transpose(1, 2)  # B x N_heads x N_tokens x C_per_head

    def _recombine_heads(self, x: Tensor) -> Tensor:
        b, n_heads, n_tokens, c_per_head = x.shape
        x = x.transpose(1, 2)
        return x.reshape(b, n_tokens, n_heads * c_per_head)  # B x N_tokens x C

    def forward(
        self, q: Tensor, k: Tensor, v: Tensor, num_k_exclude_rope: int = 0
    ) -> Tensor:
        # Input projections
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        # Separate into heads
        q = self._separate_heads(q, self.num_heads)
        k = self._separate_heads(k, self.num_heads)
        v = self._separate_heads(v, self.num_heads)

        # Apply rotary position encoding
        w = h = math.sqrt(q.shape[-2])
        self.freqs_cis = self.freqs_cis.to(q.device)
        if self.freqs_cis.shape[0] != q.shape[-2]:
            self.freqs_cis = self.compute_cis(end_x=w, end_y=h).to(q.device)
        if q.shape[-2] != k.shape[-2]:
            assert self.rope_k_repeat

        num_k_rope = k.size(-2) - num_k_exclude_rope
        q, k[:, :, :num_k_rope] = apply_rotary_enc(
            q,
            k[:, :, :num_k_rope],
            freqs_cis=self.freqs_cis,
            repeat_freqs_k=self.rope_k_repeat,
        )

        dropout_p = self.dropout_p if self.training else 0.0
        # Attention
        try:
            with sdp_kernel_context(dropout_p):
                out = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p)
        except Exception as e:
            # Fall back to all kernels if the Flash attention kernel fails
            warnings.warn(
                f"Flash Attention kernel failed due to: {e}\nFalling back to all available "
                f"kernels for scaled_dot_product_attention (which may have a slower speed).",
                category=UserWarning,
                stacklevel=2,
            )
            global ALLOW_ALL_KERNELS
            ALLOW_ALL_KERNELS = True
            out = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p)

        out = self._recombine_heads(out)
        out = self.out_proj(out)

        return out


