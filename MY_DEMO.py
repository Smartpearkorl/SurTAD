# # import matplotlib.pyplot as plt
# # plt.plot([1, 2, 3], [4, 5, 6])
# # plt.title("Test plt.show()")
# # plt.xlabel("x")
# # plt.ylabel("y")
# # plt.savefig('/home/qh/TDD/MemTAD/demo.jpg')
# # plt.show()
# # pass


# import torch
# import torch.nn.functional as F
# import torch.nn as nn
# import math
# import time
# from flash_attn.flash_attn_interface import flash_attn_varlen_qkvpacked_func, flash_attn_func 
# from flash_attn import flash_attn_varlen_func

# def attention(query, key, value,  q_lengths=None, kv_lengths=None, masksize=1, mask = None ,dropout=0.0, use_flash_attn=False):         
#     """
#     通用注意力函数，支持变长 FlashAttention。
    
#     Args:
#         query: [B, H, L_q, D]
#         key:   [B, H, L_k, D]
#         value: [B, H, L_v, D]
#         q_lengths: list[int] or tensor[B], 每个样本的 query 有效长度
#         kv_lengths: list[int] or tensor[B], 每个样本的 key/value 有效长度
#         masksize: 窗口大小，仅用于非 flash 模式
#         dropout: float
#         use_flash_attn: 是否使用 FlashAttention 变长实现
#     """
#     B, H, L_q, D = query.shape
#     _, _, L_k, _ = key.shape
#     device = query.device

#     # 分支1：变长 FlashAttention 模式
#     if use_flash_attn:
#         # 如果没有传长度信息，fallback 到普通 flash_attn_func
#         if q_lengths is None and kv_lengths is None:
#             # 走普通 flash attention
#             out = flash_attn_func(
#                 query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2),
#                 dropout_p=dropout
#             ).transpose(1, 2)
#             return out, None
        
#         # 检查长度输入
#         # 构造默认长度
#         if q_lengths is None:
#             q_lengths = [query.shape[2]] * query.shape[0]
#         if kv_lengths is None:
#             kv_lengths = [key.shape[2]] * key.shape[0]
#         assert len(q_lengths) == B and len(kv_lengths) == B, "lengths size mismatch with batch"

#         # 把 list 转成 tensor
#         q_lengths = torch.as_tensor(q_lengths, dtype=torch.int32, device=device)
#         kv_lengths = torch.as_tensor(kv_lengths, dtype=torch.int32, device=device)

#         # 拼接有效 token
#         q_list, k_list, v_list = [], [], []
#         for i in range(B):
#             q_list.append(query[i, :, :q_lengths[i], :])
#             k_list.append(key[i, :, :kv_lengths[i], :])
#             v_list.append(value[i, :, :kv_lengths[i], :])
#         q_cat = torch.cat(q_list, dim=1).transpose(0, 1).contiguous()  # [total_q, H, D]
#         k_cat = torch.cat(k_list, dim=1).transpose(0, 1).contiguous()
#         v_cat = torch.cat(v_list, dim=1).transpose(0, 1).contiguous()

#         # 构造 cumulative sequence lengths
#         cu_q = torch.cat([torch.tensor([0], device=device),torch.cumsum(q_lengths, dim=0)]).type(torch.int32)
#         cu_k = torch.cat([torch.tensor([0], device=device),torch.cumsum(kv_lengths, dim=0)]).type(torch.int32)                  


#         # 计算 max seq lens
#         max_q, max_k = int(q_lengths.max()), int(kv_lengths.max())

#         # 调用变长 FlashAttention 核心函数
#         out_cat = flash_attn_varlen_func(
#             q_cat, k_cat, v_cat,
#             cu_seqlens_q=cu_q,
#             cu_seqlens_k=cu_k,
#             max_seqlen_q=max_q,
#             max_seqlen_k=max_k,
#             dropout_p=dropout,
#             causal=False
#         )  # [total_q, H, D]

#         # 还原回 batch
#         outs = []
#         start = 0
#         for l in q_lengths:
#             outs.append(out_cat[start:start + l].transpose(0, 1))
#             start += l

#         # pad 回统一 shape，方便后续层处理
#         max_len = int(q_lengths.max())
#         padded = query.new_zeros((B, H, max_len, D))
#         for i, o in enumerate(outs):
#             padded[i, :, :o.shape[1], :] = o
#         return padded, None

#     #  分支2：普通 Attention（不使用 Flash）
#     else:
#         if mask is not None:
#             assert isinstance(mask, torch.Tensor), f'mask should be bool type'
#         elif masksize != 1:
#             masksize = int(masksize / 2)
#             mask = torch.ones([B, H, L_q, L_k], device=device)
#             for i in range(L_q):
#                 if i - masksize > 0:
#                     mask[:, :, i, :i - masksize] = 0
#                 if i + masksize + 1 < L_k:
#                     mask[:, :, i, masksize + i + 1:] = 0
#         else:
#             mask = None
#         out = F.scaled_dot_product_attention(query, key, value, attn_mask=mask, dropout_p=dropout)
#         return out, None


# def benchmark(func, *args, **kwargs):
#     torch.cuda.synchronize()
#     start = time.time()
#     out,_ = func(*args,**kwargs)
#     torch.cuda.synchronize()
#     elapsed = time.time() - start
#     return out, elapsed

# def test_attention_consistency():
#     # ---------- 测试 ----------
#     B, H, L, D = 2, 8, 1024, 256  # batch, heads, seq_len, head_dim
#     torch.manual_seed(42)

#     q = torch.randn(B, H, L, D, dtype=torch.float16, device='cuda')
#     k = torch.randn(B, H, 2*L, D, dtype=torch.float16, device='cuda')
#     v = torch.randn(B, H, 2*L, D, dtype=torch.float16, device='cuda')

#     dropout = 0.0

#     # --------------------- PyTorch Attention ---------------------
#     torch.cuda.reset_peak_memory_stats()
#     out_pt, t_pt = benchmark(attention, q, k, v, dropout=dropout)
#     mem_pt = torch.cuda.max_memory_allocated() / 1024**2

#     # --------------------- FlashAttention ---------------------
#     torch.cuda.reset_peak_memory_stats()
#     out_fa, t_fa = benchmark(attention,  q, k, v, dropout=dropout, use_flash_attn=True)
#     mem_fa = torch.cuda.max_memory_allocated() / 1024**2

#     max_diff = (out_pt - out_fa).abs().max().item()
#     print(f"PyTorch Attention耗时:  {t_pt*1000:.2f} ms, 显存: {mem_pt:.1f} MB")
#     print(f"FlashAttention耗时:     {t_fa*1000:.2f} ms, 显存: {mem_fa:.1f} MB")
#     print(f"结果最大差异: {max_diff:.6f}")
#     print(f"是否接近一致: {torch.allclose(out_pt, out_fa, atol=1e-2)}")


# def pad_to_max_len(tensors, max_len):
#     B = len(tensors)
#     shapes = tensors[0].shape
#     if len(shapes) == 2: # [N,D]
#         H, D = tensors[0].shape[0], tensors[0].shape[1]
#         padded = tensors[0].new_zeros((B, max_len, D))
#         for i, t in enumerate(tensors):
#             padded[i, :t.shape[0], :] = t
#         return padded
#     elif len(shapes) == 3: # [N,H,D]
#         H, D = tensors[0].shape[1], tensors[0].shape[2]
#         padded = tensors[0].new_zeros((B, H, max_len, D))
#         for i, t in enumerate(tensors):
#             padded[i, :, :t.shape[0], :] = t.transpose(0, 1)
#         return padded

# def test_varlen_attention_consistency():
#     print("\n====== 测试 变长 FlashAttention 与 普通Attention 一致性 ======")
#     torch.manual_seed(0)
#     torch.cuda.manual_seed(0)

#     B, H, D = 4, 8, 64
#     lengths = q_lengths = [8, 4, 2, 1]
#     k_lengths = [7, 6, 5, 4]
#     # lengths = q_lengths = [8] *4
#     # k_lengths = [7]*4
#     q_max_len = max(q_lengths)
#     k_max_len = max(k_lengths)

#     # 构造变长输入
#     q_list = [torch.randn(l, H, D, dtype=torch.float16, device='cuda') for l in q_lengths]
#     k_list = [torch.randn(l, H, D, dtype=torch.float16, device='cuda') for l in k_lengths]
#     v_list = [torch.randn(l, H, D, dtype=torch.float16, device='cuda') for l in k_lengths]

#     # pad 版本（普通 attention 使用）
#     q_pad = pad_to_max_len(q_list, q_max_len)
#     k_pad = pad_to_max_len(k_list, k_max_len)
#     v_pad = pad_to_max_len(v_list, k_max_len)

#     # --------------------- 普通 Attention（加mask模拟变长） ---------------------
#     mask = torch.ones(B, 1, q_max_len, k_max_len, device='cuda', dtype=torch.bool)
#     for i, l in enumerate(q_lengths):
#         mask[i, :,  l:, :] = False  # mask掉超出长度的部分
#     for i, l in enumerate(k_lengths):
#         mask[i, :,  :, l:] = False  # mask掉超出长度的部分

#     out_ref, _ = attention(q_pad, k_pad, v_pad, mask=mask, dropout=0.0, use_flash_attn=False)
#     out_ref_masked = []
#     for i, l in enumerate(lengths):
#         out_ref_masked.append(out_ref[i, :, :l, :].detach())

#     # --------------------- 变长 FlashAttention ---------------------
#     out_varlen, _ = attention(q_pad, k_pad, v_pad,
#                               q_lengths=q_lengths, kv_lengths=k_lengths,
#                               dropout=0.0, use_flash_attn=True)
#     out_varlen_masked = []
#     for i, l in enumerate(lengths):
#         out_varlen_masked.append(out_varlen[i, :, :l, :].detach())

#     # --------------------- 对比每个样本结果 ---------------------
#     all_diffs = []
#     for i, (r, f) in enumerate(zip(out_ref_masked, out_varlen_masked)):
#         diff = (r - f).abs().max().item()
#         all_diffs.append(diff)
#         print(f"样本{i}: 长度={lengths[i]}, 最大差异={diff:.6f}")

#     print(f"平均差异={sum(all_diffs)/len(all_diffs):.6f}")
#     print(f"是否整体一致: {all(d < 1e-2 for d in all_diffs)}")

# # 运行测试
# test_varlen_attention_consistency()



# import torch
# import torch.nn.functional as F
# import torch.nn as nn
# import math
# import time
# from flash_attn.flash_attn_interface import flash_attn_varlen_qkvpacked_func, flash_attn_func 
# from flash_attn import flash_attn_varlen_func

# def memory_bank_compress(memory_bank: torch.Tensor, compression_size: torch.Tensor) -> tuple:
#     """
#     Compresses the memory bank if the current memory bank length is greater than the threshold.
#     Compression_size is the number of frames that are compressed into each position.
    
#     Args:
#         memory_bank (torch.Tensor): The input memory bank to compress. Shape: (B, T, N, C)
#         compression_size (torch.Tensor): The number of frames to compress into each position. Shape: (B, T, N)
    
#     Returns:
#         compressed_memory_bank (torch.Tensor): The compressed memory bank. Shape: (B, T-1, N, C)
#         compressed_size (torch.Tensor): The number of frames compressed into each position. Shape: (B, T-1, N)
#     """
#     B, T, N, C = memory_bank.shape
#     # Calculate the cosine similarity between adjacent frames
#     similarity_matrix = F.cosine_similarity(memory_bank[:, :-1, :], memory_bank[:, 1:, :], dim=-1)
#     # Select the frame indices with the top-1 similarity 
#     _, max_similarity_indices = torch.max(similarity_matrix, dim=1, keepdim=True)

#     # Calculate source and dst indices for compression
#     src_indices = max_similarity_indices + 1
#     dst_indices = torch.arange(T - 1).to(memory_bank.device)[None, :, None].repeat(B, 1, N)
#     dst_indices[dst_indices > max_similarity_indices] += 1

#     # Gather source and dst memory banks and sizes
#     src_memory_bank = memory_bank.gather(dim=1, index=src_indices.unsqueeze(-1).expand(-1, -1, -1, C))
#     dst_memory_bank = memory_bank.gather(dim=1, index=dst_indices.unsqueeze(-1).expand(-1, -1, -1, C))
#     src_size = compression_size.gather(dim=1, index=src_indices)
#     dst_size = compression_size.gather(dim=1, index=dst_indices)

#     # Multiply the memory banks by their corresponding sizes
#     src_memory_bank *= src_size.unsqueeze(-1)
#     dst_memory_bank *= dst_size.unsqueeze(-1)

#     # Compress the memory bank by adding the source memory bank to the dst memory bank
#     dst_memory_bank.scatter_add_(dim=1, index=max_similarity_indices.unsqueeze(-1).expand(-1, -1, -1, C), src=src_memory_bank)
#     dst_size.scatter_add_(dim=1, index=max_similarity_indices, src=src_size)

#     # Normalize the dst memory bank by its size
#     compressed_memory_bank = dst_memory_bank / dst_size.unsqueeze(-1)
#     return compressed_memory_bank, dst_size

# torch.manual_seed(0)
# torch.cuda.manual_seed(0)

# B, T, N, D = 4, 5, 8, 64

# org_memory_bank = torch.randn(B, T, N, D, dtype=torch.float16, device='cuda')
# compression_size = torch.ones(B, T, N, dtype=torch.float16, device='cuda')
# compressed_memory_bank, compressed_size = memory_bank_compress(org_memory_bank, compression_size)
# pass

import sys
import os
import json
import random
from collections import defaultdict
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QPushButton, QLabel, QTreeWidget, QTreeWidgetItem, QSpinBox, 
                             QMessageBox, QFileDialog, QGroupBox)
from PyQt5.QtCore import Qt

class SceneSelector(QMainWindow):
    def __init__(self, metadata_path):
        super().__init__()
        self.setWindowTitle("验收场景挑选器 (Sub-class Selector)")
        self.setGeometry(200, 200, 800, 700)
        self.metadata_path = metadata_path
        
        # 数据结构: { "Collision:bike2bike": ["000768", "000729", ...], ... }
        self.grouped_scenes = defaultdict(list)
        self.total_scenes = 0
        
        self.initUI()
        self.load_metadata()

    def initUI(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)

        # ================= 左侧：树形选择器 =================
        left_layout = QVBoxLayout()
        left_layout.addWidget(QLabel("<b>视频场景列表 (按异常类别分组)</b>"))
        
        self.tree = QTreeWidget()
        self.tree.setHeaderLabel("异常类别 / 视频ID")
        self.tree.itemChanged.connect(self.on_item_changed)
        left_layout.addWidget(self.tree)

        # ================= 右侧：操作面板 =================
        right_layout = QVBoxLayout()
        
        # 随机抽样组
        sample_group = QGroupBox("自动随机抽样")
        sample_vbox = QVBoxLayout()
        
        box_layout = QHBoxLayout()
        box_layout.addWidget(QLabel("每类抽取数量:"))
        self.spin_n = QSpinBox()
        self.spin_n.setRange(1, 100)
        self.spin_n.setValue(3)
        box_layout.addWidget(self.spin_n)
        sample_vbox.addLayout(box_layout)
        
        self.btn_random = QPushButton("一键随机抽样")
        self.btn_random.setStyleSheet("background-color: #2196F3; color: white; height: 35px;")
        self.btn_random.clicked.connect(self.random_sample)
        sample_vbox.addWidget(self.btn_random)
        
        self.btn_clear = QPushButton("清空所有选择")
        self.btn_clear.clicked.connect(self.clear_selection)
        sample_vbox.addWidget(self.btn_clear)
        
        sample_group.setLayout(sample_vbox)
        right_layout.addWidget(sample_group)
        
        # 状态与保存组
        status_group = QGroupBox("导出与保存")
        status_vbox = QVBoxLayout()
        
        self.lbl_status = QLabel("已选视频: 0")
        self.lbl_status.setStyleSheet("font-size: 14px; font-weight: bold; color: green;")
        status_vbox.addWidget(self.lbl_status)
        
        self.btn_save = QPushButton("导出选中场景至 JSON")
        self.btn_save.setStyleSheet("background-color: #4CAF50; color: white; height: 40px; font-weight: bold;")
        self.btn_save.clicked.connect(self.save_selection)
        status_vbox.addWidget(self.btn_save)
        
        status_group.setLayout(status_vbox)
        right_layout.addWidget(status_group)
        right_layout.addStretch()

        # 比例布局
        main_layout.addLayout(left_layout, 2)
        main_layout.addLayout(right_layout, 1)

    def load_metadata(self):
        if not os.path.exists(self.metadata_path):
            QMessageBox.warning(self, "错误", f"找不到 Metadata 文件: \n{self.metadata_path}")
            return
            
        with open(self.metadata_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
            
        for vid, info in self.data.items():
            cat = info.get('anomaly_class', 'Unknown')
            self.grouped_scenes[cat].append(vid)
            self.total_scenes += 1
            
        self.populate_tree()

    def populate_tree(self):
        self.tree.clear()
        self.tree.blockSignals(True)
        
        for cat, vids in sorted(self.grouped_scenes.items()):
            # 父节点 (类别)
            parent_item = QTreeWidgetItem(self.tree)
            parent_item.setText(0, f"{cat} ({len(vids)} 个)")
            parent_item.setFlags(parent_item.flags() | Qt.ItemIsUserCheckable | Qt.ItemIsTristate)
            parent_item.setCheckState(0, Qt.Unchecked)
            parent_item.setExpanded(False)
            
            # 子节点 (视频ID)
            for vid in sorted(vids):
                child_item = QTreeWidgetItem(parent_item)
                child_item.setText(0, vid)
                child_item.setFlags(child_item.flags() | Qt.ItemIsUserCheckable)
                child_item.setCheckState(0, Qt.Unchecked)
                # 藏一个真实 ID 到 UserRole 方便提取
                child_item.setData(0, Qt.UserRole, vid) 
                
        self.tree.blockSignals(False)
        self.update_status()

    def on_item_changed(self, item, column):
        self.update_status()

    def clear_selection(self):
        self.tree.blockSignals(True)
        for i in range(self.tree.topLevelItemCount()):
            parent = self.tree.topLevelItem(i)
            parent.setCheckState(0, Qt.Unchecked)
            for j in range(parent.childCount()):
                parent.child(j).setCheckState(0, Qt.Unchecked)
        self.tree.blockSignals(False)
        self.update_status()

    def random_sample(self):
        self.clear_selection()
        n = self.spin_n.value()
        self.tree.blockSignals(True)
        
        for i in range(self.tree.topLevelItemCount()):
            parent = self.tree.topLevelItem(i)
            child_count = parent.childCount()
            
            # 随机挑选索引
            sample_size = min(n, child_count)
            sampled_indices = random.sample(range(child_count), sample_size)
            
            for idx in sampled_indices:
                parent.child(idx).setCheckState(0, Qt.Checked)
                
        self.tree.blockSignals(False)
        self.update_status()

    def get_selected_scenes(self):
        selected = []
        for i in range(self.tree.topLevelItemCount()):
            parent = self.tree.topLevelItem(i)
            for j in range(parent.childCount()):
                child = parent.child(j)
                if child.checkState(0) == Qt.Checked:
                    selected.append(child.data(0, Qt.UserRole))
        return selected

    def update_status(self):
        selected = self.get_selected_scenes()
        self.lbl_status.setText(f"已选视频: {len(selected)} / {self.total_scenes}")

    def save_selection(self):
        selected = self.get_selected_scenes()
        if not selected:
            QMessageBox.warning(self, "警告", "当前未选中任何视频！")
            return
        # 设置默认保存目录和默认文件名
        default_dir = "/data/qh/STDA/output/acceptance_workspace/metadata"
        os.makedirs(default_dir, exist_ok=True)   
        default_path = os.path.join(default_dir, "selected_metadata_val.json")
        
        save_path, _ = QFileDialog.getSaveFileName(
            self, 
            "保存选中场景", 
            default_path,  # 传入完整的绝对路径
            "JSON Files (*.json)"
        )
        if save_path:
            selected_data = {k:self.data[k] for k in selected}
            with open(save_path, 'w', encoding='utf-8') as f:     
                json.dump(selected_data, f, indent=4)
            QMessageBox.information(self, "成功", f"成功导出了 {len(selected)} 个场景的列表！\n可以在 AcceptancePipeline 中读取该文件。")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    # 指向你的 metadata json 文件
    META_PATH = "/data/qh/STDA/data/metadata/[in]_metadata_val.json" 
    
    ex = SceneSelector(META_PATH)
    ex.show()
    sys.exit(app.exec_())
