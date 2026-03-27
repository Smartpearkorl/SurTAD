import os
import torch
import numpy as np
import pickle
import json
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image
import scipy.signal as signal
import torch.distributed as dist


# ===== backend flag =====
USE_NON_GUI_BACKEND = True
if USE_NON_GUI_BACKEND:
    import matplotlib
    matplotlib.use("Agg")

# --- 导入现有的模块 ---
from alchemy_cat.dl_config import load_config
from runner.test import pama_test
from runner.src.dataset import prepare_dataset, stad_anomalies
from runner.src.utils import resume_from_checkpoint, load_results, get_result_filename
from runner.src.tools import init_distributed, setup_seed  # 引入分布式初始化工具
from runner.src.metrics import (evaluation, print_results, write_results, 
                                calculate_metrics_multiclass, calculate_per_class_map, 
                                Accuracy_on_scene, safe_auc, normalize_video)

# 映射表定义
SUB_ANOMALIES = ['normal'] + stad_anomalies

def parse_config():
    """完全还原 main.py 的参数解析逻辑，支持 DDP 启动"""
    parser = argparse.ArgumentParser(description='PromptTAD Acceptance Pipeline')

    parser.add_argument('--local_rank', '--local-rank', type=int, default=0,
                        help='local rank passed from distributed launcher')
    parser.add_argument('--distributed', action='store_true',default=True,
                        help='if DDP is applied.')
    
    parser.add_argument('--testdata', default='None',
                        help='if DDP is applied.')
    # 修改点：设置 default=True，默认开启混合精度加速
    parser.add_argument('--fp16', action='store_true', default=True,
                        help='if fp16 is applied. (Default: True)')
    
    parser.add_argument('--phase', default='test', choices=['test', 'train', 'play'],
                        help='Training or testing or play phase.')
    parser.add_argument('--num_workers', type=int, default=4, metavar='N')
    parser.add_argument('--epoch', type=int, default=24, 
                        help='The epoch to eval (testing).')
    parser.add_argument('--config', default='no_config')
    parser.add_argument('--output', default="/data/qh/STDA/output/in,subcls,ep=24,lr=1e-5,plain/",
                        help='Directory where save the output.')
    
    # 可视化特有参数
    parser.add_argument('--vis_count', type=int, default=5, help='Number of top scenes to visualize')
    parser.add_argument('--last_frame', action='store_true', default=False, help='Only plot the last frame of the scene')
    parser.add_argument('--add_popr', action='store_true', help='Plot both raw and smoothed scores')
    parser.add_argument('--only_popr', action='store_true', help='Plot ONLY smoothed scores')
    
    args = parser.parse_args()
    cfg = vars(args)

    device = torch.device(f'cuda:{cfg["local_rank"]}') if torch.cuda.is_available() else torch.device('cpu')
    n_nodes = torch.cuda.device_count()
    cfg.update(device=device)
    cfg.update(n_nodes=n_nodes)
    return cfg


class AcceptancePipeline:
    def __init__(self, parse_cfg):
        self.parse_cfg = parse_cfg
        self.output_path = parse_cfg['output']
        
        # 1. 加载与合并配置
        self.SoC = load_config(parse_cfg['config'])
        self.SoC.unfreeze()  # 解冻以允许修改
        self.basecfg = self.SoC.basecfg
        self.basecfg.basic.update(self.parse_cfg)
        self.cfg = self.basecfg.basic
        
        # 2. 分布式环境初始化 & 随机种子
        init_distributed(self.cfg)
        setup_seed(self.cfg.seed if hasattr(self.cfg, 'seed') else 42)
        
        # 3. 准备数据
        if self.cfg.local_rank == 0:
            print(f"[*] Initializing model and loading weights from: {self.output_path}/checkpoints/")
        
        if self.parse_cfg.get('testdata') != None:
            self.SoC.datacfg.test_dataset.data_type = self.parse_cfg.get('testdata')

        print(f'testdata name: {self.SoC.datacfg.test_dataset.data_type}')
        _, self.test_sampler, _, self.testdata_loader = prepare_dataset(
            self.cfg, self.SoC.datacfg.train_dataset, self.SoC.datacfg.test_dataset
        )
        
        # 4. 模型实例化
        model_cfg = self.SoC.modelcfg
        self.model = model_cfg.model(
            vst_cfg=model_cfg.vst, vit_cfg=model_cfg.vit, fpn_cfg=model_cfg.fpn,
            ins_encoder_cfg=model_cfg.ins_encoder, ins_decoder_cfg=model_cfg.ins_decoder, 
            ano_decoder_cfg=model_cfg.ano_decoder, proxy_task_cfg=model_cfg.proxy_task
        )
        
        # 5. DDP / DP 挂载逻辑
        if self.cfg.distributed:
            self.model.cuda(self.cfg.local_rank)
            self.model = torch.nn.parallel.DistributedDataParallel(
                self.model, device_ids=[self.cfg.local_rank], find_unused_parameters=True
            )
        else:
            self.model.to(self.cfg.device)
            self.model = torch.nn.DataParallel(self.model)
        
        # 6. 加载权重
        resume_from_checkpoint(self.cfg, self.model.module, None, None)
        self.model.eval()

    def post_process_data(self, preds, kernel_size=31):
        preds = np.array(preds)
        if len(preds) > kernel_size:
            preds = signal.medfilt(preds, kernel_size=kernel_size)
        else:
            ks = len(preds) if len(preds) % 2 != 0 else len(preds) - 1
            if ks > 0:
                preds = signal.medfilt(preds, kernel_size=ks)
        return normalize_video(np.array(preds))

    def run(self):
        pkl_path = get_result_filename(self.cfg, self.cfg.epoch)
        strict_test = False
        # --- 阶段 1：多卡并行推理 ---
        if strict_test or not os.path.exists(pkl_path):
            if self.cfg.local_rank == 0:
                print(f"[*] Inference results not found. Running distributed test...")
            with torch.no_grad():
                pama_test(self.cfg, self.model, self.test_sampler, self.testdata_loader, self.cfg.epoch, pkl_path)
        
        # 等待所有卡推理完成并销毁非主进程，防止冲突
        if dist.is_initialized() and self.cfg.local_rank != 0:
            dist.destroy_process_group()
            return # 子进程完成推理任务后直接退出
            
        # --- 阶段 2：仅在主进程 (local_rank == 0) 执行指标计算与绘图 ---
        if self.cfg.local_rank == 0:
            self._evaluate_and_visualize(pkl_path)

    def _evaluate_and_visualize(self, pkl_path):
        content = load_results(pkl_path)
        per_class = self.SoC.datacfg.train_dataset.name in ['dota', 'stad']
        
        # A. 二分类指标评估
        results = evaluation(**content, post_process=True, per_class=per_class)
        print("\n" + "="*20 + " ACCEPTANCE EVALUATION " + "="*20)
        print_results(self.cfg, *results)
        
        txt_path = os.path.join(self.output_path, 'evaluation', 'eval_acceptance.txt')
        os.makedirs(os.path.dirname(txt_path), exist_ok=True)
        write_results(txt_path, self.cfg.epoch, *results)

        # B. 多分类子类分析
        if self.cfg.get('apply_sub_class', False) and 'sub_targets' in content:
            flat_sub_targets = np.concatenate(content['sub_targets'], axis=0)
            flat_sub_outputs = np.concatenate(content['sub_outputs'], axis=0)
            valid_idx = flat_sub_targets != -100
            
            if len(flat_sub_targets[valid_idx]) > 0:
                _, mAP, ap_report = calculate_per_class_map(
                    flat_sub_outputs[valid_idx], flat_sub_targets[valid_idx], class_names=SUB_ANOMALIES
                )
                print(ap_report)
                with open(txt_path, 'a') as f:
                    f.write(ap_report)

        # C. 自动筛选与可视化
        vis_count = self.cfg.get('vis_count', 1)
        last_frame = self.cfg.get('last_frame', False)
        add_popr = self.cfg.get('add_popr', False)
        only_popr = self.cfg.get('only_popr', False)
        
        self._visualize_demo(pkl_path, content, vis_count, last_frame, add_popr, only_popr)

    def _visualize_demo(self, pkl_path, pkl_data, count, last_frame, add_popr, only_popr):
        print(f"\n[*] Visualizing top {count} scenes (add_popr={add_popr}, only_popr={only_popr})...")
        scene_accs = Accuracy_on_scene(pkl_path, acc_type='frame', post_process=False)
        best_scenes = list(scene_accs.keys())[-count:] 

        vis_dir = os.path.join(self.output_path, 'anomaly_score', 'Acceptance_Demo')
        os.makedirs(vis_dir, exist_ok=True)
        
        NF = self.cfg.get('NF', 4)
        img_folder = os.path.join(self.SoC.datacfg.test_dataset.root_path, "frames")
        gt_folder = os.path.join(self.SoC.datacfg.test_dataset.root_path, "annotations")

        for scene in tqdm(best_scenes, desc="Drawing Scenes"):
            idx = np.where(pkl_data['video_name'] == scene)[0][0]
            
            raw_outputs = pkl_data['outputs'][idx]
            
            plot_scores = {}
            if only_popr:
                plot_scores['Smoothed'] = self.post_process_data(raw_outputs)
            else:
                plot_scores['Raw'] = raw_outputs
                if add_popr:
                    plot_scores['Smoothed'] = self.post_process_data(raw_outputs)
            
            sub_outputs = pkl_data['sub_outputs'][idx] if 'sub_outputs' in pkl_data else None
            sub_targets = pkl_data['sub_targets'][idx] if 'sub_targets' in pkl_data else None

            with open(os.path.join(gt_folder, f"{scene}.json"), 'r') as f:
                gt_data = json.load(f)
            
            anomaly_start, anomaly_end = gt_data['anomaly_start'], gt_data['anomaly_end']
            frames_path = [os.path.join(img_folder, x['image_path']) for x in gt_data['labels']]
            frames = np.array(list(map(lambda x: np.asarray(Image.open(x)), frames_path)))
            frames_slice = frames[NF-1:-1] if NF else frames
            
            assert len(frames_slice) == len(raw_outputs), f"Misaligned! Frames: {len(frames_slice)}, Outputs: {len(raw_outputs)}"
            
            scene_save_path = os.path.join(vis_dir, f"[ACC_{scene_accs[scene]*100:.1f}]_{scene}")
            os.makedirs(scene_save_path, exist_ok=True)
            
            n_frames = len(raw_outputs)
            xvals = np.arange(n_frames)
            
            for index, frame in enumerate(tqdm(frames_slice, desc='Frames', leave=False)):
                if last_frame and index != len(frames_slice) - 1:
                    continue 

                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
                
                # ================= 顶部：原图 =================
                ax1.imshow(frame)
                title1 = f"Scene: {scene} | Overall ACC: {scene_accs[scene]*100:.1f}%"
                ax1.set_title(title1, fontsize=16, fontweight='bold', pad=15)
                ax1.axis('off')
                
                # ================= 中间：类别说明 =================
                # 把多分类的文字说明提取到这里，使用 fig.text 画在两张图中间
                if sub_outputs is not None and sub_targets is not None:
                    pred_sub_idx = np.argmax(sub_outputs[index])
                    gt_sub_idx = sub_targets[index]
                    
                    pred_name = SUB_ANOMALIES[pred_sub_idx]
                    gt_name = SUB_ANOMALIES[gt_sub_idx] if gt_sub_idx != -100 else 'Normal / Ignore'
                    
                    description_text = f"Pred Class: [ {pred_name} ]   |   GT Class: [ {gt_name} ]"
                    
                    # 动态控制文本颜色（如果预测和真值不同，可以用红色高亮提示，这里默认黑色）
                    text_color = 'red' if (pred_sub_idx != gt_sub_idx and gt_sub_idx != -100) else 'green'
                    
                    fig.text(0.5, 0.48, description_text, ha='center', va='center', fontsize=14, color=text_color, 
                             bbox=dict(facecolor='white', alpha=0.9, edgecolor='green', boxstyle='round,pad=0.5'))

                # ================= 底部：分数曲线 =================
                current_score_texts = [f"{name}: {scores[index]*100:.1f}" for name, scores in plot_scores.items()]
                ax2.set_title(f"Frame {index+NF-1} | " + " | ".join(current_score_texts), pad=15, fontsize=14)
                ax2.set_ylabel('Frame Anomaly Score', labelpad=5, fontsize=12)
                
                ax2.set_ylim(0, 1.05)
                ax2.set_xlim(0, n_frames)
                ax2.set_aspect(n_frames * 0.135)
                
                offset = NF - 1   
                shift_ticks = [tick - offset for tick in ax2.get_xticks() if tick >= offset]
                ax2.set_xticks(shift_ticks)
                ax2.set_xticklabels([int(x + offset) for x in shift_ticks], fontsize=12) 
                
                for label in ax2.get_yticklabels():  
                    label.set_fontsize(12) 
       
                # ================= 修复：异常区间边界对齐与防失效 =================
                # 首先确保它是一个有异常的视频 (通常 -1 表示 normal 视频)
                if anomaly_start >= 0 and anomaly_end >= 0:
                    toa = anomaly_start - NF + 1
                    tea = anomaly_end - NF + 1
                    
                    # 关键修改：边界裁剪对齐
                    # 防止 start < NF-1 导致 toa 为负数被直接丢弃
                    toa = max(0, toa)
                    # 防止 end 超出当前曲线最大长度
                    tea = min(n_frames - 1, tea)
                    
                    # 确保裁剪后依然是一个有效的正向区间（有可能整个异常都在 NF-1 之前结束了）
                    if toa <= tea:
                        ax2.axvline(x=toa, ymax=1.0, linewidth=1.5, color='r', linestyle='--')
                        ax2.axvline(x=tea, ymax=1.0, linewidth=1.5, color='r', linestyle='--')
                        ax2.fill_between([toa, tea], 0, 1, color='orange', alpha=0.3, interpolate=True, label="GT Interval")

                colors, linestyles = ['red', 'blue'], ['-', '--']
                for i, (name, scores) in enumerate(plot_scores.items()):
                    ax2.plot(xvals[:index+1], scores[:index+1], color=colors[i % len(colors)], 
                             linestyle=linestyles[i % len(linestyles)], linewidth=2.5, label=name)

                ax2.legend(loc='best', fontsize=12, frameon=True)
        
                # ================= 保存与布局调整 =================
                image_savepath = os.path.join(scene_save_path, f"frame_{index+NF-1:04d}.png")
                
                # 注意：这里去掉了 plt.tight_layout()，因为 tight_layout 会强制缩减子图间距，把咱们中间的字给盖住。
                # 改用 subplots_adjust 给中间文字硬留出 0.1 的间隙
                plt.subplots_adjust(hspace=0.1)  
                plt.savefig(image_savepath, dpi=150, bbox_inches='tight', pad_inches=0.1)  
                plt.close('all')

if __name__ == "__main__":
    pass
    # 解析命令行参数 (支持 torchrun / launch 启动)
    parsed_cfg = parse_config()
    
    # 你依然可以在此强行覆盖参数用于代码内调试，或者直接通过命令行传入
    parsed_cfg['config'] = "/home/qh/TDD/SurTAD/configs/train/vit/base_detection/stad_debug.py"
    parsed_cfg['testdata'] = 'selected_'
    # parsed_cfg['output'] = "/data/qh/STDA/output/acceptance_workspace/"
    parsed_cfg['output'] = "/data/qh/STDA/output/demo/"
    parsed_cfg['vis_count'] = 5
    parsed_cfg['last_frame'] = False
    parsed_cfg['add_popr'] = False
    
    pipeline = AcceptancePipeline(parsed_cfg)
    pipeline.run()