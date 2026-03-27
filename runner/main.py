from alchemy_cat.dl_config import load_config, Config ,Param2Tune,IL
from alchemy_cat.py_tools import Logger,get_local_time_str
import torch
import argparse
import os
import sys
import datetime
import torch.nn as nn
import torch.distributed as dist
from torch.cuda.amp import GradScaler as GradScaler

# ===== backend flag =====
USE_NON_GUI_BACKEND = False
if USE_NON_GUI_BACKEND:
    import matplotlib
    matplotlib.use("Agg")

# Custom imports
import sys
from pathlib import Path
FILE = Path(__file__).resolve() # /home/qh/TDD/pama/runner/main.py
sys.path.insert(0, str(FILE.parents[1]))
import os 
os.chdir(FILE.parents[1])

from runner.train import pama_train 
from runner.test import pama_test 
from runner.src.dataset import prepare_dataset, SUB_ANOMALIES
from runner.src.optimizer import prepare_optim_sched
from runner.src.tools import *
from runner.src.utils import resume_from_checkpoint , get_result_filename , load_results
from runner.src.metrics import evaluation, print_results , write_results , evaluation_on_obj, plot_figures_multiclass,\
                        calculate_metrics_multiclass, calculate_per_class_map, classification_report
# torch.autograd.set_detect_anomaly(True)
def parse_config():
    parser = argparse.ArgumentParser(description='PromptTAD implementation')

    parser.add_argument('--local_rank','--local-rank',
                        type=int,
                        default=0,
                        help='local rank passed from distributed launcher')

    parser.add_argument('--distributed',
                        action='store_true',
                        help='if DDP is applied.')
    
    parser.add_argument('--fp16',
                        action='store_true',
                        help='if fp16 is applied.')
    
    parser.add_argument('--phase',
                    default='train',
                    choices=['test', 'train', 'play'],
                    help='Training or testing or play phase.')
    
    parser.add_argument('--num_workers',
                    type = int,
                    default = 4,
                    metavar='N',)
    
    help_epoch = 'The epoch to restart from (training) or to eval (testing).'
    parser.add_argument('--epoch',
                        type=int,
                        default=-1,
                        help=help_epoch)

    parser.add_argument('--config',
                        default='no_config')
                        
    parser.add_argument('--output',
                        default = "/data/qh/STDA/output/debug/vscode_debug/",
                        # default = "/data/qh/DoTA/poma_v2/rnn/vst,base,dim=1024,fpn(0),prompt(0),rnn,vcl=8/",
                        help='Directory where save the output.')
    
    args = parser.parse_args()
    cfg = vars(args)

    device = torch.device(f'cuda:{cfg["local_rank"]}') if torch.cuda.is_available() else torch.device('cpu')
    n_nodes = torch.cuda.device_count()
    cfg.update(device=device)
    cfg.update(n_nodes=n_nodes)
    return cfg

if __name__ == "__main__":
    parse_cfg = parse_config()

    if parse_cfg['config'] == 'no_config':   
        # cfg_path = "/home/qh/TDD/MemTAD/configs/train/vst/mem_based_detection/vst_debug_config.py"
        # cfg_path = "/home/qh/TDD/MemTAD/configs/train/vit/mem_based_detection/debug_config.py"
        # cfg_path = "/home/qh/TDD/SurTAD/configs/train/vit/base_detection/lr=1e-5,plain.py"
        cfg_path = "/home/qh/TDD/SurTAD/configs/train/vit/base_detection/stad_debug.py"
    else:
        cfg_path = parse_cfg['config']

    SoC = load_config(cfg_path)
    basecfg , datacfg , modelcfg =SoC.basecfg , SoC.datacfg , SoC.modelcfg
    
    basecfg.basic.unfreeze()    
    basecfg.basic.update(parse_cfg)

    # basecfg.basic.directly_load = "/data/qh/DoTA/poma_v2/instance/vst,ins,prompt_mean,rnn,depth=4/checkpoints/model-200.pt"
    # basecfg.basic.whole_load = True
  
    init_distributed(basecfg.basic)
    setup_seed(basecfg.basic.seed)
    print(basecfg) 
    
    rank = basecfg.basic.local_rank
    name = f'{basecfg.basic.output}/{get_local_time_str(for_file_name=True)}-{rank=}.log'
    Logger(out_file = name, real_time = True)

    print('prepare dataset...')
    train_sampler, test_sampler, traindata_loader, testdata_loader = prepare_dataset(basecfg.basic, datacfg.train_dataset, datacfg.test_dataset)
    print('loading model...')

    if modelcfg.model_type == 'poma':                                  
         model =  modelcfg.model(   vst_cfg = modelcfg.vst,
                                    vit_cfg = modelcfg.vit,
                                    fpn_cfg = modelcfg.fpn,
                                    ins_encoder_cfg = modelcfg.ins_encoder , 
                                    ins_decoder_cfg = modelcfg.ins_decoder, 
                                    ano_decoder_cfg = modelcfg.ano_decoder,
                                    proxy_task_cfg = modelcfg.proxy_task)
 
    # freeze vit: misconvergence for sam
    # if datacfg.train_dataset.cfg.pre_process_type == 'rgb':
    #     freeze_vit_backbone(basecfg.basic.model_type,model)

    if basecfg.basic.distributed:
        model.cuda(basecfg.basic.local_rank)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[basecfg.basic.local_rank], find_unused_parameters=True)
    else:
        model.to(basecfg.basic.device)
        model = nn.DataParallel(model)

    if basecfg.basic.phase == 'train':
        print('prepare optimizer...')
        optimizer, lr_scheduler = prepare_optim_sched(model, basecfg.optimizer, basecfg.sched)
        ckp = resume_from_checkpoint(basecfg.basic,  model.module , optimizer , lr_scheduler)
        # resume summarywriter index for tensorboard
        index_video = ckp.get('index_video', 0)
        index_frame = ckp.get('index_frame', 0)
    else:
        resume_from_checkpoint(basecfg.basic,  model.module , None , None)
    
    # basecfg.basic.fp16 = True
    scaler = GradScaler(enabled=basecfg.basic.fp16)
    print(f'apply FP16 {basecfg.basic.fp16}')
        
    if basecfg.basic.phase=='train':
        # backup config
        backup_file(SoC, modelcfg, model)
        pama_train(basecfg.basic, model, train_sampler, traindata_loader, scaler,
                optimizer, lr_scheduler, test_sampler, testdata_loader, index_video, index_frame )
        
        if dist.is_initialized():
            dist.barrier()
            dist.destroy_process_group()
    
    strictly_test = False # False
    if basecfg.basic.phase=='test':
        cfg = basecfg.basic
        epoch = cfg.epoch 
        filename = get_result_filename(basecfg.basic, basecfg.basic.epoch)
        if not os.path.exists(filename) or strictly_test:
            with torch.no_grad():
                pama_test(basecfg.basic, model, test_sampler, testdata_loader, basecfg.basic.epoch, filename)
                   
        if dist.is_initialized() and basecfg.basic.local_rank != 0:
            dist.destroy_process_group()
        
        # write test results
        if cfg.local_rank == 0: 
            content = load_results(filename)
            # apply per-class
            per_class = datacfg.train_dataset.name =='dota' or datacfg.train_dataset.name =='stad'
            # frame level eval
            print(f'post_process: True')
            print_results(cfg, *evaluation(**content,post_process=True,per_class = per_class))
            print(f'post_process: False')
            print_results(cfg, *evaluation(**content,post_process=False,per_class = per_class))
            txt_folder = os.path.join(cfg.output, 'evaluation')
            os.makedirs(txt_folder,exist_ok=True)
            txt_path = os.path.join(txt_folder, 'eval.txt')
            write_results(txt_path, epoch , *evaluation( **content,per_class = per_class))
            
            # instance level eval
            if 'obj_targets' in content and sum([len(x) for x in content['obj_targets']]):
                write_results(txt_path,epoch,*evaluation_on_obj(content['obj_outputs'],content['obj_targets'],content['video_name']),eval_type='instacne')
            
            # frame level eval
            if 'fra_outputs' in content and content['fra_outputs'][0][0] != -100:
                write_results(txt_path, epoch , *evaluation(outputs = content['fra_outputs'], targets = content['targets']) , eval_type ='prompt frame')

            # ==========================================================
            # --- 多分类 (Subclass) 评估与打印写入 (图文双存版) ---
            # ==========================================================
            apply_sub_class = cfg.get('apply_sub_class', False)
            if apply_sub_class and 'sub_targets' in content:      
                # 展平并过滤 -100 (有效帧提取)
                flat_sub_targets = np.concatenate(content['sub_targets'], axis=0)
                flat_sub_outputs = np.concatenate(content['sub_outputs'], axis=0)
                valid_idx = flat_sub_targets != -100
                flat_sub_targets = flat_sub_targets[valid_idx]
                flat_sub_outputs = flat_sub_outputs[valid_idx]

                if len(flat_sub_targets) > 0:
                    # 获取基本指标 (宏平均) 和混淆矩阵
                    sub_metrics = calculate_metrics_multiclass(flat_sub_outputs, flat_sub_targets, num_classes=13)
                    preds_classes = np.argmax(flat_sub_outputs, axis=1)     
                    class_names = SUB_ANOMALIES
                    # 生成简写名
                    short_names = [name.replace("Collision:", "C:") for name in class_names]
                    
                    # 生成基础分类报告
                    sub_report_str = classification_report(
                        flat_sub_targets, preds_classes, target_names=class_names, labels=range(13), digits=4, zero_division=0
                    )

                    # 调用封装的 mAP 函数
                    ap_per_class, mAP, ap_report_str = calculate_per_class_map(
                        flat_sub_outputs=flat_sub_outputs,
                        flat_sub_targets=flat_sub_targets,
                        sub_cls_num=13, 
                        class_names=class_names
                    )

                    # --- 【新增：保存混淆矩阵图片到本地】 ---
                    # 重新生成带热力图效果的 plot (确保你已经更新了上文修改的 plot_figures_multiclass)
                    sub_plots = plot_figures_multiclass(
                        sub_metrics['confusion_matrix'], 
                        sub_metrics['pr_curve'],  
                        sub_metrics['roc_curve'],
                        class_names=class_names
                    )
                    cm_fig = sub_plots['confusion_matrix']
                    # 保存图片到 evaluation 文件夹下
                    cm_image_path = os.path.join(txt_folder, f'subcls_cmatrix_e{epoch}.png')
                    cm_fig.savefig(cm_image_path, bbox_inches='tight', dpi=300)
                    print(f"\n[INFO] Confusion matrix image saved to: {cm_image_path}")

                    # --- 打印到控制台 ---
                    print("\n***************** SUBCLASS TEST RESULTS *****************")
                    print("[Subclass Correctness]")
                    print("          Accuracy = %.5f" % sub_metrics['accuracy'])
                    print("       Macro f-AUC = %.5f" % sub_metrics['auroc'])
                    print("               mAP = %.5f" % mAP) 
                    print("           F1-Mean = %.5f" % sub_metrics['f1'])
                    print("\n[Per-Class Report]")
                    print(sub_report_str)
                    print(ap_report_str) 
                    print("\n*********************************************************\n")

                    # --- 追加写入到 eval.txt ---
                    with open(txt_path, 'a') as file:
                        try:
                            now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        except AttributeError:
                            now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            
                        file.write(f"\n######################### SUBCLASS TEST #########################\n")
                        file.write(f'{now_str}\n')
                        file.write(f"---------------- Test Eval on Epoch = {epoch} ----------------\n")
                        file.write("[Subclass Correctness]\n")
                        file.write("          Accuracy = %.5f\n" % sub_metrics['accuracy'])
                        file.write("       Macro f-AUC = %.5f\n" % sub_metrics['auroc'])
                        file.write("               mAP = %.5f\n" % mAP) 
                        file.write("           F1-Mean = %.5f\n" % sub_metrics['f1'])
                        
                        file.write("\n[Per-Class Report]\n")
                        file.write(sub_report_str + "\n")
                        file.write(ap_report_str + "\n") 
                        
                        # --- 终极不粘连版：混淆矩阵文本写入 ---
                        file.write("\n***************** Confusion Matrix *****************\n")
                        cm = sub_metrics['confusion_matrix']
                        
                        # 动态计算所需的列宽（最长缩写名字长度 + 1 或 2 个空格的 padding）
                        max_name_len = max([len(name) for name in short_names])
                        first_col_w = max_name_len + 2  # 第一列行表头宽一点
                        data_col_w = max_name_len + 1   # 数据列宽 (比如 14)
                        
                        # 写入列表头: T \ P      Normal   C:car2car   C:car2bike ...
                        header_cols = [name.rjust(data_col_w) for name in short_names]
                        header_str = "T \\ P".ljust(first_col_w) + "".join(header_cols)
                        file.write(header_str + "\n")
                        
                        # 写入数据行
                        for i, row in enumerate(cm):
                            row_name = short_names[i].ljust(first_col_w)
                            # 用对应的列宽格式化数字
                            row_data = "".join([f"{int(val):{data_col_w}d}" for val in row])
                            file.write(row_name + row_data + "\n")
                        file.write("\n")