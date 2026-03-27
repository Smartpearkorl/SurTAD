import numpy as np
import torch

from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist
from tqdm import tqdm
import datetime
import os
import yaml
import json
import copy
from runner.src.tools import CEloss , sub_cls_CEloss, NLLloss
from runner.src.dataset import gt_cls_target, SUB_ANOMALIES
from runner.src.utils import TensorboardLogger,debug_weights,debug_guess,get_result_filename,load_results,gather_predictions_nontensor
from runner.src.metrics import evaluation, write_results , evaluation_on_obj ,calculate_per_class_map, classification_report, plot_figures_multiclass, \
                               calculate_metrics, plot_figures, calculate_metrics_multiclass,plot_figures_multiclass
from runner.test import pama_test
from models.componets import HungarianMatcher

def save_checkpoint(cfg, e , model, optimizer, lr_scheduler, index_video, index_frame ):
    dir_chk = os.path.join(cfg.output, 'checkpoints')
    os.makedirs(dir_chk, exist_ok=True)
    path = os.path.join(dir_chk, 'model-{:02d}.pt'.format(e+1))
    torch.save({
        'epoch': e,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'lr_scheduler_state_dict': lr_scheduler.state_dict(),
        'index_video': index_video,
        'index_frame': index_frame,
    }, path)

def check_unused_para(model):
    for name, param in model.named_parameters():
        if param.grad is None:
             print(name)

class FrameState():
    def __init__(self) -> None:
        self.t = 0
        self.begin_t = 0
        self.T = 0
        pass

# debug:查看每部分时间
from time import perf_counter
flag_debug_t = False
debug_t = 0
def updata_debug_t():
        global debug_t
        debug_t = perf_counter()
        
def print_t(process='unknown process'):
    global flag_debug_t
    if flag_debug_t:
        print(f"{process} takes {(perf_counter() - debug_t):.4f}",force=True)
        updata_debug_t()     

def pama_train(cfg, model, train_sampler, traindata_loader, scaler, optimizer,lr_scheduler, 
               test_sampler=None, testdata_loader=None , index_video = 0 , index_frame= 0, ):

    target_metrics = ['auroc', 'accuracy', 'recall', 'precision', 'f1', 'ap']
    is_dist = dist.is_initialized()
    # log writer
    if cfg.local_rank == 0:
        # Tensorboard
        writer = TensorboardLogger(cfg.output + '/tensorboard/train_{}'.format(
            datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")))
        # add custom title to distinguish
        writer.add_scalar(cfg.output, 0, 0)

    # matcher
    apply_ins_loss = cfg.get('apply_ins_loss',False)
    apply_bottleneck_loss = cfg.get('apply_bottleneck_loss',False)
    if apply_ins_loss or apply_bottleneck_loss:
        matcher = HungarianMatcher()

    # NF for vst
    fb = cfg.get('NF',0)

    # apply sub class
    apply_sub_class = cfg.get('apply_sub_class',False)
    if apply_sub_class:
        sub_cls_celoss = sub_cls_CEloss()
        sub_cls_num = cfg.get('sub_class_num',0)
        if sub_cls_num == 0:
            raise ValueError("sub_class_num must be greater than 0 when apply_sub_class is True")

    celoss = CEloss(cfg)
    begin_epoch , total_epoch =  cfg.epoch , cfg.total_epoch 
    model.train(True)
    for e in range(begin_epoch, total_epoch):
        # DDP sampler
        if is_dist:
            train_sampler.set_epoch(e)
        
        # tqdm
        if cfg.local_rank == 0:
            pbar = tqdm(total=len(traindata_loader),desc='Epoch: %d / %d' % (e + 1, total_epoch))
        
        epoch_preds , epoch_labels = [] , []
                
        if apply_sub_class:
            epoch_subcls_preds_probs = [] # 收集 [N, 13] 的 softmax 概率
            epoch_subcls_labels = []      # 收集 [N] 的 0-12 标签

        print_t(process="-----")
        # run in single video
        for j, (video_data, data_info, yolo_boxes, frames_boxes, video_name) in enumerate(traindata_loader):
            print_t(process="Get batch")
            # prepare data for model and loss func
            video_data = video_data.to(cfg.device, non_blocking=True) # [B,T,C,H,W]
            data_info = data_info.to(cfg.device, non_blocking=True)
            # yolo_boxes : list B x list T x nparray(N_obj, 4)
            yolo_boxes = np.array(yolo_boxes,dtype=object)
            # matcher between yolo and gt
            if apply_ins_loss or apply_bottleneck_loss:
                match_index = [matcher(yolo,gt) for yolo , gt in zip(yolo_boxes,frames_boxes)]

            # record whole video data
            B,T = video_data.shape[:2]
            t_shape = (B,T-fb)
            targets = torch.full(t_shape, -100).to(video_data.device)
            outputs = torch.full(t_shape, -100, dtype=float).to(video_data.device)        
            video_len_orig, toa_batch, tea_batch = data_info[:, 0] , data_info[:, 2] , data_info[:, 3]

            # sub_cls_target
            if apply_sub_class:
                subcls_target = data_info[:,7]

            # loop in video frames
            rnn_state , frame_state = None , FrameState()
            for i in range(fb  , T):
                # preparation
                frame_state.t = i-fb
                frame_state.begin_t = 0
                frame_state.T = T-1-fb
                target = gt_cls_target(i-1, toa_batch, tea_batch).long()          
                batch_image_data , batch_boxes = video_data[:,i-fb:i] , yolo_boxes[:,i-1]

                with torch.cuda.amp.autocast(enabled=cfg.fp16):
                    optimizer.zero_grad()
                    ret = model(batch_image_data, batch_boxes, rnn_state, frame_state)
                    output_dict , rnn_state, outputs_ins_anormal= ret['output'] , ret['rnn_state'] , ret['ins_anomaly']
                    output, sub_output = output_dict['frame_out'], output_dict['sub_class_out']
                    
                    flt = i >= video_len_orig 
                    target = torch.where(flt, torch.full(target.shape, -100).to(video_data.device), target)
                    output = torch.where(flt.unsqueeze(dim=1).expand(-1,2), torch.full(output.shape, -100, dtype=output.dtype).to(video_data.device), output)
                    loss_frame = celoss(output,target)
                    loss = loss_frame
                    if apply_sub_class:
                        now_subcls = torch.where(target > 0, subcls_target, torch.zeros_like(subcls_target)).long()
                        # 注意：如果 flt 逻辑存在，这里需要应用，将超出有效长度的部分设为 -100
                        now_subcls[flt] = -100 
                        loss_subcls = sub_cls_celoss(sub_output, now_subcls)
                        loss = loss + loss_subcls

                    # instance anomaly loss
                    if apply_ins_loss or apply_bottleneck_loss:
                        if fb: 
                            now_match_index = [ x[i-1] for x in match_index ]
                        # dinov2  
                        else:
                            now_match_index = [ x[i] for x in match_index ]
                        '''
                        根据每帧yolo预测的object构造实例损失,例如:
                        outputs_ins_anormal: list[batch: list[object_nums,2]]
                        '''
                        # 构造gt objects
                        batch_shapes = [single.shape[0] if single.shape[0] else 1 for single in batch_boxes ]
                        gt_zeros = [torch.zeros(x).type(torch.int64).to(video_data.device) for x in batch_shapes] 
                        # gt_zeros = [torch.zeros(single.shape[0]).type(torch.int64).to(video_data.device) for single in outputs_ins_anormal]                        
                        gt_objects = copy.deepcopy(gt_zeros)
                        for ind,single in enumerate(gt_objects):
                            single[now_match_index[ind][0]]=1
                        gt_objects = torch.cat(gt_objects, dim=0)
                        if apply_ins_loss:
                            # batch single frame objects:  [ [box_num1,2], [box_num2,2], [box_num3,2]]  
                            src_objects = torch.cat(outputs_ins_anormal, dim=0)
                            loss_ins_anomaly = celoss(src_objects, gt_objects)
                            loss = loss + cfg.ins_loss_weight * loss_ins_anomaly
                        if apply_bottleneck_loss:
                            bottle_objects = torch.cat(ret['bottleneck_weight'], dim=0).squeeze(dim=-1)
                            loss_bottleneck = torch.nn.functional.binary_cross_entropy_with_logits(bottle_objects, gt_objects.to(dtype=bottle_objects.dtype))
                            loss = loss + cfg.bottle_loss_weight * loss_bottleneck
         
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                # record data (loss) per iteration 
                loss_dict = {}
                loss_dict['loss_frame'] = loss_frame
                if apply_ins_loss:
                    loss_dict['loss_instance'] = loss_ins_anomaly
                if apply_bottleneck_loss:
                    loss_dict['loss_bottleneck'] = loss_bottleneck
                if apply_sub_class:
                    loss_dict['loss_subclass'] = loss_subcls

                if cfg.local_rank == 0:
                    debug_weights(cfg.model_type, writer, model, loss_dict , index_frame, **cfg.train_debug)

                index_frame+=1

                # record whole video target and output
                targets[:, i-fb] = target.clone()
                out = output.softmax(dim=-1).max(1)[1] # select output(normal or anomaly)
                out[target == -100] = -100
                outputs[:, i-fb] = out    
                print_t(process=f"loop step{i}")
      
                # --- 新增 ---
                if apply_sub_class:
                    # 获取子类的概率分布 [Batch, 13]
                    # sub_output 形状应为 [Batch, 13]
                    sub_out_probs = sub_output.softmax(dim=-1).detach()
                    
                    # 为了只记录有效的帧，屏蔽掉 target == -100 的地方
                    valid_mask = (target != -100)
                    
                    if valid_mask.any():
                        # 只保存有效帧的数据，放到列表中
                        epoch_subcls_preds_probs.append(sub_out_probs[valid_mask])
                        epoch_subcls_labels.append(now_subcls[valid_mask])
                
                # NCCL barrier
                if is_dist:
                    dist.barrier()

            # update for scheduler
            lr_scheduler.step()
            
            # record batch data 
            epoch_preds.append(outputs.view(-1))
            epoch_labels.append(targets.view(-1))
            
            # record data(lr,val) per epoch 
            if cfg.local_rank == 0: 
                writer.add_scalar('lr',optimizer.param_groups[-1]['lr'],index_video)    
                debug_guess(writer, outputs, targets, index_video)
                pbar.set_description('Epoch: %d / %d, Loss: %.4f' % (e + 1, total_epoch, loss))
                pbar.update(1)
                
            index_video+=1

        # save checkpoint
        if cfg.local_rank == 0 and (e+1) % cfg.snapshot_interval == 0:
            save_checkpoint(cfg, e , model, optimizer, lr_scheduler, index_video, index_frame )

        # batch level logger
        if is_dist :
            epoch_preds = gather_predictions_nontensor(epoch_preds, world_size=dist.get_world_size())
            epoch_labels = gather_predictions_nontensor(epoch_labels, world_size=dist.get_world_size())

            # --- 新增：使用已有函数收集变长子类特征 ---
            if apply_sub_class and len(epoch_subcls_labels) > 0:
                epoch_subcls_preds_probs = gather_predictions_nontensor(epoch_subcls_preds_probs, world_size=dist.get_world_size())
                epoch_subcls_labels = gather_predictions_nontensor(epoch_subcls_labels, world_size=dist.get_world_size())

        if cfg.local_rank == 0:
            # === 原有二分类逻辑 ===
            all_preds = torch.cat(epoch_preds, dim=0).cpu()
            all_labels = torch.cat(epoch_labels, dim=0).cpu()
            
            # 过滤掉二分类中的 -100 (防止报错)
            valid_idx = all_labels != -100
            all_preds = all_preds[valid_idx]
            all_labels = all_labels[valid_idx]

            metrics_out = calculate_metrics(all_preds, all_labels)
            metrics_plots = plot_figures(metrics_out['confusion_matrix'], metrics_out['pr_curve'],  metrics_out['roc_curve'] )
            
            log_metrics = {x:metrics_out[x] for x in target_metrics}
            writer.update(head='train_metrics', step=e, **log_metrics)
            [writer.add_figure(f"train_plots/train_{k}", fig, global_step=e) for k, fig in metrics_plots.items()]
            log_stats = {**{f'train_{k}': log_metrics[k] for k in target_metrics}}
            
            # === 新增多分类 (Subclass) 逻辑 ===
            if apply_sub_class and len(epoch_subcls_labels) > 0:
                # 拼接刚才 Gather 回来的所有数据
                all_sub_probs = torch.cat(epoch_subcls_preds_probs, dim=0).cpu().numpy()
                all_sub_labels = torch.cat(epoch_subcls_labels, dim=0).cpu().numpy()
                
                # 前面帧循环已经做过 valid_mask，这里数据全是干净的
                if len(all_sub_labels) > 0:
                    sub_metrics_out = calculate_metrics_multiclass(all_sub_probs, all_sub_labels, num_classes=sub_cls_num)
                    
                    # 定义你的 13 个类别名称
                    class_names = [
                        "Normal", "Collision:car2car", "Collision:car2bike", "Collision:car2person",
                        "Collision:car2large", "Collision:large2large", "Collision:large2vru",
                        "Collision:bike2bike", "Collision:bike2person", "Collision:obstacle",
                        "Rollover", "Collision:others", "Unknown"
                    ]
                    
                    sub_plots = plot_figures_multiclass(
                        sub_metrics_out['confusion_matrix'], 
                        sub_metrics_out['pr_curve'],  
                        sub_metrics_out['roc_curve'],
                        class_names=class_names
                    )
                    
                    sub_log_metrics = {f"subcls_{x}": sub_metrics_out[x] for x in target_metrics}
                    writer.update(head='train_subcls_metrics', step=e, **sub_log_metrics)
                    [writer.add_figure(f"train_subcls_plots/{k}", fig, global_step=e) for k, fig in sub_plots.items()]
                    
                    log_stats.update({f'train_{k}': v for k, v in sub_log_metrics.items()})

            # === 写入 log.txt ===
            with open(os.path.join(cfg.output, "log.txt"), mode="a", encoding="utf-8") as f:
                # 写入 Epoch 标题作为分隔
                f.write(f"========== Epoch: {e + 1} ==========\n")
                
                # 将字典转为由格式化字符串组成的列表
                formatted_items = []
                for k, v in log_stats.items():
                    # 对浮点数保留4位小数，其他类型直接转为字符串
                    if isinstance(v, (float, int)):
                        item_str = f"{k}: {v:.4f}"
                    else:
                        item_str = f"{k}: {v}"
                    formatted_items.append(item_str)
                
                # 设置每行 5 列，固定每列的宽度（例如 32 个字符）来保证严格对齐
                num_cols = 2
                col_width = 32
                
                for i in range(0, len(formatted_items), num_cols):
                    # 取出当前行的 5 个元素
                    row_items = formatted_items[i:i + num_cols]
                    # 左对齐并用管道符 "|" 分隔
                    row_str = " | ".join([item.ljust(col_width) for item in row_items])
                    f.write(row_str + "\n")
                
                # 加一个空行，让不同 Epoch 之间看起来更清晰
                f.write("\n")

        # test
        if cfg.test_inteval != -1 and (e + 1) % cfg.test_inteval == 0:
            filename = get_result_filename(cfg, e + 1)
            with torch.no_grad():
                pama_test(cfg, model, test_sampler, testdata_loader,  e + 1, filename)
                # updatate model stage
                model.train(True)
            if is_dist:
                dist.barrier()
            # record eval data
            if cfg.local_rank == 0: 
                txt_folder = os.path.join(cfg.output, 'evaluation')
                os.makedirs(txt_folder,exist_ok=True)
                txt_path = os.path.join(txt_folder, 'eval.txt')
                content = load_results(filename)
                per_class = cfg.get("dataset_type",'dota') == 'dota'
                write_results(txt_path, e + 1, *evaluation(**content,post_process=False,per_class=per_class))
                # instance level eval
                if 'obj_targets' in content and sum([len(x) for x in content['obj_targets']]):
                    write_results(txt_path,e + 1,*evaluation_on_obj(content['obj_outputs'],content['obj_targets'],content['video_name']),eval_type='instacne')
                # frame level eval
                if 'fra_outputs' in content and content['fra_outputs'][0][0] != -100:
                    write_results(txt_path, e + 1 , *evaluation(outputs = content['fra_outputs'], targets = content['targets']) , eval_type ='prompt frame')

        # test
        if cfg.test_inteval != -1 and (e + 1) % cfg.test_inteval == 0:
            filename = get_result_filename(cfg, e + 1)
            with torch.no_grad():
                pama_test(cfg, model, test_sampler, testdata_loader,  e + 1, filename)
                # updatate model stage
                model.train(True)
            if is_dist:
                dist.barrier()
            
            # ============== record eval data ==============
            # write test results
            if cfg.local_rank == 0: 
                content = load_results(filename)
                # apply per-class
                per_class = cfg.train_dataset.name =='dota' or cfg.train_dataset.name =='stad'
                txt_folder = os.path.join(cfg.output, 'evaluation')
                os.makedirs(txt_folder,exist_ok=True)
                txt_path = os.path.join(txt_folder, 'eval.txt')
                write_results(txt_path, e , *evaluation( **content,per_class = per_class))
                
                # instance level eval
                if 'obj_targets' in content and sum([len(x) for x in content['obj_targets']]):
                    write_results(txt_path,e,*evaluation_on_obj(content['obj_outputs'],content['obj_targets'],content['video_name']),eval_type='instacne')
                
                # frame level eval
                if 'fra_outputs' in content and content['fra_outputs'][0][0] != -100:
                    write_results(txt_path, e , *evaluation(outputs = content['fra_outputs'], targets = content['targets']) , eval_type ='prompt frame')

                # ==========================================================
                # --- 多分类 (Subclass) 评估与打印写入 (图文双存版) ---
                # ==========================================================
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
                        cm_image_path = os.path.join(txt_folder, f'subcls_cmatrix_e{e}.png')
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
                            file.write(f"---------------- Test Eval on Epoch = {e} ----------------\n")
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


            # if cfg.local_rank == 0: 
            #     txt_folder = os.path.join(cfg.output, 'evaluation')
            #     os.makedirs(txt_folder,exist_ok=True)
            #     txt_path = os.path.join(txt_folder, 'eval.txt')
            #     content = load_results(filename)
            #     per_class = cfg.get("dataset_type",'dota') == 'dota'
                
            #     # 1. 二分类评估写入
            #     write_results(txt_path, e + 1, *evaluation(**content,post_process=False,per_class=per_class))
                
            #     # instance level eval
            #     if 'obj_targets' in content and sum([len(x) for x in content['obj_targets']]):
            #         write_results(txt_path,e + 1,*evaluation_on_obj(content['obj_outputs'],content['obj_targets'],content['video_name']),eval_type='instance')
                
            #     # prompt frame level eval
            #     if 'fra_outputs' in content and content['fra_outputs'][0][0] != -100:
            #         write_results(txt_path, e + 1 , *evaluation(outputs = content['fra_outputs'], targets = content['targets']) , eval_type ='prompt frame')

            #     # 2. --- 新增：多分类评估写入 ---
            #     if apply_sub_class and 'sub_targets' in content:
            #         # 将 list of lists 展平
            #         # sub_targets: [N_videos] -> [Total_Frames]
            #         flat_sub_targets = np.concatenate(content['sub_targets'], axis=0)
            #         # sub_outputs: [N_videos, seq_len, 13] -> [Total_Frames, 13]
            #         flat_sub_outputs = np.concatenate(content['sub_outputs'], axis=0)

            #         # 仅保留有效的帧 (非 -100 的帧，如果前面清理得很干净可能不需要，但加了更安全)
            #         valid_idx = flat_sub_targets != -100
            #         flat_sub_targets = flat_sub_targets[valid_idx]
            #         flat_sub_outputs = flat_sub_outputs[valid_idx]

            #         if len(flat_sub_targets) > 0:
            #             # 借用你之前导入/写入的 multi_class 函数计算结果
            #             sub_metrics = calculate_metrics_multiclass(flat_sub_outputs, flat_sub_targets, num_classes=sub_cls_num)
                        
            #             # 向 eval.txt 追加多分类评估结果，保持格式整洁
            #             with open(txt_path, 'a') as file:
            #                 now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            #                 file.write(f"\n######################### SUBCLASS TEST EPOCH #########################\n")
            #                 file.write(f'{now}\n')
            #                 file.write(f"---------------- Test Eval on Epoch = {e + 1} ----------------\n")
            #                 file.write("[Subclass Correctness]\n")
            #                 file.write("          Accuracy = %.5f\n" % sub_metrics['accuracy'])
            #                 file.write("      Macro f-AUC = %.5f\n" % sub_metrics['auroc'])
            #                 file.write("         Macro AP = %.5f\n" % sub_metrics['ap'])
            #                 file.write("          F1-Mean = %.5f\n" % sub_metrics['f1'])
            #                 file.write("           Recall = %.5f\n" % sub_metrics['recall'])
            #                 file.write("        Precision = %.5f\n" % sub_metrics['precision'])
                            
            #                 file.write("\n***************** Confusion Matrix *****************\n")
            #                 # 格式化混淆矩阵输出
            #                 cm = sub_metrics['confusion_matrix']
            #                 for row in cm:
            #                     row_str = " ".join([f"{val:5d}" for val in row])
            #                     file.write(row_str + "\n")
            #                 file.write("\n")


        #  close an epoch bar 
        if cfg.local_rank == 0:
            pbar.close()
        
        # NCCL barrier
        if is_dist:
            dist.barrier()

                   
                    



    
