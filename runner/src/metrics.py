
import numpy as np

from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, \
                    average_precision_score, classification_report  
from datetime import datetime
import scipy.signal as signal
from scipy.ndimage import uniform_filter1d

from runner.src import utils
from runner.src.dataset import anomalies, stad_anomalies

anomaly_class = stad_anomalies #
'''
前向中值和后向中值滤波（不用）
def forward_medfilt(signal, kernel_size):
    n = len(signal)
    filtered_signal = np.zeros_like(signal)
    
    for i in range(n):
        if i < kernel_size - 1:
            # 在前面没有足够的点时，考虑所有可用点
            filtered_signal[i] = np.median(signal[:i+1])
        else:
            # 只考虑前面的 kernel_size 个点
            filtered_signal[i] = np.median(signal[i-kernel_size+1:i+1])
    
    return filtered_signal

def backward_medfilt(signal, kernel_size):
    n = len(signal)
    filtered_signal = np.zeros_like(signal)
    
    for i in range(n):
        # 确保只使用当前点和其后的点来计算中值
        end_idx = min(i + kernel_size, n)
        filtered_signal[i] = np.median(signal[i:end_idx])
    
    return filtered_signal
'''

def normalize_video(video_scores):
    video_scores_max = video_scores.max()
    video_scores_min = video_scores.min()
    if video_scores_max - video_scores_min > 0:
        video_scores = (video_scores - video_scores_min) / (video_scores_max - video_scores_min)
    return video_scores

def evaluation_on_obj(outputs,targets,video_name,eva_type='all'):
    if eva_type == 'all':
        # concate video output: [B,[T,[N]]] -> [B,sum N]
        outputs = [np.concatenate(x,axis=0) for x in outputs]
        targets = [np.concatenate(x,axis=0) for x in targets]
        # concate whole video output
        outputs = np.concatenate(outputs,axis=0) 
        targets = np.concatenate(targets,axis=0)
        return evaluation(outputs,targets)

    elif eva_type == 'scene':
        # concate video output: [B,[T,[N]]] -> [B,sum N]
        outputs = [np.concatenate(x,axis=0) for x in outputs]
        targets = [np.concatenate(x,axis=0) for x in targets]
        return evaluation_per_scene(outputs,targets,video_name)

    else:
        raise Exception(f'not supported instance eval type {eva_type}')

def evaluation_per_scene(outputs, targets, video_name, auc_type = 'frame', metric_type = 'AUC', post_process=False , kernel_size = 31, **kwargs):
    scene_score ={}
    assert len(outputs) == len(targets)," length of outputs is not equal to targets' "
    for i in range(len(outputs)):
        if auc_type == 'frame' : 
            preds = np.array(outputs[i])
            gts = np.array(targets[i])
            if post_process:
                if len(preds)>kernel_size:
                    preds = signal.medfilt(preds, kernel_size=kernel_size)
                else:
                    ks = len(preds) if len(preds)%2!=0 else len(preds)-1
                    preds = signal.medfilt(preds, kernel_size=ks)
                preds = normalize_video(np.array(preds))
        elif auc_type == 'instance':
            preds = np.concatenate(outputs[i])
            gts = np.concatenate(targets[i])
            # 计算AUC必须要有两个类
            if not np.any(gts):
                continue

        # 计算AUC必须要有两个类
        if len(np.unique(gts)) == 1:
            red_data = gts[0]==0
            gts = np.append(gts,red_data)
            preds = np.append(preds,red_data)
        
        # metric_type
        if metric_type == 'AUC':
            scene_score[video_name[i]]= roc_auc_score(gts, preds)
        elif metric_type == 'Accuracy':
            scene_score[video_name[i]]= accuracy_score(gts, preds > 0.5)

    scene_score = dict(sorted(scene_score.items(), key=lambda item: item[1]))
    return scene_score

def safe_auc(gts, preds):
    """
    不修改原始 gts / preds，仅在单类别时构造临时数据以保证 AUC 可计算
    """
    gts_tmp = gts
    preds_tmp = preds

    # 计算AUC必须要有两个类
    if len(np.unique(gts_tmp)) == 1:
        red_data = 1 if gts_tmp[0] == 0 else 0
        gts_tmp = np.append(gts_tmp, red_data)
        preds_tmp = np.append(preds_tmp, red_data)

    return roc_auc_score(gts_tmp, preds_tmp)

def subclass_ranking_per_scene(pkl_path, metric_type='AP', class_names=None, sub_cls_num=13):
    """
    计算每个视频在各个子类上的指标，并返回按指标降序排列的视频名称列表。
    支持 metric_type='AP' (推荐) 或 'Accuracy'。
    
    返回格式:
    {
        "Normal": ["video_8", "video_2", ...],
        "Collision:car2car": ["video_5", "video_1", ...],
        ...
    }
    """
    content = utils.load_results(pkl_path)
    
    # 纠正赋值：读取真实的预测概率分布和目标标签
    sub_outputs = content['sub_outputs']  # 列表，每个元素形状 [T, 13]
    sub_targets = content['sub_targets']  # 列表，每个元素形状 [T]
    video_name = content['video_name']

    if class_names is None:
        from runner.src.dataset import stad_anomalies
        class_names = ["Normal"] + stad_anomalies
        
    # 内部字典，用于暂存 (视频名, 得分) 的元组，例如 {"Normal": [("vid1", 0.99), ("vid2", 0.85)]}
    class_scores_temp = {name: [] for name in class_names}
    
    assert len(sub_outputs) == len(sub_targets), "length of sub_outputs is not equal to sub_targets"
    
    for i in range(len(sub_outputs)):
        preds = np.array(sub_outputs[i])  # 形状: [T, 13]
        gts = np.array(sub_targets[i])    # 形状: [T]
        
        # 1. 过滤掉无效帧
        valid_idx = gts != -100
        preds = preds[valid_idx]
        gts = gts[valid_idx]
        
        if len(gts) == 0:
            continue
            
        # 2. 找出当前视频实际包含了哪些类别
        present_classes = np.unique(gts)
        
        if metric_type == 'AP':
            # 将该视频的真实标签转为 One-hot，以匹配 [T, 13] 的预测概率
            targets_onehot = label_binarize(gts, classes=range(sub_cls_num))
            
            # 计算每个类的 AP
            ap_per_class = average_precision_score(targets_onehot, preds, average=None)
            
            # 仅记录该视频实际存在的类别的 AP 得分
            for c in present_classes:
                score = ap_per_class[c]
                # 排除因为全 0 或全 1 导致的 NaN
                if not np.isnan(score):
                    class_scores_temp[class_names[c]].append((video_name[i], score))
                    
        elif metric_type == 'Accuracy':
            # 如果需要算 Accuracy，内部自动取 argmax
            preds_classes = np.argmax(preds, axis=1)
            for c in present_classes:
                class_mask = (gts == c)
                if np.sum(class_mask) > 0:
                    score = accuracy_score(gts[class_mask], preds_classes[class_mask])
                    class_scores_temp[class_names[c]].append((video_name[i], score))
                    
        else:
            raise ValueError(f"Unsupported metric_type: {metric_type}")

    # 3. 对每个类别的列表按得分降序排序，并提取视频名字
    ranked_videos_per_class = {}
    for name in class_names[1:]:
        # 降序排序: 得分越高的视频排在越前面
        sorted_items = sorted(class_scores_temp[name], key=lambda x: x[1], reverse=True)
        # 只保留视频名称
        ranked_videos_per_class[name] = [item[0] for item in sorted_items]
        
    return ranked_videos_per_class

'''
对每个scene计算AUC.
return : Dict{'scene_name' : auc_score,}
'''
def AUC_on_scene(pkl_path , auc_type = 'frame' , post_process=False):
    content = utils.load_results(pkl_path)
    if auc_type == 'frame':
        return evaluation_per_scene(**content , post_process = post_process)
    elif auc_type == 'instance':
        if 'obj_targets' in content and sum([len(x) for x in content['obj_targets']]):
            return evaluation_per_scene(content['obj_outputs'],content['obj_targets'],content['video_name'],auc_type)
        else:
            raise Exception(f'set AUC type is instance , but no obj_targets in {pkl_path}')
    raise Exception(f'unsupported AUC type {auc_type}, must be frame or instance')

'''
对每个scene计算 Accuracy.
return : Dict{'scene_name' : acc_score,}
'''
def Accuracy_on_scene(pkl_path , acc_type = 'frame' , post_process=False):
    content = utils.load_results(pkl_path)
    if acc_type == 'frame':
        return evaluation_per_scene(**content , post_process = post_process, metric_type='Accuracy')
    elif acc_type == 'instance':
        if 'obj_targets' in content and sum([len(x) for x in content['obj_targets']]):
            return evaluation_per_scene(content['obj_outputs'],content['obj_targets'],content['video_name'],acc_type, metric_type='Accuracy')
        else:
            raise Exception(f'set Accuracy type is instance , but no obj_targets in {pkl_path}')
    raise Exception(f'unsupported Accuracy type {acc_type}, must be frame or instance')


def evaluation(outputs, targets, info=None, post_process=False, kernel_size = 31, popr_type = 'mid' , per_class=False, **kwargs):
    if post_process:
        post_outputs = []
        for preds in outputs:
            ks = len(preds) if len(preds)%2!=0 else len(preds)-1 # make sure odd
            now_ks = min(kernel_size,ks)
            if popr_type == 'mid':
                preds = signal.medfilt(preds, kernel_size=now_ks)     
            elif popr_type == 'mean':
                preds = uniform_filter1d(preds, size=kernel_size)
            scores_each_video = normalize_video(np.array(preds))
            post_outputs.append(scores_each_video)
        outputs = post_outputs

    preds = np.array(utils.flat_list(outputs))
    gts = np.array(utils.flat_list(targets))
    F1_mean, _, F1_one = f1_mean(gts, preds)
    return (
        roc_auc_score(gts, preds),
        average_precision_score(gts, preds),
        F1_one,
        F1_mean,
        accuracy_score(gts, preds > 0.5),
        classification_report(
            gts, preds > 0.5, target_names=['normal', 'anomaly']),
        get_eval_per_class(outputs, targets, info, utils.split_by_class) if per_class else None,
        None
        # get_eval_per_class(outputs, targets, info, utils.split_by_class_ego) if per_class else None,
    )


def print_results(cfg, AUC_frame, PRAUC_frame, f1score, f1_mean, accuracy,
                  report, eval_per_class, eval_per_class_ego , per_class=True):
    
    print("[Correctness] f-AUC = %.5f" % (AUC_frame))
    print("             PR-AUC = %.5f" % (PRAUC_frame))
    print("           F1-Score = %.5f" % (f1score))
    print("           F1-Mean  = %.5f" % (f1_mean))
    print("           Accuracy = %.5f" % (accuracy))
    #  print("      accident pred = %.5f" % (acc_pred))
    #  print("        normal pred = %.5f" % (nor_pred))
    print()
    print(report)
    print()

    if per_class:
        print("***************** per class eval *****************\n")
        if eval_per_class is not None:
            print("---------------------- ALL ----------------------\n")         
            print('F1-mean per class\n')
            for key, values in eval_per_class.items():
                print('{:03f} F1-mean class {}\n'.format(list(values)[2], anomaly_class[int(key)-1]))
            print("\n")
            
            print('f-AUC per class\n')
            for key, values in eval_per_class.items():
                print('{:03f} f-AUC class {}\n'.format(list(values)[0], anomaly_class[int(key)-1]))

        if eval_per_class_ego is not None:
            clss = sorted(set([cls for cls, _ in eval_per_class_ego.keys()]))
            print("\n")
            # non_involve ego
            print("---------------- Non-involve ego ----------------\n")    
            fauc_ego = [list(eval_per_class_ego[(cls, 0.)])[0] for cls in clss]
            for key, values in zip(clss,fauc_ego):
                print('{:03f} f-AUC class {}\n'.format(
                    values, anomaly_class[int(key)-1]))
            print("\n")
            
            print("------------------ Involve ego ------------------\n")       
            fauc_ego = [list(eval_per_class_ego[(cls, 1.)])[0] for cls in clss]
            for key, values in zip(clss,fauc_ego):
                print('{:03f} f-AUC class {}\n'.format(
                    values, anomaly_class[int(key)-1]))  
                     
def write_results(file_name, epoch, AUC_frame, PRAUC_frame, f1score, f1_mean, accuracy,
                  report, eval_per_class, eval_per_class_ego, eval_type = 'frame', prefix = None, per_class=True):
    # 获取当前日期和时间
    now = datetime.now()
    # 格式化日期和时间
    formatted_time = now.strftime("%Y-%m-%d %H:%M:%S")

    with open(file_name, 'a') as file:
        file.write("\n######################### EPOCH #########################\n")
        # 写入评估指标
        file.write(f'{formatted_time}')
        if prefix:
            file.write(f'\n{prefix}')  
        file.write(f"\n----------------{eval_type} eval on epoch = {epoch}----------------\n")
        file.write("[Correctness] f-AUC = %.5f\n" % (AUC_frame))
        file.write("             PR-AUC = %.5f\n" % (PRAUC_frame))
        file.write("           F1-Score = %.5f\n" % (f1score))
        file.write("           F1-Mean  = %.5f\n" % (f1_mean))
        file.write("           Accuracy = %.5f\n" % (accuracy))
        # file.write("      accident pred = %.5f\n" % (acc_pred))
        # file.write("        normal pred = %.5f\n" % (nor_pred))
        file.write("\n")
        file.write(report + "\n")
        file.write("\n")
        
        if per_class:
            file.write("***************** per class eval *****************\n")
            if eval_per_class is not None:
                file.write("---------------------- ALL ----------------------\n")         
                file.write('F1-mean per class\n')
                for key, values in eval_per_class.items():
                    file.write('{:03f} F1-mean class {}\n'.format(list(values)[2], anomaly_class[int(key)-1]))
                file.write("\n")
                
                file.write('f-AUC per class\n')
                for key, values in eval_per_class.items():
                    file.write('{:03f} f-AUC class {}\n'.format(list(values)[0], anomaly_class[int(key)-1]))

            if eval_per_class_ego is not None:
                clss = sorted(set([cls for cls, _ in eval_per_class_ego.keys()]))
                file.write("\n")
                # non_involve ego
                file.write("---------------- Non-involve ego ----------------\n")    
                fauc_ego = [list(eval_per_class_ego[(cls, 0.)])[0] for cls in clss]
                for key, values in zip(clss,fauc_ego):
                    file.write('{:03f} f-AUC class {}\n'.format(
                        values, anomaly_class[int(key)-1]))
                file.write("\n")
                
                file.write("------------------ Involve ego ------------------\n")       
                fauc_ego = [list(eval_per_class_ego[(cls, 1.)])[0] for cls in clss]
                for key, values in zip(clss,fauc_ego):
                    file.write('{:03f} f-AUC class {}\n'.format(
                        values, anomaly_class[int(key)-1]))        
                
def f1_mean(gts, preds):
    F1_one = f1_score(gts, preds > 0.5)
    F1_zero = f1_score(
        (gts.astype('bool') == False).astype('long'),
        preds <= 0.5)
    F1_mean = 2 * (F1_one * F1_zero) / (F1_one + F1_zero)
    return F1_mean, F1_zero, F1_one

def get_eval_per_class(outputs, targets, info, split_fun):
    # retrocompat
    if info is None:
        return None
    # outputs/targets split per class
    ot = split_fun(outputs, targets, info)
    data = {}
    for cls, vals in ot.items():
        if vals['outputs'].shape[0]>0:
            # 🚨 关键修改：这里必须用中括号 [ ] 变成 List，不能用大括号 { }
            data[cls] = [ 
                roc_auc_score(vals['targets'], vals['outputs']),              # index 0: AUC
                average_precision_score(vals['targets'], vals['outputs']),    # index 1: AP
                f1_mean(vals['targets'], vals['outputs'])[0],                 # index 2: F1-mean
                accuracy_score(vals['targets'], vals['outputs'] > 0.5),       # index 3: Accuracy
            ] 
        else:
            data[cls] = [0,0,0,0]  # not existed the class
    return data


import torch
import pandas as pd
from typing import Sequence, Optional
import matplotlib.pyplot as plt

import seaborn as sns
import numpy as np
import cv2
from sklearn.metrics import (
    auc,
    accuracy_score,
    recall_score,
    precision_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    roc_curve,
    matthews_corrcoef
)


THRESHOLDS = np.arange(0.00, 1.001, 0.01).tolist()

def calculate_metrics(preds, labels):
    # Convert to numpy
    preds = np.array(preds)
    labels = np.array(labels)

    # Binary prediction at threshold 0.5
    binary_preds = (preds >= 0.5).astype(int)

    # Metrics at threshold 0.5
    metr_acc   = accuracy_score(labels, binary_preds)
    recall_val = recall_score(labels, binary_preds)
    precision_val = precision_score(labels, binary_preds)
    f1_val     = f1_score(labels, binary_preds)
    confmat    = confusion_matrix(labels, binary_preds).tolist()

    # Threshold-independent metrics
    auroc = roc_auc_score(labels, preds)
    ap    = average_precision_score(labels, preds)
    pr_curve_vals = precision_recall_curve(labels, preds)
    roc_curve_vals = roc_curve(labels, preds)

    return {
        "accuracy": metr_acc,
        "recall": recall_val,
        "precision": precision_val,
        "f1": f1_val,
        "auroc": auroc,
        "ap": ap,
        "confusion_matrix": confmat,
        "pr_curve": pr_curve_vals,
        "roc_curve": roc_curve_vals,
    }


def fig_to_cv2_image(fig):
    # Save the Matplotlib figure to a buffer in memory
    fig.canvas.draw()
    img = np.array(fig.canvas.renderer.buffer_rgba())
    img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    [ax.cla() for ax in fig.get_axes()]
    plt.close(fig)
    return img


def threshold_curve_plots(
        x_values: Sequence[float],
        y_values: Sequence[float],
        thresholds: Sequence[float],
        x_label: str,
        y_label: str,
        plot_name: str,
        score: bool = False,
        to_img: bool = False,
        curve_type_correction: Optional[str] = None
):
    """
    Plot a threshold curve and optionally calculate the area under the curve (AUC).

    Parameters:
    - x_values: Sequence of x-axis values (e.g., recall or false positive rate)
    - y_values: Sequence of y-axis values (e.g., precision or true positive rate)
    - thresholds: Sequence of threshold values corresponding to x and y values
    - x_label: Label for the x-axis
    - y_label: Label for the y-axis
    - plot_name: Title of the plot
    - score: Boolean to indicate if AUC should be calculated and displayed (default is False)
    - to_img: Boolean to indicate if convert the resulting plot to image (default is False)
    - curve_type_correction: str to indicate correction type, 'roc' or 'pr'.
    """
    assert len(x_values) == len(y_values) == len(thresholds)

    # Ensure x_values are sorted and unique
    sorted_indices = np.argsort(x_values)
    x_values = np.array(x_values)[sorted_indices]
    y_values = np.array(y_values)[sorted_indices]
    thresholds = np.array(thresholds)[sorted_indices]

    # Remove duplicates
    unique_x_values, unique_indices = np.unique(x_values, return_index=True)
    y_values = y_values[unique_indices]
    thresholds = thresholds[unique_indices]
    x_values = unique_x_values

    if curve_type_correction is not None:
        if curve_type_correction == "roc":
            y0, y1 = 0., 1.
        elif curve_type_correction == "pr":
            y0, y1 = 1., 0.
        else:
            raise ValueError
        x_values = np.insert(x_values, 0, 0.)
        x_values = np.append(x_values, 1.)
        y_values = np.insert(y_values, 0, y0)
        y_values = np.append(y_values, y1)

    plt.rcParams['figure.dpi'] = 100
    plt.rcParams['savefig.dpi'] = 100
    px = 1 / plt.rcParams['figure.dpi']
    figsize = (640 * px, 480 * px)

    fig, ax = plt.subplots(figsize=figsize)
    sns.set_style("whitegrid")

    ax.plot(x_values, y_values, marker='o', linestyle='-', color='b')

    step = 5
    y = (10, -20)
    for i in range(0, len(thresholds), step):
        ax.plot(x_values[i], y_values[i], marker='o', linestyle='-', color='g')
        ax.annotate(f"{thresholds[i]:.2f}", (x_values[i], y_values[i]), textcoords="offset points", xytext=(0, y[i % 2]),
                    ha='center')

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(plot_name)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax.grid(True, which='both', linestyle='--', color='gray', alpha=0.5)
    ax.xaxis.set_major_locator(plt.MultipleLocator(0.1))
    ax.yaxis.set_major_locator(plt.MultipleLocator(0.1))
    ax.xaxis.set_minor_locator(plt.MultipleLocator(0.1 / 2))
    ax.yaxis.set_minor_locator(plt.MultipleLocator(0.1 / 2))
    ax.grid(which='major', linestyle='--', color='gray', alpha=0.5)
    ax.grid(which='minor', linestyle=':', color='gray', alpha=0.3)

    if score and x_values.shape[0] > 1:
        auc_score = auc(x_values, y_values)
        ax.text(0.40, 0.20, f'AUC: {auc_score:.2f}', transform=ax.transAxes, fontsize=12, ha='left')

    if to_img:
        fig = fig_to_cv2_image(fig)

    return fig

def plot_figures(confmat, pr_curve, roc_curve):
    # plot figures
    df_cm = pd.DataFrame(confmat)
    fig1 = sns.heatmap(df_cm, annot=True, cmap='Spectral').get_figure()
    plt.close(fig1)
    pr_precision, pr_recall, pr_thresholds = pr_curve
    fig2 = threshold_curve_plots(
        x_values=pr_recall.tolist(), y_values=pr_precision.tolist(), thresholds=pr_thresholds.tolist() + [1.],
        x_label="Recall", y_label="Precision", plot_name="PR curve",
        score=True, to_img=False, curve_type_correction="pr"
    )
    plt.close(fig2)
    roc_fpr, roc_tpr, roc_thresholds = roc_curve
    fig3 = threshold_curve_plots(
        x_values=roc_fpr.tolist(), y_values=roc_tpr.tolist(), thresholds=roc_thresholds.tolist(),
        x_label="FP rate", y_label="TP rate", plot_name="ROC curve",
        score=True, to_img=False, curve_type_correction="roc"
    )
    plt.close(fig3)
    plots = {"confusion_matrix": fig1, "PR_curve": fig2, "ROC_curve": fig3}
    return plots

from sklearn.preprocessing import label_binarize
# --- 新增：多分类指标计算 ---
def calculate_metrics_multiclass(preds_probs, labels, num_classes=13):
    """
    preds_probs: [N, C] numpy array of probabilities (after softmax)
    labels: [N] numpy array of integer labels (0 to C-1)
    """
    preds_probs = np.array(preds_probs)
    labels = np.array(labels)
    
    # 获取硬标签 (argmax)
    preds_classes = np.argmax(preds_probs, axis=1)

    # 1. 基础指标 (使用 macro 平均)
    metr_acc = accuracy_score(labels, preds_classes)
    recall_val = recall_score(labels, preds_classes, average='macro', zero_division=0)
    precision_val = precision_score(labels, preds_classes, average='macro', zero_division=0)
    f1_val = f1_score(labels, preds_classes, average='macro', zero_division=0)
    confmat = confusion_matrix(labels, preds_classes, labels=range(num_classes)).tolist()

    # 2. 阈值无关指标 (One-vs-Rest 策略)
    # 将 labels 转换为 One-hot 形式以便计算每类的 ROC/PR
    labels_onehot = label_binarize(labels, classes=range(num_classes))
    
    # 计算 Macro AUROC (处理某些类在 batch 中不存在的情况)
    try:
        auroc = roc_auc_score(labels_onehot, preds_probs, average='macro', multi_class='ovr')
    except ValueError:
        auroc = 0.0 # 数据集中可能缺失某些类
        
    # 计算 Macro AP (Average Precision)
    from sklearn.metrics import average_precision_score
    try:
        ap = average_precision_score(labels_onehot, preds_probs, average='macro')
    except ValueError:
        ap = 0.0

    # 3. 为绘图准备 Macro 平均的 ROC 和 PR 曲线数据
    # 计算微平均 (Micro-average) 或 宏平均 (Macro-average) 的曲线
    # 为了简化画图，这里计算 Micro-average 曲线传给原有的画图函数
    fpr_micro, tpr_micro, thresholds_roc = roc_curve(labels_onehot.ravel(), preds_probs.ravel())
    precision_micro, recall_micro, thresholds_pr = precision_recall_curve(labels_onehot.ravel(), preds_probs.ravel())

    return {
        "accuracy": metr_acc,
        "recall": recall_val,
        "precision": precision_val,
        "f1": f1_val,
        "auroc": auroc,
        "ap": ap,
        "confusion_matrix": confmat,
        "pr_curve": (precision_micro, recall_micro, thresholds_pr), # 兼容你的返回值拆包
        "roc_curve": (fpr_micro, tpr_micro, thresholds_roc),
    }

# # --- 新增/修改：多分类绘图函数 ---
# def plot_figures_multiclass(confmat, pr_curve, roc_curve, class_names=None):
#     if class_names is None:
#         class_names = [f"C{i}" for i in range(len(confmat))]

#     # 1. 绘制混淆矩阵 (放大画布以适应 13x13)
#     plt.figure(figsize=(12, 10))
#     df_cm = pd.DataFrame(confmat, index=class_names, columns=class_names)
#     # 使用 fmt='d' 显示整数，防止科学计数法
#     fig1 = sns.heatmap(df_cm, annot=True, cmap='Blues', fmt='g').get_figure()
#     plt.close(fig1)

#     # 2. PR 曲线 (利用你原有的函数，只是数据换成了 Micro-average)
#     pr_precision, pr_recall, pr_thresholds = pr_curve
#     fig2 = threshold_curve_plots(
#         x_values=pr_recall.tolist(), y_values=pr_precision.tolist(), 
#         thresholds=pr_thresholds.tolist() + [1.], # 保持你的逻辑
#         x_label="Recall (Micro)", y_label="Precision (Micro)", plot_name="PR curve (Micro-Avg)",
#         score=True, to_img=False, curve_type_correction="pr"
#     )
#     plt.close(fig2)

#     # 3. ROC 曲线
#     roc_fpr, roc_tpr, roc_thresholds = roc_curve
#     fig3 = threshold_curve_plots(
#         x_values=roc_fpr.tolist(), y_values=roc_tpr.tolist(), thresholds=roc_thresholds.tolist(),
#         x_label="FP rate (Micro)", y_label="TP rate (Micro)", plot_name="ROC curve (Micro-Avg)",
#         score=True, to_img=False, curve_type_correction="roc"
#     )
#     plt.close(fig3)

#     plots = {"confusion_matrix": fig1, "PR_curve": fig2, "ROC_curve": fig3}
#     return plots

# --- 修改：多分类绘图函数，以生成类似用户图片的混淆矩阵热力图 ---
def plot_figures_multiclass(confmat, pr_curve, roc_curve, class_names=None):
    if class_names is None:
        class_names = [f"C{i}" for i in range(len(confmat))]

    # 1. 绘制混淆矩阵 (调整画布大小和参数以匹配图片)
    # 增加 figsize 以适应长标签和清晰的数字
    fig1 = plt.figure(figsize=(16, 14)) 
    df_cm = pd.DataFrame(confmat, index=class_names, columns=class_names)
    
    # 创建热力图
    # cmap='Blues': 使用蓝色调颜色映射
    # annot=True: 显示数值
    # fmt='g': 格式化数值，防止科学计数法
    # square=True: 使方格成为正方形
    # cbar=True: 显示颜色条
    # annot_kws: 可以调整数字大小，这里使用默认
    ax = sns.heatmap(df_cm, annot=True, cmap='Blues', fmt='g', square=True, cbar=True)
    
    # 旋转列标签90度，并右对齐，确保不被切掉
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90, horizontalalignment='right')
    # 确保行标签水平显示
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    
    # 调整布局，为旋转标签和颜色条留出空间
    # 使用 tight_layout 自动调整，确保标签清晰
    plt.tight_layout()
    # 也可以根据需要手动微调边距，例如:
    # fig1.subplots_adjust(bottom=0.25, left=0.2) 

    # 获取 figure 对象
    fig1 = ax.get_figure()
    plt.close(fig1)

    # 2. PR 曲线 (利用你原有的函数，只是数据换成了 Micro-average)
    pr_precision, pr_recall, pr_thresholds = pr_curve
    fig2 = threshold_curve_plots(
        x_values=pr_recall.tolist(), y_values=pr_precision.tolist(), 
        thresholds=pr_thresholds.tolist() + [1.], # 保持你的逻辑
        x_label="Recall (Micro)", y_label="Precision (Micro)", plot_name="PR curve (Micro-Avg)",
        score=True, to_img=False, curve_type_correction="pr"
    )
    plt.close(fig2)

    # 3. ROC 曲线
    roc_fpr, roc_tpr, roc_thresholds = roc_curve
    fig3 = threshold_curve_plots(
        x_values=roc_fpr.tolist(), y_values=roc_tpr.tolist(), thresholds=roc_thresholds.tolist(),
        x_label="FP rate (Micro)", y_label="TP rate (Micro)", plot_name="ROC curve (Micro-Avg)",
        score=True, to_img=False, curve_type_correction="roc"
    )
    plt.close(fig3)

    plots = {"confusion_matrix": fig1, "PR_curve": fig2, "ROC_curve": fig3}
    return plots

def calculate_per_class_map(flat_sub_outputs, flat_sub_targets, sub_cls_num=13, class_names=None):
    """
    计算多分类任务中每个子类的 AP (Average Precision) 以及总体的 mAP。

    Args:
        flat_sub_outputs (np.ndarray): 形状为 [N, sub_cls_num] 的预测概率分布。
        flat_sub_targets (np.ndarray): 形状为 [N] 的真实类别标签 (0 到 sub_cls_num-1)。
        sub_cls_num (int): 子类别的总数，默认为 13。
        class_names (list of str, optional): 类别名称列表，用于格式化输出。长度必须等于 sub_cls_num。

    Returns:
        ap_per_class (np.ndarray): 包含每个子类 AP 的数组，形状为 [sub_cls_num]。
        mAP (float): 忽略 NaN 后的平均 AP。
        report_str (str): 格式化好的可以直接 print 或写入文件的报表字符串。
    """
    # 确保输入是 Numpy 数组
    flat_sub_outputs = np.array(flat_sub_outputs)
    flat_sub_targets = np.array(flat_sub_targets)

    # 如果没有传入类别名，则自动生成 Class_0, Class_1 ...
    if class_names is None:
        class_names = [f"Class_{i}" for i in range(sub_cls_num)]
    assert len(class_names) == sub_cls_num, "class_names 的长度必须与 sub_cls_num 一致"

    # 将一维标签转化为 One-hot 编码，形状变为 [N, sub_cls_num]
    targets_onehot = label_binarize(flat_sub_targets, classes=range(sub_cls_num))

    # 计算每个子类的 AP
    # average=None 返回一个数组，包含每个类的 AP
    ap_per_class = average_precision_score(targets_onehot, flat_sub_outputs, average=None)

    # 计算 mAP (忽略因为真实标签中没有该类导致的 NaN)
    mAP = np.nanmean(ap_per_class)

    # 构造格式化的报表字符串
    report_str_list = ["\n[Per-Class Average Precision (AP)]"]
    for i, name in enumerate(class_names):
        ap_val = ap_per_class[i]
        # 如果验证集中完全没有某个类别的真实样本，AP 会是 NaN，显示为 N/A
        ap_val_str = f"{ap_val:.4f}" if not np.isnan(ap_val) else "N/A   "
        report_str_list.append(f"{name.rjust(25)} : {ap_val_str}")
    
    report_str_list.append("-" * 35)
    report_str_list.append(f"{'mAP (Mean AP)'.rjust(25)} : {mAP:.4f}")
    
    report_str = "\n".join(report_str_list)

    return ap_per_class, mAP, report_str