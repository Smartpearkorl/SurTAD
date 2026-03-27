import itertools
import math
import numpy as np
import os
import pickle
import torch

from collections import OrderedDict

import functools
import operator

import torch.nn as nn


def load_checkpoint(cfg):
    dir_chk = os.path.join(cfg.output, 'checkpoints')
    # build file path
    if cfg.epoch != -1:
        path = os.path.join(dir_chk, 'model-{:02d}.pt'.format(cfg.epoch))
    else:
        raise FileNotFoundError()
    if not os.path.exists(path):
        raise FileNotFoundError()
    print('load file {}'.format(path))
    checkpoint = torch.load(path, map_location=cfg.device)
    # 判断是不是nn.DataParallel包装的模型
    if 'module' in list(checkpoint['model_state_dict'].keys())[0]:
        weights = checkpoint['model_state_dict']
        checkpoint['model_state_dict'] = OrderedDict([(k[7:], v) for k, v in weights.items()])
            
    return checkpoint

'''
fpn_load(cfg, model)
'''
def fpn_load(cfg , model):
    checkpoint = torch.load(cfg.directly_load,map_location=torch.device('cpu'))
    # 判断是不是nn.DataParallel包装的模型
    if 'module' in list(checkpoint['model_state_dict'].keys())[0]:
        weights = checkpoint['model_state_dict']
        checkpoint['model_state_dict'] = OrderedDict([(k[7:], v) for k, v in weights.items()])

    ckp = checkpoint['model_state_dict']
    ckp_update = {}
    for key in ckp.keys():
       tmp = key
       key = key.replace('fpn_model.backbone','vst_model')
       key = key.replace('fpn_model.fpn','fpn')

       key = key.replace('fpn_model.bn.weight','reducer.reducer.2.weight')
       key = key.replace('fpn_model.bn.bias','reducer.reducer.2.bias')
       key = key.replace('fpn_model.linear.weight','reducer.reducer.3.weight')
       key = key.replace('fpn_model.linear.bias','reducer.reducer.3.bias')
       key = key.replace('prompt_linear.weight','reducer.cls.0.weight')
       key = key.replace('prompt_linear.bias','reducer.cls.0.bias')


       key = key.replace('rnn_bn.weight','anomaly_regressor.ln.weight')
       key = key.replace('rnn_bn.bias','anomaly_regressor.ln.bias')

       key = key.replace('rnn.','anomaly_regressor.rnn.')
       key = key.replace('rnn_linear.weight','anomaly_regressor.cls.0.weight')
       key = key.replace('rnn_linear.bias','anomaly_regressor.cls.0.bias')
       key = key.replace('cls_linear.weight','anomaly_regressor.cls.3.weight')
       key = key.replace('cls_linear.bias','anomaly_regressor.cls.3.bias')
    
       ckp_update[key] = ckp[tmp]

    model.load_state_dict(ckp_update)
    return ckp_update

def load_state_dict(model, state_dict, prefix='', ignore_missing="relative_position_index"):
    missing_keys = []
    unexpected_keys = []
    error_msgs = []
    metadata = getattr(state_dict, '_metadata', None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata

    def load(module, prefix=''):
        local_metadata = {} if metadata is None else metadata.get(
            prefix[:-1], {})
        module._load_from_state_dict(
            state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + '.')

    load(model, prefix=prefix)

    warn_missing_keys = []
    ignore_missing_keys = []
    for key in missing_keys:
        keep_flag = True
        for ignore_key in ignore_missing.split('|'):
            if ignore_key in key:
                keep_flag = False
                break
        if keep_flag:
            warn_missing_keys.append(key)
        else:
            ignore_missing_keys.append(key)

    missing_keys = warn_missing_keys

    if len(missing_keys) > 0:
        print(f"Weights of {model.__class__.__name__} not initialized from pretrained model: \n" + \
                '\n'.join(missing_keys))
    if len(unexpected_keys) > 0:
        print(f"Weights from pretrained model not used in { model.__class__.__name__}: \n" + \
              '\n'.join(unexpected_keys))
    if len(ignore_missing_keys) > 0:
        print(f"Ignored weights of { model.__class__.__name__} not initialized from pretrained model: \n" + \
              '\n'.join(ignore_missing_keys))
    if len(error_msgs) > 0:
        print('\n'.join(error_msgs))

    return missing_keys


def direcrtly_load_checkpoint(cfg , model , whole_load = False):
    checkpoint = torch.load(cfg.directly_load,map_location=next(model.parameters()).device)
    # 判断是不是nn.DataParallel包装的模型 ,得到vst backbone 的参数
    checkpoint = checkpoint['model_state_dict']
    if 'module' in list(checkpoint.keys())[0]:
        checkpoint = dict([(k[7:], v) for k, v in checkpoint.items()])

    if not whole_load:
        ckp_update, fpn_update = {} , {}
        for key in checkpoint.keys():
            if 'vst_model' in key:
                key_up = key.replace('vst_model.','')
                ckp_update[key_up] = checkpoint[key]
            if 'fpn' in key:
                key_up = key.replace('fpn.','')
                fpn_update[key_up] = checkpoint[key]                             
        missing_keys, unexpected_keys = model.vst_model.load_state_dict(ckp_update,strict=True)
        assert not unexpected_keys and not missing_keys , f'{unexpected_keys=}\n{missing_keys=}'
        missing_keys, unexpected_keys = model.fpn.load_state_dict(fpn_update,strict=True)
        assert not unexpected_keys and not missing_keys , f'{unexpected_keys=}\n{missing_keys=}'
    else:
        missing_keys, unexpected_keys = model.load_state_dict(checkpoint,strict=True)
        assert not unexpected_keys and not missing_keys , f'{unexpected_keys=}\n{missing_keys=}'

# "/data/qh/DoTA/output/Prompt/standart_fpn_lr=0.002/checkpoints/model-200.pt"
def resume_from_checkpoint(cfg , model , optimizer , lr_scheduler):
    # directly load : this means Two models are not completely equal , customize load model_state_dict
    checkpoint = {}
    if cfg.directly_load and cfg.epoch == -1:
        whole_load = cfg.get('whole_load',True)
        print(f'directly load from {cfg.directly_load}')
        direcrtly_load_checkpoint(cfg,model,whole_load)
        print(f'epoch set to 0')
        cfg.epoch  = 0  # launch first epoch    
    else:
        try:
            checkpoint = load_checkpoint(cfg)   
            cfg.epoch = checkpoint['epoch'] + 1 # move to next epoch
            missing_keys, unexpected_keys = model.load_state_dict(checkpoint['model_state_dict'],strict=False)
            if missing_keys:
                print(f'missing_keys:\n{unexpected_keys}')
            if unexpected_keys:
                print(f'unexpected_keys:\n{unexpected_keys}')
            if optimizer:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if lr_scheduler:
                lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])       
        except FileNotFoundError:
            print('no checkpoint found')

            if cfg.epoch != -1:
                raise Exception('epoch={cfg.epoch} but not find checkpoint')
            else:
                print(f'epoch set to 0')
                cfg.epoch  = 0  # launch first epoch 
    return checkpoint

def prod(iterable):
    return functools.reduce(operator.mul, iterable, 1)


def get_visual_directory(cfg, epoch):
    output_dir = os.path.join(cfg.output, 'vis-{:02d}'.format(epoch))
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def get_result_filename(cfg, epoch):
    output_dir = os.path.join(cfg.output, 'eval')
    os.makedirs(output_dir, exist_ok=True)
    return os.path.join(output_dir, 'results-{:02d}.pkl'.format(epoch))


def load_results(filename):
    print('load file {}'.format(filename))
    with open(filename, 'rb') as f:
        content = pickle.load(f)
    return content

def get_last_epoch(filenames):
    epochs = [int(name.split('-')[1].split('.')[0]) for name in filenames]
    return filenames[np.array(epochs).argsort()[-1]]


def w_count(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def flat_list(list_):
    if isinstance(list_, (np.ndarray, np.generic)):
        # to be retrocompatible
        return list_
    return list(itertools.chain(*list_))


def filter_by_class(outputs, info, cls):
    return [out for out, inf in zip(outputs, info.tolist()) if inf[0] == cls]


def filter_by_class_ego(outputs, info, cls, ego):
    return [out for out, inf in zip(outputs, info.tolist())
            if all([inf[0] == cls, inf[1] == ego])]

def split_by_class(outputs, targets, info):
    clss = np.unique(info[:, 0]).tolist()
    return {
        cls: {
            'outputs': np.array(
                flat_list(filter_by_class(outputs, info, cls))),
            'targets': np.array(
                flat_list(filter_by_class(targets, info, cls))),
        } for cls in clss
    }


def merge_oo_class(splitted):
    def _cat(v1, v2):
        return np.concatenate([v1, v2], axis=0)

    if (8.0, 0.0) in splitted and (9.0, 0.0) in splitted:
        oo0_outputs = _cat(splitted[(8.0, 0.0)]['outputs'],
                        splitted[(9.0, 0.0)]['outputs'])
        oo1_outputs = _cat(splitted[(8.0, 1.0)]['outputs'],
                        splitted[(9.0, 1.0)]['outputs'])
        oo0_targets = _cat(splitted[(8.0, 0.0)]['targets'],
                        splitted[(9.0, 0.0)]['targets'])
        oo1_targets = _cat(splitted[(8.0, 1.0)]['targets'],
                        splitted[(9.0, 1.0)]['targets'])
        
        lables = list(splitted.keys())
        ids = [x[0] for x in lables]
        next_id = max(ids) + 1
        # OO (11)  = OO-r (8) + OO-l (9)
        splitted[(next_id, 0.0)] = {
            'outputs': oo0_outputs,
            'targets': oo0_targets,
        }
        splitted[(next_id, 1.0)] = {
            'outputs': oo1_outputs,
            'targets': oo1_targets,
        }

    return splitted


def split_by_class_ego(outputs, targets, info):
    clss = np.unique(info[:, 0]).tolist()
    egos = np.unique(info[:, 1]).tolist()
    pairs = [itertools.chain(*li.tolist()) for li in np.meshgrid(clss, egos)]
    pairs = zip(*pairs)
    return merge_oo_class({
        (cls, ego): {
            'outputs': np.array(
                flat_list(filter_by_class_ego(outputs, info, cls, ego))),
            'targets': np.array(
                flat_list(filter_by_class_ego(targets, info, cls, ego))),
        } for cls, ego in pairs
    })


def get_abs_weights_grads(model):
    if isinstance(model , nn.Module):
        return torch.cat([
                p.grad.detach().view(-1) for p in model.parameters()
                if p.requires_grad and p.grad is not None
            ]).abs()
    elif isinstance(model, nn.Embedding):
        return model.weight.detach().view(-1)


def get_abs_weights(model):
    if isinstance(model , nn.Module):   
        return torch.cat([
            p.detach().view(-1) for p in model.parameters()
            if p.requires_grad and p.grad is not None
        ]).abs()
    elif isinstance(model, nn.Embedding):
        return model.weight.gard.detach().view(-1)

def log_vals(writer, model, global_key, key, fun, index_l):
    if w_count(model):
        vals = fun(model)
        writer.add_scalar(
            '{}_mean/{}'.format(global_key, key), vals.mean().item(), index_l)
        # writer.add_scalar(
        #     '{}_std/{}'.format(global_key, key), vals.std().item(), index_l)
        # writer.add_scalar(
        #     '{}_max/{}'.format(global_key, key), vals.max().item(), index_l)



def get_params(model, keys):
    return [param for name, param in model.named_parameters()
            if any([key in name for key in keys])]


def get_params_rest(model, keys):
    return [param for name, param in model.named_parameters()
            if all([key not in name for key in keys])]


def debug_weights(model_type, writer, model, loss_dict, index_l,
                  debug_train_weight, debug_train_grad, debug_loss,
                  debug_train_grad_level, debug_train_weight_level,
                  ):

    if debug_train_grad:
        scan_internal( model_type,
            writer, model, 'grads', get_abs_weights_grads,
            debug_train_grad_level, index_l)

    if debug_train_weight:
        scan_internal(model_type,
            writer, model, 'weights', get_abs_weights,
            debug_train_weight_level, index_l)

    if debug_loss:
        for key , value in loss_dict.items():
            writer.add_scalar(f'loss/{key}', value.item(), index_l)

def debug_guess(writer, outputs, targets, index):
    """Debug guess info."""
    f_tp = outputs == targets
    f_t_1 = targets == 1
    f_t_0 = targets == 0

    ok = len(outputs[f_tp])
    tpos = len(outputs[f_t_1])  # totally pos
    tneg = len(outputs[f_t_0])  # totally neg
    cpos = len(outputs[f_tp & f_t_1])  # true pos
    cneg = len(outputs[f_tp & f_t_0])  # true neg
    tot = prod(outputs.shape)

    g_all = ok / tot
    g_pos = (cpos / tpos) if tpos > 0 else 0
    g_neg = (cneg / tneg) if tneg > 0 else 0
    writer.add_scalar('guess/all', g_all, index)
    writer.add_scalar('guess/pos', g_pos, index)
    writer.add_scalar('guess/neg', g_neg, index)


def scan_internal(model_type, writer, model, global_key, fun, level, index_l):
  
    if isinstance(model, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
        model = model.module
    
    # base_plain
    if model_type == 'base_plain':
        log_vals(writer, model.anomaly_regressor.reducer, global_key, 'ano_regressor.reducer', fun, index_l)
        log_vals(writer, model.anomaly_regressor.cls, global_key, 'ano_regressor.cls', fun, index_l)

    elif model_type == 'base_rnn': 
        log_vals(writer, model.sam_model.model.image_encoder.blocks[0], global_key, 'vit.block[0]', fun, index_l)
        log_vals(writer, model.sam_model.model.image_encoder.blocks[11], global_key, 'vit.block[11]', fun, index_l)
        log_vals(writer, model.sam_model.model.image_encoder.neck, global_key, 'vit.neck', fun, index_l)
        log_vals(writer, model.reducer, global_key, 'reducer', fun, index_l)
        log_vals(writer, model.anomaly_regressor.rnn, global_key, 'ano_regressor.rnn', fun, index_l)
        log_vals(writer, model.anomaly_regressor.cls, global_key, 'ano_regressor.cls', fun, index_l)

    elif model_type=='prompt_rnn':
        # level  base -> whole model 

        # level 1 -> sam_model bottle_aug bottle_regressor instance_decoder reducer anomaly_regressor
        if level > 0:
            log_vals(writer, model.sam_model.model.mask_decoder, global_key, 'sam_model.mask_decoder', fun, index_l)
            if model.use_bottle_aug:
                log_vals(writer, model.bottle_aug, global_key, 'bottle_aug', fun, index_l)
                log_vals(writer, model.bottle_regressor, global_key, 'bottle_regressor', fun, index_l)
            log_vals(writer, model.instance_decoder, global_key, 'instance_decoder', fun, index_l)
            log_vals(writer, model.reducer, global_key, 'reducer', fun, index_l)
            log_vals(writer, model.anomaly_regressor, global_key, 'anomaly_regressor', fun, index_l)

        # level 1 -> sam_model bottle_aug bottle_regressor instance_decoder reducer anomaly_regressor
        if  level > 1:
            log_vals(writer, model.sam_model.model.prompt_encoder.point_embeddings[2], global_key, 'sam_model.pnt_embeds[2]', fun, index_l)
            log_vals(writer, model.sam_model.model.prompt_encoder.point_embeddings[3], global_key, 'sam_model.pnt_embeds[3]', fun, index_l)
            log_vals(writer, model.instance_decoder.layer[0], global_key, 'instance_layer[0]', fun, index_l)
            log_vals(writer, model.instance_decoder.layer[1], global_key, 'instance_layer[1]', fun, index_l)
            log_vals(writer, model.anomaly_regressor.rnn, global_key, 'anomaly_regressor.rnn', fun, index_l)
            log_vals(writer, model.anomaly_regressor.cls, global_key, 'anomaly_regressor.cls', fun, index_l)
    
    elif model_type == 'poma_base_fpn_rnn':
        log_vals(writer, model.vst_model.patch_embed, global_key, 'vst_model.patch_embed', fun, index_l)
        log_vals(writer, model.vst_model.layers[0], global_key, 'vst_model.layers[0]', fun, index_l)
        log_vals(writer, model.vst_model.layers[1], global_key, 'vst_model.layers[1]', fun, index_l)
        log_vals(writer, model.vst_model.layers[2], global_key, 'vst_model.layers[2]', fun, index_l)
        log_vals(writer, model.vst_model.layers[3], global_key, 'vst_model.layers[3]', fun, index_l)
        log_vals(writer, model.fpn, global_key, 'fpn', fun, index_l)
        log_vals(writer, model.reducer, global_key, 'reducer', fun, index_l)
        log_vals(writer, model.anomaly_regressor, global_key, 'anomaly_regressor', fun, index_l)
    
    elif model_type == 'poma_prompt_fpn_rnn':
        log_vals(writer, model.vst_model.patch_embed, global_key, 'vst_model.patch_embed', fun, index_l)
        log_vals(writer, model.vst_model.layers[0], global_key, 'vst_model.layers[0]', fun, index_l)
        log_vals(writer, model.vst_model.layers[1], global_key, 'vst_model.layers[1]', fun, index_l)
        log_vals(writer, model.vst_model.layers[2], global_key, 'vst_model.layers[2]', fun, index_l)
        log_vals(writer, model.vst_model.layers[3], global_key, 'vst_model.layers[3]', fun, index_l)
        log_vals(writer, model.fpn, global_key, 'fpn', fun, index_l)
        log_vals(writer, model.instance_encoder.prompt_decoder.transformer.layers[0], global_key, 'prompt_decoder.layers[0]', fun, index_l)
        log_vals(writer, model.instance_encoder.prompt_decoder.transformer.layers[1], global_key, 'prompt_decoder.layers[1]', fun, index_l)
        log_vals(writer, model.instance_decoder.layer[0], global_key, 'instance_decoder.layers[0]', fun, index_l)
        log_vals(writer, model.instance_decoder.layer[1], global_key, 'instance_decoder.layers[1]', fun, index_l)
        log_vals(writer, model.reducer, global_key, 'reducer', fun, index_l)
        log_vals(writer, model.anomaly_regressor, global_key, 'anomaly_regressor', fun, index_l)
        

'''
log writer
'''
from torch.utils.tensorboard import SummaryWriter
from collections import defaultdict, deque
import torch.distributed as dist
import time
import datetime
   
def gather_predictions_nontensor(info, world_size):
    # Step 1: Wrap the Python object in a list (if it's not already)
    if not isinstance(info, list):
        info = [info]

    # Step 2: Check if the list contains tensors
    if isinstance(info[0], torch.Tensor):
        # Move all tensors to CPU to make sure they can be gathered
        info = [tensor.cpu() for tensor in info]

    # Step 3: Gather the info object across all ranks using all_gather_object
    gathered_info = [None for _ in range(world_size)]
    dist.all_gather_object(gathered_info, info)

    # Step 4: Flatten the list of gathered objects
    all_info = []
    for rank_info in gathered_info:
        all_info.extend(rank_info)

    return all_info


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True

class TensorboardLogger(object):
    def __init__(self, log_dir):
        self.writer = SummaryWriter(log_dir=log_dir)
        self.step = 0

    def set_step(self, step=None):
        if step is not None:
            self.step = step
        else:
            self.step += 1

    def add_scalar(self, *arg, **kwags):
        return self.writer.add_scalar(*arg, **kwags)

    def add_figure(self, *arg, **kwags):
        return self.writer.add_figure(*arg, **kwags)

    def update(self, head='scalar', step=None, **kwargs):
        for k, v in kwargs.items():
            if v is None:
                continue
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.writer.add_scalar(head + "/" + k, v, self.step if step is None else step)

    def flush(self):
        self.writer.flush()


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if v is None:
                continue
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

