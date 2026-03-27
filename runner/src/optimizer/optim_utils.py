import torch
from torch import optim as optim
from torch.optim.lr_scheduler  import CosineAnnealingLR, CosineAnnealingWarmRestarts,StepLR, OneCycleLR,SequentialLR,LinearLR
import json


OPTIM_REG = {
    'sgd' : optim.SGD,
    'adamw' : optim.AdamW,
}

SCHED_REG = {
    'SequentialLR': SequentialLR,
    'CosineAnnealingLR': CosineAnnealingLR,
    'LinearLR':LinearLR,
    'CosineAnnealingWarmRestarts':CosineAnnealingWarmRestarts,
    'StepLR':StepLR,
    'OneCycleLR':OneCycleLR,
}

'''
vit-specific optimizer utils
'''
def get_num_layer_for_vit(var_name, num_max_layer):
    if var_name in ("cls_token", "mask_token", "pos_embed"):
        return 0
    elif var_name.startswith("patch_embed"):
        # module.vit_model.blocks.6.norm1.bias
        return 0
    elif var_name.startswith("rel_pos_bias"):
        return num_max_layer - 1
    elif var_name.startswith("blocks"):
        layer_id = int(var_name.split('.')[1])
        return layer_id + 1
    else:
        return num_max_layer - 1


class LayerDecayValueAssigner(object):
    def __init__(self, values: None | list[float]):
        self.values = values

    def get_scale(self, layer_id):
        return self.values[layer_id] if self.values is not None else None

    def get_layer_id(self, var_name):
        return get_num_layer_for_vit(var_name, len(self.values)) if self.values is not None else None


class UnifiedParamGrouper:
    """
    同时支持:
        1. lr_mult_dict: 按名称匹配 lr 系数
        2. layer decay: 按 ViT layer 进行 lr_scale
        3. weight decay: 是否跳过 bias / bn
    """
    def __init__(self, optimcfg, lr_mult_dict=None,
                 get_num_layer=None, get_layer_scale=None,
                 skip_list=(), default_mult=1.0):
        self.base_lr = optimcfg.lr
        self.weight_decay = optimcfg.weight_decay
        self.lr_mult_dict = lr_mult_dict or {}
        self.default_mult = default_mult

        # layer decay
        self.get_num_layer = get_num_layer
        self.get_layer_scale = get_layer_scale

        # skip weight decay list
        self.skip_list = skip_list

    def match_lr_mult(self, name):
        """旧功能：根据 lr_mult_dict 的 key 模糊匹配"""
        for key, mult in self.lr_mult_dict.items():
            if key in name:
                return mult
        return self.default_mult

    def get_wd_and_group_name(self, name, param):
        """是否需要 weight decay"""
        if len(param.shape) == 1 or name.endswith(".bias") or name in self.skip_list:
            return 0., "no_decay"
        else:
            return self.weight_decay, "decay"

    def get_layer_group_name(self, name, base_group):
        """加入 layer_id 形成类似 layer_3_decay"""
        if self.get_num_layer is None:
            return base_group, None

        layer_id = self.get_num_layer(name)
        return f"layer_{layer_id}_{base_group}", layer_id

    def __call__(self, model):
        group_dict = {}

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue

            # 1. 计算 weight decay 与基础分组
            wd, base_group = self.get_wd_and_group_name(name, param)

            # 2. lr: name-based mult + layer decay scale
            lr_mult = self.match_lr_mult(name)

            # 3. 加 layer id: 只对backbone ( vit / vst )有效
            if 'vit_model' in name : # or 'vst_model' in name
                splitname = name.split("vit_model.", 1)[1]
                layer_group, layer_id = self.get_layer_group_name(splitname, base_group)
                group_name = f'{layer_group}_{lr_mult}'
            else:
                group_name,  layer_id = f'base_{base_group}_{lr_mult}', None

            layer_scale = self.get_layer_scale(layer_id) if (self.get_layer_scale and layer_id is not None) else 1.0

            final_lr = self.base_lr * lr_mult * layer_scale
            # 4. 放入组
            if group_name not in group_dict:
                group_dict[group_name] = {
                    "params": [],
                    "lr": final_lr,
                    "weight_decay": wd,
                }
            group_dict[group_name]["params"].append(param)

        return list(group_dict.values())

def prepare_optim_sched(model, optimcfg, schedcfg ):
    
    layer_decay = optimcfg.get('layer_decay',1.0)
    default_mult = optimcfg.get('default_mult',1.0)
    lr_mult_dict = optimcfg.get('lr_mult_dict',None)
    skip_list = optimcfg.get('skip_list',())
    
    if layer_decay < 1.0:
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            num_layers = model.module.get_num_layers()
        else:
            num_layers = model.get_num_layers()

        assigner = LayerDecayValueAssigner(list(layer_decay ** (num_layers + 1 - i) for i in range(num_layers + 2)))
    else:
        assigner = LayerDecayValueAssigner(None)

    param_grouper = UnifiedParamGrouper(
        optimcfg=optimcfg,
        lr_mult_dict=lr_mult_dict,
        get_num_layer=assigner.get_layer_id,
        get_layer_scale=assigner.get_scale,
        skip_list=skip_list,
        default_mult=default_mult
    )

    params_to_optimize = param_grouper(model)

    # SGD
    if optimcfg.type == 'sgd':
        optimizer = OPTIM_REG[optimcfg.cls](params_to_optimize, lr=optimcfg.lr, momentum=optimcfg.momentum , weight_decay = optimcfg.weight_decay)
    # adamw
    elif optimcfg.type == 'adamw':
        optimizer = OPTIM_REG[optimcfg.cls](params_to_optimize, lr=optimcfg.lr, weight_decay=optimcfg.weight_decay)

    # LinearLR and CosineAnnealingLR
    warm_sched = SCHED_REG[schedcfg.warm.cls](optimizer, schedcfg.warm.ini.start_factor, schedcfg.warm.ini.end_factor, schedcfg.warm.ini.total_iters)
    main_sched = SCHED_REG[schedcfg.main.cls](optimizer, schedcfg.main.ini.T_max ,schedcfg.main.ini.eta_min)
    lr_scheduler = SCHED_REG[schedcfg.cls](optimizer, [warm_sched, main_sched], [schedcfg.warm.warm_iters])

    return optimizer, lr_scheduler


def freeze_vit_backbone(model_type,model):
    if 'sam' in model_type :
        vit_model = model.sam_model.model.image_encoder
        for name,para in vit_model.named_parameters():
            para.requires_grad = False
        print(f'sam image-encoder is frozen ')
    # elif 'dinov2' in model_type:
    #     vit_model = model.vit_model
    #     for name,para in vit_model.named_parameters():
    #         para.requires_grad = False
    #     print(f'dinov2 image-encoder is frozen ')
    else:
        print(f'image-encoder is unfrezon ')

# def get_params_group(model, optimcfg, lr_mult_dict=None, default_mult=1.0):
#     """
#     lr_mult_dict 用于根据参数名设置学习率倍率，比如：
#         lr_mult_dict = {
#             "backbone": 0.1,
#             "head": 0.5,
#             "classifier": 2.0,
#         }
#     default_mult 是没有匹配到时的默认倍率
#     """
#     if lr_mult_dict is None:
#         lr_mult_dict = {}

#     # 初始化参数组映射 {倍率: [param_list]}
#     group_dict = {}

#     for name, param in model.named_parameters():
#         if not param.requires_grad:
#             continue

#         # 找匹配的倍率
#         matched = False
#         for key, mult in lr_mult_dict.items():
#             if key in name:  # 模糊匹配参数名
#                 group_dict.setdefault(mult, []).append(param)
#                 matched = True
#                 break

#         if not matched:
#             # 放入默认倍率组
#             group_dict.setdefault(default_mult, []).append(param)

#     # 生成优化器参数格式
#     params_to_optimize = []
#     for mult, params in group_dict.items():
#         params_to_optimize.append({
#             "params": params,
#             "lr": optimcfg.lr * mult
#         })
#     return params_to_optimize

# def prepare_optim_sched(model , optimcfg,  schedcfg, lr_mult_dict = None ):
#     # parameter to optimize
#     # params_to_optimize = filter(lambda p: p.requires_grad, model.parameters())
#     params_to_optimize = get_params_group(model, optimcfg, lr_mult_dict)
  
#     # SGD
#     if optimcfg.type == 'sgd':
#         optimizer = OPTIM_REG[optimcfg.cls](params_to_optimize, lr=optimcfg.lr, momentum=optimcfg.momentum , weight_decay = optimcfg.weight_decay)
#     # adamw
#     elif optimcfg.type == 'adamw':
#         optimizer = OPTIM_REG[optimcfg.cls](params_to_optimize, lr=optimcfg.lr, weight_decay=optimcfg.weight_decay)

#     # LinearLR and CosineAnnealingLR
#     warm_sched = SCHED_REG[schedcfg.warm.cls](optimizer, schedcfg.warm.ini.start_factor, schedcfg.warm.ini.end_factor, schedcfg.warm.ini.total_iters)

#     main_sched = SCHED_REG[schedcfg.main.cls](optimizer, schedcfg.main.ini.T_max ,schedcfg.main.ini.eta_min)
#     lr_scheduler = SCHED_REG[schedcfg.cls](optimizer, [warm_sched, main_sched], [schedcfg.warm.warm_iters])
#     return optimizer , lr_scheduler


def get_parameter_groups(model, weight_decay=1e-5, skip_list=(), get_num_layer=None, get_layer_scale=None):
    parameter_group_names = {}
    parameter_group_vars = {}

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
            group_name = "no_decay"
            this_weight_decay = 0.
        else:
            group_name = "decay"
            this_weight_decay = weight_decay
        if get_num_layer is not None:
            layer_id = get_num_layer(name)
            group_name = "layer_%d_%s" % (layer_id, group_name)
        else:
            layer_id = None

        if group_name not in parameter_group_names:
            if get_layer_scale is not None:
                scale = get_layer_scale(layer_id)
            else:
                scale = 1.

            parameter_group_names[group_name] = {
                "weight_decay": this_weight_decay,
                "params": [],
                "lr_scale": scale
            }
            parameter_group_vars[group_name] = {
                "weight_decay": this_weight_decay,
                "params": [],
                "lr_scale": scale
            }

        parameter_group_vars[group_name]["params"].append(param)
        parameter_group_names[group_name]["params"].append(name)
    print("Param groups = %s" % json.dumps(parameter_group_names, indent=2))
    return list(parameter_group_vars.values())


def create_optimizer(args, model, get_num_layer=None, get_layer_scale=None, filter_bias_and_bn=True, skip_list=None):
    opt_lower = args.opt.lower()
    weight_decay = args.weight_decay
    if weight_decay and filter_bias_and_bn:
        skip = {}
        if skip_list is not None:
            skip = skip_list
        elif hasattr(model, 'no_weight_decay'):
            skip = model.no_weight_decay()
        parameters = get_parameter_groups(model, weight_decay, skip, get_num_layer, get_layer_scale)
        weight_decay = 0.
    else:
        parameters = model.parameters()


    opt_args = dict(lr=args.lr, weight_decay=weight_decay)
    if hasattr(args, 'opt_eps') and args.opt_eps is not None:
        opt_args['eps'] = args.opt_eps
    if hasattr(args, 'opt_betas') and args.opt_betas is not None:
        opt_args['betas'] = args.opt_betas

    print("optimizer settings:", opt_args)

    return optimizer
