from alchemy_cat.dl_config import Config, load_config, Config ,Param2Tune,IL

basecfg_path = './configs/base/basecfg.py'
datacfg_path = './configs/base/dota_video_vst_rgb.py'
'''

Need To Check

'''
ano_decoder = 'memory' # 'rnn' or 'memory' or 'plain'
if ano_decoder == 'memory':
    modelcfg_path = './configs/model/vst/vst_mem.py'
elif ano_decoder == 'plain':
    modelcfg_path = './configs/model/vst/vst_plain.py'
elif ano_decoder == 'rnn':
    modelcfg_path = './configs/model/vst/vst_rnn.py'


cfg = Config()
cfg.basecfg = Config(basecfg_path)
cfg.datacfg = Config(datacfg_path) 
cfg.modelcfg = Config(modelcfg_path)
cfg.unfreeze()

# update batch size
# cfg.basecfg.basic.train_len = 4597
cfg.basecfg.basic.base_lr = 1e-5
cfg.basecfg.basic.total_epoch = 12
cfg.basecfg.basic.batch_size = 8
# update VCL
cfg.datacfg.train_dataset.data_type = ''  
cfg.datacfg.train_dataset.cfg.VCL = 20
cfg.datacfg.train_dataset.trans_cfg.aug_type = 'plain' # 'plain' # randaug

# update NF
cfg.modelcfg.NF = 4

cfg.basecfg.basic.model_type = 'poma_vit_base_mem'
cfg.basecfg.basic.VCL = cfg.datacfg.train_dataset.cfg.VCL
cfg.basecfg.basic.NF = cfg.modelcfg.NF


if ano_decoder in ['rnn', 'memory']:
    cfg.modelcfg.ins_encoder.pmt_decoder.twoway.depth = 4
    cfg.modelcfg.ins_encoder.aggregation_type = 'mean' # 'bottle_neck' 'mean'

if ano_decoder == 'rnn':
    cfg.modelcfg.basic.use_ins_encoder = True
    cfg.modelcfg.ins_decoder.block_depth = 4
    # cfg.modelcfg.basic.use_ins_decoder = 'mean'

# update model-aware cfg
if ano_decoder == 'memory':
    cfg.modelcfg.ano_decoder.regressor.block_type = 'memblock'  # 'qformer' or 'memblock'
    cfg.modelcfg.ano_decoder.regressor.Qformer_depth = 4
    cfg.modelcfg.ano_decoder.apply_vis_mb = True
    cfg.modelcfg.ano_decoder.apply_obj_mb = True
    cfg.modelcfg.ano_decoder.query_type = 'vit_tokens' # 'vit_tokens'
    cfg.modelcfg.basic.use_ins_encoder = IL(lambda c: c.modelcfg.ano_decoder.apply_obj_mb)

    if cfg.modelcfg.ano_decoder.query_type == 'learnable_tokens':
        cfg.modelcfg.ano_decoder.regressor.Qformer_cfg.vis_attn_rope = False
        cfg.modelcfg.ano_decoder.regressor.Qformer_cfg.obj_attn_rope = False

    if not cfg.modelcfg.ano_decoder.apply_obj_mb:
        cfg.modelcfg.ins_encoder = {}

    cfg.modelcfg.ano_decoder.memory_bank_length = 8


if ano_decoder in ['rnn', 'memory']:
    cfg.modelcfg.proxy_task.use_ins_anomaly = True # False True
    cfg.modelcfg.ano_decoder.regressor.apply_ins_cls = IL(lambda c: c.modelcfg.proxy_task.use_ins_anomaly)
    if cfg.modelcfg.proxy_task.use_ins_anomaly:
        cfg.basecfg.basic.apply_ins_loss = True
        cfg.basecfg.basic.ins_loss_weight = 1

    cfg.modelcfg.proxy_task.use_bottleneck_loss = False
    if cfg.modelcfg.proxy_task.use_bottleneck_loss:
        cfg.basecfg.basic.apply_bottleneck_loss = True
        cfg.basecfg.basic.bottle_loss_weight = 1