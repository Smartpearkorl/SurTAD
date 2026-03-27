from alchemy_cat.dl_config import Config, load_config, Config ,Param2Tune,IL


basecfg_path = './configs/base/basecfg.py'
# datacfg_path = './configs/base/dota_movad_rgb_cfg.py'
datacfg_path = './configs/base/dota_video_vst_rgb.py'
modelcfg_path = './configs/model/poma/vst/memory/vst_base_mem.py'

cfg = Config()
cfg.basecfg = Config(basecfg_path)
cfg.datacfg = Config(datacfg_path) 
cfg.modelcfg = Config(modelcfg_path)
cfg.unfreeze()

# update batch size
cfg.basecfg.basic.batch_size = 8
# update VCL
cfg.datacfg.train_dataset.cfg.VCL = 20
cfg.datacfg.train_dataset.trans_cfg.aug_type = 'plain' # 'plain' # randaug

# update NF
cfg.modelcfg.NF = 4

# update some data
cfg.basecfg.basic.model_type = 'poma_vit_base_mem'
cfg.basecfg.basic.VCL = cfg.datacfg.train_dataset.cfg.VCL
cfg.basecfg.basic.NF = cfg.modelcfg.NF

cfg.modelcfg.vst.type = 'swin_base_patch244_window877_kinetics400_22k'  # swin_base_patch244_window1677_sthv2 swin_base_patch244_window877_kinetics400_22k
cfg.modelcfg.ins_encoder.pmt_decoder.twoway.depth = 4
cfg.modelcfg.ano_decoder.regressor.Qformer_depth = 4
cfg.modelcfg.ano_decoder.apply_vis_mb = True
cfg.modelcfg.ano_decoder.apply_obj_mb = True
cfg.modelcfg.ano_decoder.query_type = 'vit_tokens' # 'vit_tokens'
cfg.modelcfg.ins_encoder.aggregation_type = 'mean' # 'bottle_neck' 'mean'
cfg.modelcfg.basic.use_ins_encoder = IL(lambda c: c.modelcfg.ano_decoder.apply_obj_mb)

if cfg.modelcfg.ano_decoder.query_type == 'learnable_tokens':
    cfg.modelcfg.ano_decoder.regressor.Qformer_cfg.vis_attn_rope = False
    cfg.modelcfg.ano_decoder.regressor.Qformer_cfg.obj_attn_rope = False

if not cfg.modelcfg.ano_decoder.apply_obj_mb:
    cfg.modelcfg.ins_encoder = {}

cfg.modelcfg.ano_decoder.memory_bank_length = 8

cfg.modelcfg.proxy_task.use_ins_anomaly = False # False
cfg.modelcfg.ano_decoder.regressor.apply_ins_cls = IL(lambda c: c.modelcfg.proxy_task.use_ins_anomaly)
if cfg.modelcfg.proxy_task.use_ins_anomaly:
    cfg.basecfg.basic.apply_ins_loss = True
    cfg.basecfg.basic.ins_loss_weight = 1

cfg.modelcfg.proxy_task.use_bottleneck_loss = False
if cfg.modelcfg.proxy_task.use_bottleneck_loss:
    cfg.basecfg.basic.apply_bottleneck_loss = True
    cfg.basecfg.basic.bottle_loss_weight = 1