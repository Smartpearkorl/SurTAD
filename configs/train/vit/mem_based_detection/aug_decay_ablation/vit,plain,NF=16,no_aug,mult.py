from alchemy_cat.dl_config import Config, load_config, Config ,Param2Tune,IL

basecfg_path = './configs/base/basecfg.py'
datacfg_path = './configs/base/dota_video_vit_rgb.py'
# datacfg_path = './configs/base/dota_video_vit_rgb_448.py'
'''

Need To Check

'''
ano_decoder = 'plain' # 'rnn' or 'memory' or 'plain'
if ano_decoder == 'memory':
    modelcfg_path = './configs/model/poma/vit/mem_based_detection/vit_base_mem.py'
elif ano_decoder == 'plain':
    modelcfg_path = './configs/model/poma/vit/mem_based_detection/vit_plain.py'

cfg = Config()
cfg.basecfg = Config(basecfg_path)
cfg.datacfg = Config(datacfg_path) 
cfg.modelcfg = Config(modelcfg_path)
cfg.unfreeze()

# update batch size
cfg.basecfg.basic.batch_size = 24
# update datacfg
patch_len = 14
patch_size = 16
cfg.datacfg.train_dataset.VCL = 30
cfg.datacfg.train_dataset.trans_cfg.aug_type = 'plain' # 'plain' # randaug

# update NF
cfg.modelcfg.NF = 16

cfg.basecfg.optimizer.lr_mult_dict = {'vit_model':0.2}
# cfg.basecfg.optimizer.layer_decay = 0.6
# update some data
cfg.basecfg.basic.model_type = 'poma_vit_base_mem'
cfg.basecfg.basic.VCL = cfg.datacfg.train_dataset.VCL
cfg.basecfg.basic.NF = cfg.modelcfg.NF

# update model-aware cfg
if ano_decoder == 'memory':
    cfg.modelcfg.vit.img_size = patch_len * patch_size   
    if cfg.modelcfg.basic.use_ins_encoder:
        # update from Dota
        cfg.modelcfg.ins_encoder.pmt_encoder.image_embedding_size = (patch_len , patch_len)
        cfg.modelcfg.ins_encoder.pmt_encoder.input_image_size = (patch_len * patch_size , patch_len * patch_size)
        cfg.modelcfg.ins_encoder.resize.target_size = IL(lambda c: c.modelcfg.ins_encoder.pmt_encoder.input_image_size, priority=0, rel=False)

    cfg.modelcfg.ins_encoder.pmt_decoder.twoway.depth = 2
    cfg.modelcfg.ano_decoder.regressor.Qformer_depth = 2
    cfg.modelcfg.ano_decoder.apply_vis_mb = True
    cfg.modelcfg.ano_decoder.apply_obj_mb = False
    cfg.modelcfg.ins_encoder.aggregation_type = 'mean' # 'bottle_neck' 'mean'
    if not cfg.modelcfg.ano_decoder.apply_obj_mb:
        cfg.modelcfg.ins_encoder = {}  # prompt encoder not needed if no object memory bank

    cfg.modelcfg.ano_decoder.regressor.Qformer_cfg.apply_vis_mb = IL(lambda c: c.modelcfg.ano_decoder.apply_vis_mb)
    cfg.modelcfg.ano_decoder.regressor.Qformer_cfg.apply_obj_mb = IL(lambda c: c.modelcfg.ano_decoder.apply_obj_mb)
    cfg.modelcfg.ano_decoder.memory_bank_length = 8
    cfg.modelcfg.ano_decoder.regressor.Qformer_cfg.memory_bank_length = IL(lambda c: c.modelcfg.ano_decoder.memory_bank_length)

    cfg.modelcfg.proxy_task.use_ins_anomaly = False
    cfg.modelcfg.ano_decoder.regressor.apply_ins_cls = IL(lambda c: c.modelcfg.proxy_task.use_ins_anomaly)
    if cfg.modelcfg.proxy_task.use_ins_anomaly:
        cfg.basecfg.basic.apply_ins_loss = True
        cfg.basecfg.basic.ins_loss_weight = 1

    cfg.modelcfg.proxy_task.use_bottleneck_loss = False
    if cfg.modelcfg.proxy_task.use_bottleneck_loss:
        cfg.basecfg.basic.apply_bottleneck_loss = True
        cfg.basecfg.basic.bottle_loss_weight = 1