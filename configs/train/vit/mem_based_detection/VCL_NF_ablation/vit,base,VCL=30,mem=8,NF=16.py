from alchemy_cat.dl_config import Config, load_config, Config ,Param2Tune,IL


basecfg_path = './configs/base/basecfg.py'
datacfg_path = './configs/base/dota_video_vit_rgb.py'
modelcfg_path = './configs/model/poma/vit/mem_based_detection/vit_base_mem.py'


cfg = Config()
cfg.basecfg = load_config(basecfg_path)
cfg.datacfg = load_config(datacfg_path)
cfg.modelcfg = load_config(modelcfg_path)

cfg.unfreeze()

# update batch size
cfg.basecfg.basic.batch_size = 24
cfg.basecfg.basic.lr = IL(lambda c: 0.00001 * c.basecfg.basic.batch_size / 8 , priority=0)
cfg.basecfg.basic.max_iter = IL(lambda c:  int(c.basecfg.basic.total_epoch * 3196 // (c.basecfg.basic.n_nodes*c.basecfg.basic.batch_size)) , priority=0) 

# update VCL
cfg.datacfg.train_dataset.cfg.VCL = 30
# prepare train dataset
cfg.datacfg.train_dataset.data = cfg.datacfg.train_dataset.cfg.type(**cfg.datacfg.train_dataset.cfg)
# prepare test dataset
cfg.datacfg.test_dataset.data = cfg.datacfg.test_dataset.cfg.type(**cfg.datacfg.test_dataset.cfg)

# NF
cfg.modelcfg.NF = 16
cfg.modelcfg.vit.num_frames = IL(lambda c: c.modelcfg.NF, priority=0, rel=False)

cfg.unfreeze()
# update some data
cfg.basecfg.basic.model_type = 'poma_vit_base_mem'
cfg.basecfg.basic.VCL = cfg.datacfg.train_dataset.cfg.VCL
cfg.basecfg.basic.NF = cfg.modelcfg.NF

cfg.basecfg.lr_mult_dict = {'vit_model':0.1}

cfg.basecfg.basic.proxy_tasks = cfg.modelcfg.proxy_task.task_names
if cfg.modelcfg.proxy_task.use_ins_anomaly:
    cfg.basecfg.basic.apply_ins_loss = True
    cfg.basecfg.basic.ins_loss_weight = 1

# update model-aware cfg
cfg.modelcfg.ano_decoder.regressor.Qformer_depth = 2
cfg.modelcfg.ano_decoder.apply_vis_mb = True
cfg.modelcfg.ano_decoder.apply_obj_mb = False
if not cfg.modelcfg.ano_decoder.apply_obj_mb:
    cfg.modelcfg.ins_encoder = {}  # prompt encoder not needed if no object memory bank

cfg.modelcfg.ano_decoder.regressor.Qformer_cfg.apply_vis_mb = IL(lambda c: c.modelcfg.ano_decoder.apply_vis_mb)
cfg.modelcfg.ano_decoder.regressor.Qformer_cfg.apply_obj_mb = IL(lambda c: c.modelcfg.ano_decoder.apply_obj_mb)
cfg.modelcfg.ano_decoder.memory_bank_length = 8
cfg.modelcfg.ano_decoder.regressor.Qformer_cfg.memory_bank_length = IL(lambda c: c.modelcfg.ano_decoder.memory_bank_length)