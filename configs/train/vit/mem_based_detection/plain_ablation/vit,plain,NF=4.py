from alchemy_cat.dl_config import Config, load_config, Config ,Param2Tune,IL

basecfg_path = './configs/base/basecfg.py'
datacfg_path = './configs/base/dota_video_vit_rgb.py'
# datacfg_path = './configs/base/dota_video_vit_rgb_448.py'
modelcfg_path = './configs/model/poma/vit/mem_based_detection/vit_plain.py'

cfg = Config()
cfg.basecfg = Config(basecfg_path)
cfg.datacfg = Config(datacfg_path) 
cfg.modelcfg = Config(modelcfg_path)
cfg.unfreeze()

cfg.basecfg.basic.total_epoch = 24
# update batch size
cfg.basecfg.basic.batch_size = 64
# update VCL
cfg.datacfg.train_dataset.cfg.VCL = 20
# prepare train dataset
cfg.datacfg.train_dataset.data = cfg.datacfg.train_dataset.cfg.type(**cfg.datacfg.train_dataset.cfg)
# prepare test dataset
cfg.datacfg.test_dataset.data = cfg.datacfg.test_dataset.cfg.type(**cfg.datacfg.test_dataset.cfg)

# update NF
cfg.modelcfg.NF = 4

# update bacbone mult
cfg.basecfg.lr_mult_dict = {'vit_model':0.2}

# update some data
cfg.basecfg.basic.model_type = 'poma_vit_base_mem'
cfg.basecfg.basic.VCL = cfg.datacfg.train_dataset.cfg.VCL
cfg.basecfg.basic.NF = cfg.modelcfg.NF

