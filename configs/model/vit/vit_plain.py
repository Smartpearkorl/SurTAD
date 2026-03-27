from alchemy_cat.dl_config import Config,IL
import sys
from runner import simpletad_ft_dapt_l_weight_path
from models.prompt_models.poma import poma
cfg = Config()

cfg.model = poma
cfg.model_type = 'poma'
cfg.NF = 4
cfg.basic.hid_dim = 256
cfg.basic.dropout = 0.3

# transformer block
cfg.basic.trans.hid_dim = IL(lambda c: c.basic.hid_dim, priority=0, rel=True)
cfg.basic.trans.nhead = 8
cfg.basic.trans.ffn_dim = 256
cfg.basic.trans.dropout = 0.1

# video swin transformer
cfg.vit.model_name = 'vit_large_patch16_224'
cfg.vit.num_classes = 0
cfg.vit.num_frames = IL(lambda c: c.NF, priority=0, rel=True)
cfg.vit.pretrained_path = str(simpletad_ft_dapt_l_weight_path)

# fpn 
cfg.fpn.apply_linar = True # False True
cfg.fpn.linear.vit_dim = 1024
cfg.fpn.linear.target_dim = IL(lambda c: c.basic.hid_dim, priority=0, rel=True)


cfg.basic.selected_ano_decoder = 'plain' 
if cfg.basic.selected_ano_decoder == 'plain':
    # Anomaly Decoder : plain regressor
    cfg.ano_decoder.type = 'plain'
    cfg.ano_decoder.regressor.input_dim = IL(lambda c: c.basic.hid_dim)
    cfg.ano_decoder.regressor.hidden_dim = IL(lambda c: c.basic.hid_dim)
    cfg.ano_decoder.regressor.dropout = IL(lambda c: c.basic.dropout)
    cfg.ano_decoder.regressor.output_dim = 2
    cfg.ano_decoder.regressor.num_layers = 2
    cfg.ano_decoder.regressor.num_layers = 2
    cfg.ano_decoder.regressor.sub_class_num = 0


