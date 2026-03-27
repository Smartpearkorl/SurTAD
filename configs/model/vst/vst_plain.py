from alchemy_cat.dl_config import Config,IL
import sys
from runner import pretrained_weight_path
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
cfg.vst.type = 'swin_base_patch244_window1677_sthv2'  # swin_base_patch244_window1677_sthv2 swin_base_patch244_window877_kinetics400_22k
cfg.vst.checkpoint = IL(lambda c: str(pretrained_weight_path/f'{c.vst.type}.pth'), priority=0, rel=True)

# fpn 
cfg.fpn.dimen_reduce_type = 'mlp'
cfg.basic.selected_ano_decoder = 'plain' # 'rnn' or 'memory' or 'plain'

if cfg.basic.selected_ano_decoder == 'plain':
    # Anomaly Decoder : plain regressor
    cfg.ano_decoder.type = 'plain'
    cfg.ano_decoder.regressor.input_dim = IL(lambda c: c.basic.hid_dim)
    cfg.ano_decoder.regressor.hidden_dim = IL(lambda c: c.basic.hid_dim)
    cfg.ano_decoder.regressor.dropout = IL(lambda c: c.basic.dropout)
    cfg.ano_decoder.regressor.output_dim = 2
    cfg.ano_decoder.regressor.num_layers = 2

