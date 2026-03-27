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

cfg.basic.use_ins_encoder = True
cfg.basic.use_ins_decoder = True
cfg.basic.selected_ano_decoder = 'rnn' # 'rnn' or 'memory' or 'plain'

cfg.proxy_task.task_names = []
cfg.proxy_task.use_ins_anomaly = False
cfg.proxy_task.use_bottleneck_loss = False

if cfg.basic.use_ins_encoder:
    # instace encoder : prompt encoder
    cfg.ins_encoder.pmt_encoder.prompt_type = 'boxes'
    cfg.ins_encoder.pmt_encoder.embed_dim = IL(lambda c: c.basic.hid_dim, priority=0, rel=True)
    # update from Dota
    cfg.ins_encoder.pmt_encoder.image_embedding_size = (60 , 80)
    cfg.ins_encoder.pmt_encoder.input_image_size = (480 , 640)
    cfg.ins_encoder.resize.target_size = IL(lambda c: c.ins_encoder.pmt_encoder.input_image_size, priority=0, rel=True)
    cfg.ins_encoder.resize.original_size = (720, 1280)

    # instace encoder : prompt decoder
    cfg.ins_encoder.pmt_decoder.transformer_dim = IL(lambda c: c.basic.hid_dim, priority=0, rel=True)
    cfg.ins_encoder.pmt_decoder.twoway.depth = 2
    cfg.ins_encoder.pmt_decoder.twoway.embedding_dim = IL(lambda c: c.basic.hid_dim, priority=0, rel=True)
    cfg.ins_encoder.pmt_decoder.twoway.num_heads = 8
    cfg.ins_encoder.pmt_decoder.twoway.mlp_dim = 2048
    # instance-aware frame embedding aggregation type:  'mean' 'bottle_neck'
    cfg.ins_encoder.aggregation_type = 'bottle_neck'
    if cfg.ins_encoder.aggregation_type == 'bottle_neck':
        cfg.ins_encoder.bottle_aug.hid_dim = IL(lambda c: c.basic.trans.hid_dim, priority=1, rel=True)
        cfg.ins_encoder.bottle_aug.nhead = IL(lambda c: c.basic.trans.nhead, priority=1, rel=True)
        cfg.ins_encoder.bottle_aug.ffn_dim = IL(lambda c: c.basic.trans.ffn_dim, priority=1, rel=True)
        cfg.ins_encoder.bottle_aug.dropout= IL(lambda c: c.basic.trans.dropout, priority=1, rel=True)


if cfg.basic.use_ins_decoder:
    # instance decoder
    cfg.ins_decoder.num_query_token = 12
    cfg.ins_decoder.hid_dim = IL(lambda c: c.basic.hid_dim)
    cfg.ins_decoder.initializer_range = 0.02
    cfg.ins_decoder.block_depth = 2
    cfg.ins_decoder.block.hid_dim = IL(lambda c: c.basic.trans.hid_dim, priority=1, rel=True)
    cfg.ins_decoder.block.nhead = IL(lambda c: c.basic.trans.nhead, priority=1, rel=True)
    cfg.ins_decoder.block.ffn_dim = IL(lambda c: c.basic.trans.ffn_dim, priority=1, rel=True)
    cfg.ins_decoder.block.dropout= IL(lambda c: c.basic.trans.dropout, priority=1, rel=True)


if cfg.basic.selected_ano_decoder == 'rnn':
    # Anomaly Decoder : rnn regressor
    cfg.ano_decoder.type = 'rnn'
    cfg.ano_decoder.reducer.has_obj_embs = IL(lambda c: c.basic.use_ins_encoder)
    # MLP for object embedding to latent
    if cfg.basic.use_ins_decoder:
        cfg.ano_decoder.reducer.mlp_dim = IL(lambda c: c.ins_decoder.num_query_token*c.basic.hid_dim)
        cfg.ano_decoder.reducer.mlp_depth = 2

    cfg.ano_decoder.reducer.pool_shape = (6,6) 
    cfg.ano_decoder.reducer.pool_dim = IL(lambda c: c.basic.hid_dim*
                                                c.ano_decoder.reducer.pool_shape[0]*
                                                c.ano_decoder.reducer.pool_shape[1])
    cfg.ano_decoder.reducer.dim_latent = IL(lambda c: c.basic.hid_dim, priority=0, rel=True)
    cfg.ano_decoder.reducer.dropout = IL(lambda c: c.basic.dropout)
    cfg.ano_decoder.regressor.dim_latent = IL(lambda c: c.ano_decoder.reducer.dim_latent)
    cfg.ano_decoder.regressor.rnn_state_size = 256
    cfg.ano_decoder.regressor.rnn_cell_num = 3
    cfg.ano_decoder.regressor.hidden_dim = IL(lambda c: c.basic.hid_dim)
    cfg.ano_decoder.regressor.dropout = IL(lambda c: c.basic.dropout)
    cfg.ano_decoder.regressor.output_dim = 2
    cfg.ano_decoder.regressor.num_layers = 2
    cfg.ano_decoder.regressor.apply_ins_cls = IL(lambda c: c.proxy_task.use_ins_anomaly)
