from alchemy_cat.dl_config import Config ,IL
import torch
cfg = Config()

# basic
cfg.basic.seed = 123
cfg.basic.total_epoch = 12
cfg.basic.snapshot_interval = 6

cfg.basic.test_inteval = 3
cfg.basic.batch_size = 64
cfg.basic.test_batch_size = 8
cfg.basic.num_workers = 8

cfg.basic.train_len = 1047 # dota 3196 dada 198 stad 1047
cfg.basic.n_nodes = 3
cfg.basic.base_lr = 1e-5
cfg.basic.lr = IL(lambda c: c.basic.base_lr  * c.basic.batch_size / 8 , priority=0)
cfg.basic.max_iter = IL(lambda c:  int(c.basic.total_epoch * c.basic.train_len // (c.basic.n_nodes*c.basic.batch_size)) , priority=0) 
cfg.basic.class_weights = (0.3 , 0.7) #(0.3 , 0.7)

cfg.basic.train_debug.debug_train_weight = False # False
cfg.basic.train_debug.debug_train_weight_level = 2
cfg.basic.train_debug.debug_train_grad = False
cfg.basic.train_debug.debug_train_grad_level = 2
cfg.basic.train_debug.debug_loss = True

# optimizer and lr_scheduler strategy
cfg.optimizer.type = 'adamw'
cfg.optimizer.cls = IL(lambda c: c.optimizer.type , priority=0)
cfg.optimizer.lr = IL(lambda c: c.basic.lr , priority=1)

if cfg.optimizer.type == 'sgd':
    cfg.optimizer.momentum = 0.9
    cfg.optimizer.weight_decay = 0.001
elif cfg.optimizer.type == 'adamw':
    cfg.optimizer.weight_decay = 0.001

cfg.sched.cls = 'SequentialLR'

cfg.sched.warm.warm_iters = IL(lambda c: int(c.basic.train_len // (c.basic.n_nodes * c.basic.batch_size )), priority=1)  # ~ 1 epochs
cfg.sched.warm.ini.start_factor = 0.05
cfg.sched.warm.ini.end_factor = 1.0
cfg.sched.warm.ini.total_iters = IL(lambda c: c.sched.warm.warm_iters)
cfg.sched.warm.cls = 'LinearLR'

cfg.sched.main.ini.T_max = IL(lambda c: c.basic.max_iter - c.sched.warm.warm_iters)
cfg.sched.main.ini.eta_min = IL(lambda c: 0.05*c.optimizer.lr )
cfg.sched.main.cls = 'CosineAnnealingLR'






