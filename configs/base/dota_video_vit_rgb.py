from alchemy_cat.dl_config import Config , IL
import sys
from runner import DoTA_FOLDER
from runner.src.dataset import Dota

cfg = Config()

'''
 prepare train dataset
'''
patch_h = 14
patch_w = 14
patch_size = 16
cfg.crop_h = patch_h * patch_size 
cfg.crop_w = patch_w * patch_size  

# cfg.dataset.type = Dota 
cfg.dataset.name = 'dota'
cfg.dataset.root_path = str(DoTA_FOLDER)
cfg.dataset.phase = 'train'
cfg.dataset.pre_process_type = 'rgb' # rgb or sam_emb

cfg.dataset.trans_cfg.aug_type = 'randaug' # 'randaug' 'randaug'
cfg.dataset.trans_cfg.crop_h = IL(lambda c: c.crop_h, priority=0)
cfg.dataset.trans_cfg.crop_w = IL(lambda c: c.crop_w, priority=0)
cfg.dataset.trans_cfg.origin_shape = (720, 1280)
cfg.dataset.trans_cfg.augmix = True
cfg.dataset.trans_cfg.mean_std = 'imagenet'
# random aug
cfg.dataset.trans_cfg.aa = 'rand-m6-n3-mstd0.5-inc1'
cfg.dataset.trans_cfg.train_interpolation = 'bicubic'
# flip
cfg.dataset.trans_cfg.vertical_flip_prob = 0.0
cfg.dataset.trans_cfg.horizontal_flip_prob = 0.5
# random erase
cfg.dataset.trans_cfg.rand_erase = True
cfg.dataset.trans_cfg.erase_cfg.reprob = 0.25
cfg.dataset.trans_cfg.erase_cfg.remode = 'pixel'
cfg.dataset.trans_cfg.erase_cfg.recount = 1

cfg.dataset.VCL = 20
cfg.dataset.sorted_num_frames = False
cfg.dataset.data_type = ''
cfg.train_dataset = cfg.dataset

'''
 prepare val dataset
'''
cfg.dataset.phase = 'val'
cfg.dataset.trans_cfg = Config()
cfg.dataset.trans_cfg.aug_type = 'plain'
cfg.dataset.trans_cfg.crop_h = IL(lambda c: c.crop_h, priority=0)
cfg.dataset.trans_cfg.crop_w = IL(lambda c: c.crop_w, priority=0)
cfg.dataset.trans_cfg.origin_shape = (720, 1280)
cfg.dataset.trans_cfg.mean_std = 'imagenet'
cfg.dataset.VCL = None
cfg.dataset.sorted_num_frames = True
cfg.dataset.data_type = '' # '' 'sub_' 'select_train_'
cfg.test_dataset = cfg.dataset
# delete template variable data
cfg.dataset = None


