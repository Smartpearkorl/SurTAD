from alchemy_cat.dl_config import Config

# cfg = Config(caps='/home/qh/TDD/MemTAD/alchemy_test.py')
# cfg.batch_size = 128 * 2 

cfg = Config()
cfg.basic = Config('/home/qh/TDD/MemTAD/alchemy_test.py')
cfg.basic.batch_size = 128 * 2 