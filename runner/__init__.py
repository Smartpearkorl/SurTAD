'''
1. define some shared parameters
'''
from pathlib import Path
# dataset folder
DATA_FOLDER = Path('/data/qh/STDA/data/')
DoTA_FOLDER = Path('/data/qh/DoTA/data') 
DADA_FOLDER = Path('/data/qh/DADA/')
META_FOLDER = DATA_FOLDER / 'metadata'
# pretrained model folder
pretrained_weight_path = Path('/data/qh/pretrain_models/')
vst_pretrained_sthv2_path = pretrained_weight_path / 'swin_base_patch244_window1677_sthv2.pth'
vst_pretrained_kinetics400_path = pretrained_weight_path / 'swin_base_patch244_window877_kinetics400_22k.pth'

# simpletad_ft_dapt_l_weight_path = pretrained_weight_path / 'simple_tad' / 'simpletad_ft-dota_dapt-vm1-l_auroc.pth'
simpletad_ft_dapt_l_weight_path = pretrained_weight_path / 'simple_tad' / 'simpletad_dapt_videomae-l_ep12.pth'
yolov9_c_convertd_weight_path = pretrained_weight_path / 'yolov9-c-converted.pt'
# font path for plotting figures
FONT_FOLDER = "/data/qh/dependency/arial.ttf"
CHFONT_FOLDER = "/data/qh/dependency/microsoft_yahei.ttf"
# debug folder: mostly applied in /PromptTAD/runner/src/custom_tools.py, where there are some debugging tools
DEBUG_FOLDER = Path("/data/qh/STDA/output/debug/")





