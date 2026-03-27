#!/bin/bash
set +e  

echo "===== Running ====="

###########     mem_based_detection        ##############

# CUDA_VISIBLE_DEVICES=0,1,2 python -m torch.distributed.launch --nproc_per_node=3 --master_port 29650\
#     /home/qh/TDD/SurTAD/runner/main.py \
#     --config /home/qh/TDD/SurTAD/configs/train/vit/base_detection/lr=1e-5,plain.py \
#     --fp16 --distributed \
#     --output "/data/qh/STDA/output/subcls,ep=24,lr=1e-5,plain" \
#     --phase test \
#     --epoch 24
# wait

# CUDA_VISIBLE_DEVICES=0,1,2 python -m torch.distributed.launch --nproc_per_node=3 --master_port 29650\
#     /home/qh/TDD/SurTAD/runner/main.py \
#     --config /home/qh/TDD/SurTAD/configs/train/vit/base_detection/lr=1e-5,plain.py \
#     --fp16 --distributed \
#     --output "/data/qh/STDA/output/in,subcls,ep=24,lr=1e-5,plain" \
#     --phase train \
#     --epoch -1
# wait


# CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 --master_port 29650\
#     /home/qh/TDD/SurTAD/runner/main.py \
#     --config /home/qh/TDD/SurTAD/configs/train/vit/base_detection/stad_debug.py \
#     --fp16 --distributed \
#     --output "/data/qh/STDA/output/debug/vscode_debug/" \
#     --phase train \
#     --epoch -1
# wait


CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 --master_port 29650 /home/qh/TDD/SurTAD/end2end.py 
wait



echo "===== Finished ====="
echo "=====  All jobs completed ====="







