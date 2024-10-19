#
# python -m torch.distributed.launch --master_port 12347 --nproc_per_node=8 \
#             main.py  something  RGB --arch resnet50 --num_segments 8 --gd 20 --lr 0.01 \
#             --lr_scheduler step --lr_steps  30 45 55 --epochs 61 --batch-size 8 \
#             --wd 5e-4 --dropout 0.5 --consensus_type=avg --eval-freq=1 -j 6 --npb 
            # --resume '/home/yckj3949/action/TDN-mask-0819-2-copy-3-backup/checkpoint/TDN__something_RGB_resnet50_avg_segment8_e61/3_epoch_ckpt.pth.tar'



# python -m torch.distributed.launch --master_port 12347 --nproc_per_node=7 \
#             main.py  something  RGB --arch resnet50 --num_segments 8 --gd 20 --lr 0.00875 \
#             --lr_scheduler step --lr_steps  30 45 55 --epochs 60 --batch-size 8 \
#             --wd 5e-4 --dropout 0.5 --consensus_type=avg --eval-freq=1 -j 6 --npb
#             #--resume /home/yckj3949/sheng/TDN-mask-0721/checkpoint/TDN__something_RGB_resnet50_avg_segment8_e60/42_epoch_ckpt.pth.tar

# CUDA_VISIBLE_DEVICES=0 python test_models_center_crop.py something \
#     --archs='resnet50' --weights '/home/ps/sheng/TDN-mask-0519/checkpoint/TDN__something_RGB_resnet50_avg_segment8_e60/1_epoch_ckpt.pth.tar'  --test_segments=8  \
#     --test_crops=1 --batch_size=16  --gpus 0 --output_dir /home/ps/sheng/TDN-mask-0519 -j 4 --clip_index=0

# python -m torch.distributed.launch --master_port 12347 --nproc_per_node=1 \
#     main.py hmdb51 RGB --arch resnet50  --num_segments 16 --gd 20 --lr 0.0003125 \
#     --lr_steps 10 20 --epochs 25 --batch-size 4 --dropout 0.8 --consensus_type=avg \
#     --eval-freq=1 -j 4 --tune_from='checkpoint/best.pth.tar' --npb \
#     --resume /home/sheng/sheng/TDN-mask-0519/checkpoint/TDN__hmdb51_RGB_resnet50_avg_segment16_e25/8_epoch_ckpt.pth.tar


python -m torch.distributed.launch --master_port 12357 --nproc_per_node=8 \
            main.py  UAVHuman  RGB --arch resnet50 --num_segments 8 --gd 20 --lr 0.01 \
            --lr_scheduler step --lr_steps  30 45 55 --epochs 60 --batch-size 8 \
            --wd 5e-4 --dropout 0.5 --consensus_type=avg --eval-freq=1 -j 6 --npb 
            # --resume '/home/yckj3949/sheng/TDN-mask-0819-2-copy-3-backup-human-dataset/checkpoint/TDN__UAVHuman_RGB_resnet50_avg_segment16_e60/best.pth.tar'
