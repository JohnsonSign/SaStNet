########## step1 ###########
# CUDA_VISIBLE_DEVICES=0 python3 test_models_center_crop.py somethingv2 \
# --archs='resnet50' --weights /ml_workspace/yckj3949/code/sxx/TDN-mask-0819-2-copy-3/60_epoch_ckpt.pth.tar  --test_segments=16  \
# --test_crops=1 --batch_size=8  --gpus 0 --output_dir ./sthv2_crop1_test -j 4 --clip_index=0

########## step2 ##########
# python3 pkl_to_results.py --num_clips 1 --test_crops 1 --output_dir ./sthv2_crop1_test

######### step1 ##########
CUDA_VISIBLE_DEVICES=0 python3 test_models_three_crops.py  somethingv2 \
--archs='resnet50' --weights /ml_workspace/yckj3949/code/sxx/TDN-mask-0819-2-copy-3/sthv2_f16_best.tar  --test_segments=16 \
--test_crops=3 --batch_size=16  --full_res --gpus 0 --output_dir ./sthv2_test_li  \
-j 4 --clip_index=0

########## step2 #########
# python pkl_to_results.py --num_clips 4 --test_crops 3 --output_dir ./sthv1_crop3_test


### 使用test_models_center_crop，设置crop 3次，acc: 49.11%





