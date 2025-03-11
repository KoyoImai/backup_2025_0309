# GPUデバイスの指定
export CUDA_VISIBLE_DEVICES="0"


# CIFAR10
# 通常のパラメータの設定
SEED=777
NUM_PT=30
FEAT_DIM=128
BATCH_SIZE=512
LEARNING_RATE=0.5

# 継続学習的なパラメータ
MEM_SIZE=5000
EPOCHS=100
S_EPOCHS=500
TEMP=0.5
CURRENT_TEMP=0.2
PAST_TEMP=0.01
DISTILL_POWER=1.0


python3 main_co2l.py --batch_size $BATCH_SIZE --model resnet18 --dataset cifar10\
    --mem_size $MEM_SIZE --epochs $EPOCHS --start_epoch $S_EPOCHS --learning_rate 0.5 --seed $SEED\
    --temp $TEMP --current_temp $CURRENT_TEMP --past_temp $PAST_TEMP --distill_power $DISTILL_POWER\
    --cosine --feat_dim $FEAT_DIM\
    --original_name "main_proto_cifar10_sepoch${S_EPOCHS}_epoch${EPOCHS}_buff${MEM_SIZE}_seed${SEED}_fdim${FEAT_DIM}_p${NUM_PT}_wprot${WEIGHT_PROT}_dpower${DISTILL_POWER}_2025_0309"
    
# python3 main_linear.py --dataset cifar10 --seed 777 --feat_dim $FEAT_DIM --num_pt $NUM_PT --target_task 4\
#     --orig ./exp_logs/checkpoints_proto/main_proto_cifar10_sepoch3_epoch3_buff5000_seed777_fdim128_p30_wprot1.5_dpower1.0_2025_0309/results\
#     --ckpt ./exp_logs/checkpoints_proto/main_proto_cifar10_sepoch3_epoch3_buff5000_seed777_fdim128_p30_wprot1.5_dpower1.0_2025_0309/mpdel_param/cifar10_32_resnet18_lr_0.5_decay_0.0001_bsz_128_temp_0.5_trial_777_3_3_0.2_0.01_1.0_cosine\
#     --logpt ./exp_logs/checkpoints_proto/main_proto_cifar10_sepoch3_epoch3_buff5000_seed777_fdim128_p30_wprot1.5_dpower1.0_2025_0309/buffer_log/cifar10_32_resnet18_lr_0.5_decay_0.0001_bsz_128_temp_0.5_trial_777_3_3_0.2_0.01_1.0_cosine\



## tiny-imagenet
# 通常のパラメータの設定
SEED=777
NUM_PT=300
FEAT_DIM=128
BATCH_SIZE=512
LEARNING_RATE=0.1

# 継続学習的なパラメータ
MEM_SIZE=5000
EPOCHS=50
S_EPOCHS=500
TEMP=0.5
CURRENT_TEMP=0.1
PAST_TEMP=0.1
DISTILL_POWER=1.0

# python3 main_proto.py --batch_size $BATCH_SIZE --model resnet18 --dataset tiny-imagenet\
#     --mem_size $MEM_SIZE --epochs $EPOCHS --start_epoch $S_EPOCHS --learning_rate 0.5 --seed $SEED\
#     --temp $TEMP --current_temp $CURRENT_TEMP --past_temp $PAST_TEMP --distill_power $DISTILL_POWER\
#     --cosine --feat_dim $FEAT_DIM --debug_mode\
#     --original_name "main_proto_tiny-imagenet_sepoch${S_EPOCHS}_epoch${EPOCHS}_buff${MEM_SIZE}_seed${SEED}_fdim${FEAT_DIM}_p${NUM_PT}_wprot${WEIGHT_PROT}_dpower${DISTILL_POWER}_2025_0309"