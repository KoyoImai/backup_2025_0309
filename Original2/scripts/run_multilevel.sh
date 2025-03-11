# GPUデバイスの指定
export CUDA_VISIBLE_DEVICES="1"


# CIFAR10
# 通常のパラメータの設定
# SEED=777
# NUM_PT=30
# FEAT_DIM=128
# BATCH_SIZE=512
# LEARNING_RATE=0.5

# # 継続学習的なパラメータ
# MEM_SIZE=500
# EPOCHS=100
# S_EPOCHS=500
# TEMP=0.07
# CURRENT_TEMP=0.2
# PAST_TEMP=0.01
# DISTILL_POWER=1.0


# python3 main_multilevel.py --batch_size $BATCH_SIZE --model resnet18 --dataset cifar10\
#     --mem_size $MEM_SIZE --epochs $EPOCHS --start_epoch $S_EPOCHS --learning_rate 0.5 --seed $SEED\
#     --temp $TEMP --current_temp $CURRENT_TEMP --past_temp $PAST_TEMP --distill_power $DISTILL_POWER\
#     --cosine --feat_dim $FEAT_DIM\
#     --original_name "main_proto_cifar10_sepoch${S_EPOCHS}_epoch${EPOCHS}_buff${MEM_SIZE}_seed${SEED}_fdim${FEAT_DIM}_p${NUM_PT}_wprot${WEIGHT_PROT}_dpower${DISTILL_POWER}_2025_0310"



# CIFAR100
# 通常のパラメータの設定
SEED=777
NUM_PT=30
FEAT_DIM=128
BATCH_SIZE=512
LEARNING_RATE=0.5

# 継続学習的なパラメータ
MEM_SIZE=500
EPOCHS=100
S_EPOCHS=500
TEMP=0.07
CURRENT_TEMP=0.2
PAST_TEMP=0.01
DISTILL_POWER=1.0


python3 main_multilevel.py --batch_size $BATCH_SIZE --model resnet18 --dataset cifar100\
    --mem_size $MEM_SIZE --epochs $EPOCHS --start_epoch $S_EPOCHS --learning_rate 0.5 --seed $SEED\
    --temp $TEMP --current_temp $CURRENT_TEMP --past_temp $PAST_TEMP --distill_power $DISTILL_POWER\
    --cosine --feat_dim $FEAT_DIM\
    --original_name "main_proto_cifar100_sepoch${S_EPOCHS}_epoch${EPOCHS}_buff${MEM_SIZE}_seed${SEED}_fdim${FEAT_DIM}_p${NUM_PT}_wprot${WEIGHT_PROT}_dpower${DISTILL_POWER}_2025_0310"