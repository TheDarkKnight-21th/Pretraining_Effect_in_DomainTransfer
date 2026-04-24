# !!!!!! None Pruning !!!!!! No class map.

# Please refer to the pretrain arguments comments and change the arguments "if necessary".

# pretrain arguments (These are the hyperparameters I experimented with.)

MODEL="convnext_tiny.fb_in1k" # You must always append .fb_in1k. This is because we pretrain by initializing with IN1k weights, not from scratch. (For swin, it's .ms_in1k. Please note that the company name differs depending on the model.)
DATA_DIR=$1 #"../dataset/imagenet21k_train/train" #../dataset/imagenet21k/train"  # Set dataset path
BATCH_SIZE=$7 # The total batch size is "1024" divided by the number of GPUs. EX ) <= 1024 / Number of GPUs (4)
GPU=1 # Number of pretrain GPUs
GPU_NUMBER=$6 # Number of pretrain GPUs
GRAD_ACCUM_STEPS=1 # The target total batch size is 1024. If your GPU VRAM is insufficient, since < Total Batch Size = Number of GPUs x Batch per GPU x grad_accum_steps >, please adjust the batch size and grad_accum_steps appropriately to match the total batch size.
OPTIMIZER="adamw"  # We use adamw as the optimizer for all pretraining.
SCHEDULER="cosine"
EPOCHS=24
WARMUP_EPOCHS=2
EXPERIMENT="UTL-21k"
WEIGHT_DECAY=5e-2
MIXUP=0.8
CUTMIX=1.0
LOG_INTERVAL=100
SCHED_ON_UPDATES="--sched-on-updates"
NUM_CLASSES=19167
PRETRAINED="--pretrained"  
DATASET=${2:-"default"}

# grid search parameters
LEARNING_RATE=$4 # Please add parameters for grid search! | ex : (1e-4 1e-6 1e-7 ...)
WARMUP_LR=$5 # Please add parameters for grid search! | ex : (1e-6 1e-7 1e-8 ...)

# GPU number => Please change if necessary.
GPU_NUMBER1=$GPU_NUMBER
# Execute pretrain
cd pytorch-image-models
CUDA_VISIBLE_DEVICES=$GPU_NUMBER1 python train.py\
    --model $MODEL \
    --data-dir $DATA_DIR \
    --dataset $DATASET \
    --batch-size $BATCH_SIZE \
    --grad-accum-steps $GRAD_ACCUM_STEPS \
    --opt $OPTIMIZER \
    --lr $LEARNING_RATE \
    --warmup-epochs $WARMUP_EPOCHS \
    --warmup-lr $WARMUP_LR \
    --workers 12 \
    --sched $SCHEDULER \
    --epochs $EPOCHS \
    --experiment $EXPERIMENT \
    --weight-decay $WEIGHT_DECAY \
    --mixup $MIXUP \
    --cutmix $CUTMIX \
    --log-interval $LOG_INTERVAL \
    $SCHED_ON_UPDATES \
    --num-classes $NUM_CLASSES \
    $PRETRAINED \
    --log-wandb

cd ..

cd Benchmark_Domain_Transfer
pwd
#pretrained model path
CUSTOM_PRETRAINED_MODEL_PATH="../pytorch-image-models/output/train/${EXPERIMENT}/${MODEL}_${LEARNING_RATE}_${WARMUP_LR}_${WARMUP_EPOCHS}_${BATCH_SIZE}x${GPU}_${SCHEDULER}_single/checkpoint-$((${EPOCHS}-1)).pth.tar"


# Execute downstream task

# GPU number => Please change if necessary.
GPU_NUMBER2=$GPU_NUMBER

DA_PATH=$3

OFFICEHOME_PATH="${DA_PATH}/office-home/"
DOMIANNET_PATH="${DA_PATH}/domainnet/"
CUB_PATH="${DA_PATH}/cub/"

CUDA_VISIBLE_DEVICES=$GPU_NUMBER2 python main.py $OFFICEHOME_PATH \
    -d OfficeHome \
    -s Rw \
    -t Ar Cl Pr \
    -a $MODEL \
    --seed 0 \
    --log baseline/ \
    --custom-pretrained $CUSTOM_PRETRAINED_MODEL_PATH \
    --log-wandb



CUDA_VISIBLE_DEVICES=$GPU_NUMBER2 python main.py $DOMIANNET_PATH \
    -d DomainNet \
    -s r \
    -t c i p q s \
    -a $MODEL  \
    --seed 0 \
    --log baseline_domainnet/ \
    --custom-pretrained $CUSTOM_PRETRAINED_MODEL_PATH \
    --log-wandb



CUDA_VISIBLE_DEVICES=$GPU_NUMBER2 python main.py $CUB_PATH \
    -d CUB \
    -s Rw \
    -t Pr \
    -a $MODEL \
    --seed 0 \
    --log baseline_CUB/ \
    --custom-pretrained $CUSTOM_PRETRAINED_MODEL_PATH \
    --log-wandb