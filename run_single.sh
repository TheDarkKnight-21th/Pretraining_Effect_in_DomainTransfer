# Please refer to the pretrain arguments comments and change the arguments "if necessary".

# pretrain arguments (These are the hyperparameters I experimented with.)

MODEL="convnext_tiny.fb_in1k" # You must always append .fb_in1k. This is because we pretrain by initializing with IN1k weights, not from scratch. (For swin, it's .ms_in1k. Please note that the company name differs depending on the model.)
DATA_DIR=$1 #"./dataset/imagenet21k_train/train" #../dataset/imagenet21k/train"  # Set dataset path
BATCH_SIZE=$5 # The total batch size is "1024" divided by the number of GPUs. EX ) <= 1024 / Number of GPUs (4)
GPU=1 # Number of pretrain GPUs
GPU_NUMBER=$4
GRAD_ACCUM_STEPS=1 # The target total batch size is 1024. If your GPU VRAM is insufficient, since < Total Batch Size = Number of GPUs x Batch per GPU x grad_accum_steps >, please adjust the batch size and grad_accum_steps appropriately to match the total batch size.
OPTIMIZER="adamw"  # We use adamw as the optimizer for all pretraining.
LEARNING_RATE=1e-4
WARMUP_EPOCHS=2
WARMUP_LR=1e-6
SCHEDULER="cosine"
EPOCHS=24
EXPERIMENT="UTL-21k"
WEIGHT_DECAY=5e-2
MIXUP=0.8
CUTMIX=1.0
LOG_INTERVAL=1
SCHED_ON_UPDATES="--sched-on-updates"
NUM_CLASSES=19167
PRETRAINED="--pretrained"  
DATASET=${2:-"default"}


cd pytorch-image-models
# class map => The list of pruned class maps. It loads all files in that list.
CLASS_MAP=($(find ./pruning_book/class_map/feature -type f -name "*.txt" | sort -V)) # in dev

# If you want to use all of far, near, threshold, and random, please remove "near" from the path in the command above.

# Class map (The folder includes office-home, domainnet, and cub.)
# ./pytorch-image-models/pruning_book/class_map/feature
# ├─ appendix  # IN21k classes pruned in order of domain proximity to the prototype of each class in the target dataset
# │  ├─ IN21k_cub_0.1_near_pruning.txt
# │  ├─ IN21k_cub_0.5_near_pruning.txt
# │  ├─ IN21k_cub_1_near_pruning.txt
# │  └─ ...
# ├─ far # IN21k classes pruned in order of proximity to the prototype of each class in the target dataset
# │  ├─ IN21k_cub_0.1_far.txt
# │  ├─ IN21k_cub_0.5_far.txt
# │  ├─ IN21k_cub_1_far.txt
# │  └─ ....
# ├─ near  # IN21k classes pruned in order of proximity to the prototype of each class in the target dataset
# │  ├─ IN21k_cub_0.1_near.txt
# │  ├─ IN21k_cub_0.5_near.txt
# │  ├─ IN21k_cub_1_near.txt
# │  └─ ...
# ├─ random # Randomly pruned IN21k classes (based on near classes)
# │  ├─ IN21k_cub_0.1_far_random.txt
# │  ├─ IN21k_cub_0.5_far_random.txt
# │  ├─ IN21k_cub_1_far_random.txt
# │  └─ ... 
# └─ threshold # IN21k classes pruned based on a specific cosine similarity threshold
#    ├─ IN21k_cub_0.65_threshold.txt
#    ├─ IN21k_cub_0.7_threshold.txt
#    ├─ IN21k_cub_0.75_threshold.txt
#    └─ ...


# GPU number => Please change if necessary.
GPU_NUMBER1=$GPU_NUMBER
length=${#CLASS_MAP[@]}


for ((i=0; i<${length}; i++)); do
  # Execute pretrain
  cd pytorch-image-models
  CUDA_VISIBLE_DEVICES=$GPU_NUMBER1 python train.py \
      --model $MODEL \
      --data-dir $DATA_DIR \
      --dataset $DATASET \
      --class-map ${CLASS_MAP[$i]} \
      --batch-size $BATCH_SIZE \
      --grad-accum-steps $GRAD_ACCUM_STEPS \
      --opt $OPTIMIZER \
      --lr $LEARNING_RATE \
      --warmup-epochs $WARMUP_EPOCHS \
      --warmup-lr $WARMUP_LR \
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

  if [[ ${CLASS_MAP[i]} == *".txt"* ]]; then
    CUSTOM_PRETRAINED_MODEL_PATH="../pytorch-image-models/output/train/${EXPERIMENT}/${MODEL}_${LEARNING_RATE}_${WARMUP_LR}_${WARMUP_EPOCHS}_${BATCH_SIZE}x${GPU}_${SCHEDULER}_single_$(basename ${CLASS_MAP[$i]} .txt)/checkpoint-$((${EPOCHS}-1)).pth.tar"
  else
    CUSTOM_PRETRAINED_MODEL_PATH="../pytorch-image-models/output/train/${EXPERIMENT}/${MODEL}_${LEARNING_RATE}_${WARMUP_LR}_${WARMUP_EPOCHS}_${BATCH_SIZE}x${GPU}_${SCHEDULER}_single/checkpoint-$((${EPOCHS}-1)).pth.tar"
  fi
  
  # Execute downstream task

  DA_PATH=$3

  OFFICEHOME_PATH="${DA_PATH}/office-home/"
  DOMIANNET_PATH="${DA_PATH}/dommainnet/"
  CUB_PATH="${DA_PATH}/cub/"

  # GPU number => Please change if necessary.
  GPU_NUMBER2=$GPU_NUMBER
  if [[ $CUSTOM_PRETRAINED_MODEL_PATH == *"office-home"* ]]; then

    CUDA_VISIBLE_DEVICES=$GPU_NUMBER2 python main.py $OFFICEHOME_PATH \
        -d OfficeHome \
        -s Rw \
        -t Ar Cl Pr \
        -a $MODEL \
        --seed 0 \
        --log baseline/ \
        --custom-pretrained $CUSTOM_PRETRAINED_MODEL_PATH \
        --log-wandb

  elif [[ $CUSTOM_PRETRAINED_MODEL_PATH == *"domainnet"* ]]; then

    CUDA_VISIBLE_DEVICES=$GPU_NUMBER2 python main.py $DOMIANNET_PATH \
        -d DomainNet \
        -s r \
        -t c i p q s \
        -a $MODEL  \
        --seed 0 \
        --log baseline_domainnet/ \
        --custom-pretrained $CUSTOM_PRETRAINED_MODEL_PATH \
        --log-wandb

  elif [[ $CUSTOM_PRETRAINED_MODEL_PATH == *"cub"* ]]; then
    
    CUDA_VISIBLE_DEVICES=$GPU_NUMBER2 python main.py $CUB_PATH \
        -d CUB \
        -s Rw \
        -t Pr \
        -a $MODEL \
        --seed 0 \
        --log baseline_CUB/ \
        --custom-pretrained $CUSTOM_PRETRAINED_MODEL_PATH \
        --log-wandb
  fi
    
  cd ..

done