# Delving into Pre-training for Domain Transfer: A Broad Study of Pre-training for Domain Generalization and Domain Adaptation (IJCV 2026)
Jungmyung Wi, Youngkyun Jang, Dujin Lee, Myeongseok Nam, and Donghyun Kim.
#### [[Paper]](https://link.springer.com/article/10.1007/s11263-025-02590-5)

## Introduction

This is the official PyTorch implementation of the paper "Delving into Pre-training for Domain Transfer: A Broad Study of Pre-training for Domain Generalization and Domain Adaptation", published in the International Journal of Computer Vision (IJCV) 2026.

While existing domain transfer methods typically rely on outdated ResNet backbones pre-trained on ImageNet-1K, our work provides a broad study and in-depth analysis of how modern pre-training approaches impact domain transfer. We evaluate various network architectures, sizes, pre-training objectives, and datasets across Domain Generalization (DG), Unsupervised Domain Adaptation (UDA), Source Free Domain Adaptation (SFDA), and Universal Domain Adaptation (UniDA).

This repository contains the PyTorch implementation of our experiments. Our findings demonstrate that state-of-the-art pre-training (e.g., ImageNet-22K) often outperforms advanced adaptation techniques, largely because modern pre-training datasets contain a significantly higher number of classes that closely resemble those in downstream tasks.

**Bibtex**
```
@article{wi2026delving,
  title={Delving into Pre-training for Domain Transfer: A Broad Study of Pre-training for Domain Generalization and Domain Adaptation: Wi et al.},
  author={Wi, Jungmyung and Jang, Youngkyun and Lee, Dujin and Nam, Myeongseok and Kim, Donghyun},
  journal={International Journal of Computer Vision},
  volume={134},
  number={2},
  pages={50},
  year={2026},
  publisher={Springer}
}
```

# Data Download
* Please run the command below to download the pre-training and downstream task datasets!
```
bash ./download.sh
```
* Below is the dataset directory structure:
```
dataset/
├─ imagenet21k/
│  └─ train
├─ imagenet21k_resized/
│  ├─ imagenet21k_small_classes 
│  ├─ imagenet21k_train  
│  └─ imagenet21k_val
└─ da
    ├─ cub
    ├─ domainnet
    └─ office-home
```
# Library download
```
pip install -r requirements.txt
```

Additionally, if you want to use webdataset , install the below library
```
git+https://github.com/webdataset/webdataset
```


# Baseline Pretraining - Downstream finetuning and evaluation 

* **Data Prunend** To perform pre-training and fine-tuning, please run the commands below!
```
bash ./run_ddp.sh # Distributed Data Parralled (mutil-gpu) 

bash ./run_ddp.sh IN21k-PATH DATASET DA-PATH GPU_NUMBER THE_NUMBER_OF_GPU BATCHSIZE

default ex) bash ./run_ddp.sh ../dataset/imagenet21k_train/train default ../dataset/da 0,1,2,3,4,5,6,7 8 128
wds     ex) bash ./run_ddp.sh ../dataset/imagenet-w21-wds wds/ ../dataset/da 0,1,2,3,4,5,6,7 8 128
```
```
bash ./run_single.sh IN21k-PATH DATASET DA-PATH GPU_NUMBER BATCHSIZE # (sigle-gpu)

default ex) bash ./run_single.sh ../dataset/imagenet21k_train/train default ../dataset/da 0 1024
wds     ex) bash ./run_single.sh ../dataset/imagenet-w21-wds wds/ ../dataset/da 0 1024
```

* If you want to perform a grid search to find the optimal pre-training settings, please run the commands below!<br/>
(**Grid search pre-training is conducted on the original, unpruned ImageNet21k winter dataset.**)
```
# Hyperparameter tuning with the unpruned dataset (multi-gpu) (multi-gpu) 

bash ./run_hyperpram_ddp.sh IN21k-PATH DATASET DA-PATH LR WARMUP_LR GPU_NUMBER THE_NUMBER_OF_GPU BATCHSIZE

default ex) bash ./run_hyperparam_ddp.sh ../dataset/imagenet21k_train/train default ../dataset/da 1e-3 1e-5 0,1,2,3,4,5,6,7 8 128
wds     ex) bash ./run_hyperparam_ddp.sh ../dataset/imagenet-w21-wds wds/ ../dataset/da 1e-3 1e-5  0,1,2,3,4,5,6,7 8 128
```
```
# Hyperparameter tuning with the unpruned dataset (single-gpu)

bash ./run_hyperparam_single.sh IN21k-PATH DATASET DA-PATH LR WARMUP_LR GPU_NUMBER BATCHSIZE

default ex) bash ./run_hyperparam_single.sh ../dataset/imagenet21k_train/train default ../dataset/da 1e-3 1e-5 0 1024
wds     ex) bash ./run_hyperparam_single.sh ../dataset/imagenet-w21-wds wds/ ../dataset/da 1e-3 1e-5 0 1024
```

# 실행 전 유의사항

* Running run.sh will automatically proceed with pre-training and fine-tuning. The hyperparameters provided are the exact ones I used in my actual experiments. Therefore, you do not need to change them.

* Note that if you download the datasets via 'bash ./download.sh', you can run run.sh directly without setting up separate dataset paths.

* If you wish to change the assigned GPU numbers, you can modify them directly inside the run.sh file. (I have left comments in the script for this).