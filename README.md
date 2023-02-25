# Towards High-Quality Real-Time Video Denoising with Pseudo Temporal Fusion Network

# Overview 
This source code provides a PyTorch implementation of the Pseudo Temporal Fusion Network (PTFN) video denoising algorithm.


# Datasets 
The same dataset as [FastDVDNet](https://github.com/m-tassano/fastdvdnet) is used for experiments.
## Trainset 
The 2017 DAVIS train dataset was used for training. You can download `DAVIS-train.tar` from [here](). 
Put extracted `DAVIS-train.tar` to `./datasets`.

## Testset
Two testsets are used in the paper: Set8 and the 2017 DAVIS testset.
You can download `DAVIS-test.tar` from [here](). 
You can download `Set8.tar` from [here]().
Put extracted `DAVIS-test.tar` and `Set8.tar` to `./datasets`.

# Dependencies and Installation
## Environment
Create a new conda environment 
``` console
conda create -n ptfn python=3.10
conda activate ptfn
```
Install dependencies
``` console
conda install pytorch-gpu==1.12.1 torchvision==0.13.0 cudatoolkit=11.6 -c conda-forge
conda install pandas easydict tqdm -c conda-forge
pip install opencv-python
```

# Training
You can change the configuration files in `config/*.json` as you like.
## Non-blind Video Denoising
``` console
# Training on Single GPU
CUDA_VISIBLE_DEVICES=0 python -m train_codes.train -c config/config_ptfn.json
# Training on Multi GPU
CUDA_VISIBLE_DEVICES=0,1 python -m train_codes.train_dp -c config/config_ptfn.json
```
## Blind Video Denoising
``` console
# Training on Single GPU
CUDA_VISIBLE_DEVICES=0 python -m train_codes.train_blind -c config/config_ptfn.json
# Training on Multi GPU
CUDA_VISIBLE_DEVICES=0,1 python -m train_codes.train_blind_dp -c config/config_ptfn.json
```
## FineTuning for Ligher Version (PTFN Half or PTFN-L Half)
``` console
# Training on Single GPU
CUDA_VISIBLE_DEVICES=0 python -m train_codes.finetune -c config/config_finetune.json
# Training on Multi GPU
CUDA_VISIBLE_DEVICES=0,1 python -m train_codes.finetune_dp -c config/config_finetune.json
```

# Test
You may download pretrained checkpoints from [here]()
Then, run the commands below.
``` console
GPU_ID=0
CONFIG="path/to/config/files"
noise_levels="50"
CUDA_VISIBLE_DEVICES=$GPU_ID python -m eval_codes.generate_images_davis -nl $noise_levels -c $CONFIG --not_generae_inter_img
python -m eval_codes.evaluation -nl $noise_levels -c $CONFIG
#CUDA_VISIBLE_DEVICES=$GPU_ID python -m eval_codes.generate_images_set8 -nl $noise_levels -c $CONFIG --not_generae_inter_img
#python -m eval_codes.evaluation -nl $noise_levels --set8 -c $CONFIG
```

# Results
