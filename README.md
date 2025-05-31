# CNN-TA-Distillation


## 0. Project Overview

## 1. Environment Setup

## 2. Replication of TAKD 


## 3. Train and Evaluate the Model
1. First step is to train the biggest Teacher model from the CIFAR dataset with the following command. At this step, the original teacher model will be a student model of the ResNet.
```
python3 train.py --epochs {NUM_EPOCHS} --student resnet{S_SIZE} --cuda {NUM_GPUS} --dataset cifar100
```

2. Distill knowledege from the Teacher to the Students with CIFAR dataset, by editting the following user parameters. You can distill knowledge by chaning students model in sequences.
```
python3 train.py --epochs {NUM_EPOCHS} --teacher resnet{T_SIZE} --teacher-checkpoint {T_CKPT_PATH} --student resnet{S_SIZE} --cuda {NUM_GPUS} --dataset cifar100
```
- NUM_EPOCHS: Number of epochs for training
- T_CKPT_PATH: Path for teacher model
- T_SIZE: Size of teacher model
- S_SIZE: Size of student model
- NUM_GPUS: Number of GPUs to use
3. Check whether the checkpoints is created well, and the evaluation results. 
In this project, we did 16 experiment in total. The biggest size of the teacher model was 110, and the other candidates of the student models were 56, 32, 20, and 8. Since we have 4 possible sizes of student model, there should be 2 to the power of 4 (=16) experiments to test all the cases.
## 4. Results

## 5. Analysis