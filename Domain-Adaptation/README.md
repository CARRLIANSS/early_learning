# Domain-Adaptation
A toolkit based on PyTorch for domain adaptation (DA) of deep neural networks

## 0. Implemented Algorithms

As initial version, we support the following algoirthms.

- DAN (Deep Adaptation Network)[1]
- DeepCoral[2]
- BNM (Batch Nuclear-norm Maximization)[3]

## 1. Data Prepare

Office31 download: 链接：https://pan.baidu.com/s/1Ndez_lhIx02L9Lpcc8H3pA  提取码：ltf0

put it in **data** directory

## 2. Model

Backbone has two options: Resnet or Alexnet

DA loss has three options: Coral, Dan/Mmd, Bnm

## 3. Train

```
# train by single gpu
python train_single_gpu.py
```

```
# train by multi gpu
CUDA_VISIBLE_DEVICES=0,3 python -m torch.distributed.launch --nproc_per_node=2 --use_env train_multi_gpu_using_launch.py
```

## 4. Config

Don't forget to modify config.py, you should modify it according to your environment and algorithm what you need.

## 5. Contribution

The repository has re-integrated the code and provides single-gpu and multi-gpu training scripts.

## 6. Acknowledgement

We borrow code from public projects. We mainly borrow code from [DeepDA](https://github.com/jindongwang/transferlearning/tree/master/code/DeepDA).

## 7. Reference

[1] Long, Mingsheng, et al. "Learning transferable features with deep adaptation networks." International conference on machine learning. PMLR, 2015.

[2] Sun, Baochen, and Kate Saenko. "Deep coral: Correlation alignment for deep domain adaptation." European conference on computer vision. Springer, Cham, 2016.

[3] Cui, Shuhao, et al. Towards discriminability and diversity: Batch nuclear-norm maximization under label insufficient situations. CVPR 2020.
