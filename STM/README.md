# STM
Video Object Segmentation using Space-Time Memory Networks

## Requirement
- torch
- tqdm
- yacs
- progress
- imgaug
- pyyaml
- pillow
- opencv-python
- tensorboard
- numpy

## GPU support
- GPU Memory >= 30GB
- CUDA == 10.1

## Dataset

### ROOT Organization
数据集根目录放置Davis17、YoutubeVOS...等 </br>
确定数据根目录后，需要更改`libs/dataset/data.py`变量`DATA_ROOT = "path/to/root"`
```
${ROOT}--
        |--${DATASET1}
        |--${DATASET1}
        ...
```

### DAVIS17 Organization
Davis17训练集（train）被划分为train、val、test，具体配置`db_info.yaml`
```
$DAVIS17(DAVIS16)
    |----train
      |----Annotations
      |----JPEGImages
    |----db_info.yaml
```

### YoutubeVOS Organization
YoutubeVOS只使用训练集（train）
```
$YoutubeVOS
    |----train
      |----Annotations
      |----JPEGImages
```

## Release
`STM-Cycle`基础上`finetune`

## Running
```
python train.py --cfg config.yaml
```

## Results

### Global result
每个epoch结束输出

### Per-sequence results
最后一个epoch结束输出

### Output
每个epoch输出


## Acknowledgement
该仓库代码基于 [official STM-Cycle repository](https://github.com/lyxok1/STM-Training)

## Reference
仓库基于以下工作
```
@InProceedings{Oh_2019_ICCV,
author = {Oh, Seoung Wug and Lee, Joon-Young and Xu, Ning and Kim, Seon Joo},
title = {Video Object Segmentation Using Space-Time Memory Networks},
booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
month = {October},
year = {2019}
}

@InProceedings{Li_2020_NeurIPS,
author = {Li, Yuxi and Xu, Ning and Peng Jinlong and John See and Lin Weiyao},
title = {Delving into the Cyclic Mechanism in Semi-supervised Video Object Segmentation},
booktitle = {Neural Information Processing System (NeurIPS)},
year = {2020}
}
```
