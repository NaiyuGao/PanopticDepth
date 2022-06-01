# PanopticDepth: A Unified Framework for Depth-aware Panoptic Segmentatio

[[`arXiv`](https://arxiv.org/)] [[`BibTeX`](#CitingPanopticDepth)]

## Introduction
In this repository, we present a unified framework for depth-aware panoptic segmentation (DPS) which aims to reconstruct 3D scene with instance-level semantics from one single image. Prior works address this problem by simply adding a dense depth regression head to panoptic segmentation (PS) networks, resulting in two independent task branches. This neglects the mutually-beneficial relations between these two tasks, fails to exploit handy instance-level semantic cues to boost depth accuracy, and thereby produces sub-optimal depth maps. To overcome these limits, we propose a unified framework for the DPS task by applying a dynamic convolution technique to both the PS and depth prediction tasks. Specifically, instead of predicting depth for all pixels at a time, we predict depth for each thing/stuff instance with a generated instance-specific kernel, and apply the same manner to producing instance masks. Moreover, leveraging the instance-wise depth estimation scheme, we add additional instance-level depth cues to assist with supervising the depth learning via a new depth loss. Extensive experiments on Cityscapes-DPS and SemKITTI-DPS show the effectiveness and promise of our method. We hope our unified solution to DPS can lead a new paradigm in this area.

<img src="https://github.com/NaiyuGao/PanopticDepth/blob/main/overview.jpg" height = "300" alt="" align=center />

## Installation
This project is based on [Detectron2](https://github.com/facebookresearch/detectron2), which can be constructed as follows.
* Install Detectron2 following [the instructions](https://detectron2.readthedocs.io/tutorials/install.html).
* Setup the Cityscapes dataset following [this structure](https://github.com/facebookresearch/detectron2/blob/master/datasets/README.md).
* Setup the Cityscapes-DPS dataset following [this structure](https://github.com/joe-siyuan-qiao/ViP-DeepLab/blob/master/cityscapes-dvps/README.md).
* Copy this project to `/path/to/detectron2/`
* Convert the Cityscapes-DPS dataset format with [this script]().
## Training
```bash
cd ./projects/PanopticDepth/
```

```bash
python train.py --config-file configs/cityscapes/PanopticFCN-R50-cityscapes.yaml --num-gpus 8 OUTPUT_DIR "./output/ps"
```

```bash
python train.py --config-file configs/cityscapes/PanopticFCN-R50-cityscapes-FullScaleFinetune.yaml --num-gpus 8 MODEL.WEIGHTS  "./output/ps/model_final.pth" OUTPUT_DIR "./output/ps_fsf"
```

```bash
python train.py --config-file configs/cityscapes_dps/PanopticDepth-R50-cityscapes-dps.yaml --num-gpus 8 MODEL.WEIGHTS "./output/ps_fsf/model_final.pth"  OUTPUT_DIR "./output/dps"
```

## <a name="CitingPanopticDepth"></a>Citing PanopticDepth

Consider cite PanopticDepth in your publications if it helps your research.

```
@inproceedings{gao2022panopticdepth,
  title={PanopticDepth: A Unified Framework for Depth-aware Panoptic Segmentation},
  author={Naiyu Gao, Fei He, Jian Jia, Yanhu Shan, Haoyang Zhang, Xin Zhao, and Kaiqi Huang},
  booktitle={IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2022}
}
```
