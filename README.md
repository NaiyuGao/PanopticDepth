# PanopticDepth: A Unified Framework for Depth-aware Panoptic Segmentatio

## Introduction
In this repository, we present a unified framework for depth-aware panoptic segmentation (DPS) which aims to reconstruct 3D scene with instance-level semantics from one single image. Prior work addresses this problem by simply adding a dense depth regression head to panoptic segmentation (PS) networks, resulting in two independent task branches. This neglects the mutually-beneficial relations between these two tasks, fails to exploit handy instance-level semantic cues to boost depth accuracy, and thereby produces sub-optimal depth maps. To overcome these limits, we propose a unified framework for the DPS task by applying a dynamic convolution technique to both the PS and depth prediction tasks. Specifically, instead of predicting depth for all pixels at a time, we predict depth for each thing/stuff instance with a generated instance-specific kernel, and apply the same manner to producing instance masks. Moreover, leveraging the instance-wise depth estimation scheme, we add additional instance-level depth cues to assist with supervising the depth learning via a new depth loss. Extensive experiments on Cityscapes-DPS and SemKITTI-DPS show the effectiveness and promise of our method. We hope our unified solution to DPS can lead a new paradigm in this area.

<img src="https://github.com/NaiyuGao/PanopticDepth/blob/main/PanopticDepth.jpg" height = "300" alt="" align=center />

Code and model will be available soon.
