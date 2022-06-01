# Copyright (c) Facebook, Inc. and its affiliates.
import copy
import logging
import numpy as np
from PIL import Image
from typing import List, Optional, Union
import torch

from detectron2.config import configurable

from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T

from detectron2.structures import BoxMode
import pycocotools
from ..utils import dense_ind
"""
This file contains the default mapping that's applied to "dataset dicts".
"""

__all__ = ["CityscapesDPSDatasetMapper, FixedSizeCenterCrop"]

from typing import Tuple
from fvcore.transforms.transform import (
    CropTransform,
    PadTransform,
    TransformList,
)
from detectron2.data.transforms.augmentation import Augmentation

class FixedSizeCenterCrop(Augmentation):
    """
    If `crop_size` is smaller than the input image size, then it uses a center crop of
    the crop size. If `crop_size` is larger than the input image size, then it pads
    the around of the image to the crop size.
    """

    def __init__(self, crop_size: Tuple[int], pad_value: float = 128.0, with_pad=True):
        """
        Args:
            crop_size: target image (height, width).
            pad_value: the padding value.
        """
        super().__init__()
        self._init(locals())

    def get_transform(self, image: np.ndarray) -> TransformList:
        # Compute the image scale and scaled size.
        input_size = image.shape[:2]
        output_size = self.crop_size

        # Add random crop if the image is scaled up.
        max_offset = np.subtract(input_size, output_size)
        max_offset = np.maximum(max_offset, 0)
        offset = np.multiply(max_offset, 0.5)#np.random.uniform(0.0, 1.0))
        offset = np.round(offset).astype(int)
        crop_transform = CropTransform(
            offset[1], offset[0], output_size[1], output_size[0], input_size[1], input_size[0]
        )
        if not self.with_pad:
            return TransformList([crop_transform, ])

        # Add padding if the image is scaled down.
        pad_size = np.subtract(output_size, input_size)
        pad_size = np.maximum(pad_size, 0)
        pad_size_0 = pad_size // 2
        pad_size_1 = pad_size - pad_size_0
        original_size = np.minimum(input_size, output_size)
        pad_transform = PadTransform(
            pad_size_0[1], pad_size_0[0], pad_size_1[1], pad_size_1[0], original_size[1], original_size[0], self.pad_value
        )

        return TransformList([crop_transform, pad_transform])

class CityscapesDPSDatasetMapper:
    """
    1. Read the image from "file_name"
    2. Applies cropping/geometric transforms to the image and annotations
    3. Prepare data and annotations to Tensor and :class:`Instances`
    """

    @configurable
    def __init__(
        self,
        is_train: bool,
        *,
        augmentations: List[Union[T.Augmentation, T.Transform]],
        image_format: str,
        use_instance_mask: bool = False,
        instance_mask_format: str = "bitmask",
        recompute_boxes: bool = False,
        depth_bound: bool = True
    ):
        """
        NOTE: this interface is experimental.

        Args:
            is_train: whether it's used in training or inference
            augmentations: a list of augmentations or deterministic transforms to apply
            image_format: an image format supported by :func:`detection_utils.read_image`.
            use_instance_mask: whether to process instance segmentation annotations, if available
            instance_mask_format: one of "polygon" or "bitmask". Process instance segmentation
                masks into this format.
            recompute_boxes: whether to overwrite bounding box annotations
                by computing tight bounding boxes from instance mask annotations.
        """
        if recompute_boxes:
            assert use_instance_mask, "recompute_boxes requires instance masks"
        # fmt: off
        self.is_train               = is_train
        self.augmentations          = T.AugmentationList(augmentations)
        self.image_format           = image_format
        self.use_instance_mask      = use_instance_mask
        self.instance_mask_format   = instance_mask_format
        self.recompute_boxes        = recompute_boxes
        self.depth_bound            = depth_bound
        assert self.instance_mask_format == "bitmask"
        assert self.use_instance_mask
        # fmt: on
        logger = logging.getLogger(__name__)
        mode = "training" if is_train else "inference"
        logger.info(f"[DatasetMapper] Augmentations used in {mode}: {augmentations}")

    @classmethod
    def from_config(cls, cfg, is_train: bool = True):
        augs = []
        if cfg.INPUT.COLOR_AUG and is_train:
            from detectron2.projects.point_rend import ColorAugSSDTransform
            augs.append(ColorAugSSDTransform(img_format=cfg.INPUT.FORMAT))
        if is_train and cfg.INPUT.RANDOM_FLIP != "none":
            augs.append(
                T.RandomFlip(
                    horizontal=cfg.INPUT.RANDOM_FLIP == "horizontal",
                    vertical=cfg.INPUT.RANDOM_FLIP == "vertical",
                )
            )
        recompute_boxes = False
        if cfg.INPUT.ROTATE_AUG > 0. and is_train:
            augs.append(T.RandomRotation(angle=[-cfg.INPUT.ROTATE_AUG, cfg.INPUT.ROTATE_AUG],
                                       expand=False, center=None, sample_style="range", interp=None))
        if cfg.INPUT.CROP.ENABLED and is_train:
            assert cfg.INPUT.CROP.TYPE == "absolute", cfg.INPUT.CROP
            crop_size = cfg.INPUT.CROP.SIZE
            if cfg.INPUT.CROP.RESCALE[0] > 0:
                rescale_range = cfg.INPUT.CROP.RESCALE
                assert rescale_range[0] <= rescale_range[1], rescale_range
                augs.append(T.ResizeScale(min_scale=rescale_range[0], max_scale=rescale_range[1], target_height=crop_size[0], target_width=crop_size[1]))
            augs.append(FixedSizeCenterCrop(crop_size=crop_size, pad_value=0, with_pad=cfg.INPUT.CROP.WITH_PAD))
        ret = {
            "is_train": is_train,
            "augmentations": augs,
            "image_format": cfg.INPUT.FORMAT,
            "use_instance_mask": cfg.MODEL.MASK_ON,
            "instance_mask_format": cfg.INPUT.MASK_FORMAT,
            "recompute_boxes": recompute_boxes,
            "depth_bound": cfg.INPUT.DEPTH_BOUND,
        }
        return ret

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
            dataset_dict keys:
                file_name
                image_id
                vps_label_file_name
                depth_label_file_name
                next_frame
        """
        dataset_dict = copy.deepcopy(dataset_dict)
    
        image = utils.read_image(dataset_dict["file_name"], format=self.image_format)
        utils.check_image_size(dataset_dict, image)

        if "vps_label_file_name" in dataset_dict:
            vps_seg_gt = np.asarray(Image.open(dataset_dict.pop("vps_label_file_name")), order="F")
            sem_seg_gt = vps_seg_gt // 1000
            sem_seg_gt[sem_seg_gt > 18] = 255
            sem_seg_gt = sem_seg_gt.astype(np.uint8)
            vps_seg_gt, convert_dict = dense_ind(vps_seg_gt,stay_shape=True,shuffle=False,return_convert=True)
            convert_dict_reverse = {v:k for k,v in convert_dict.items()}
        else:
            vps_seg_gt, sem_seg_gt = None, None

        if "depth_label_file_name" in dataset_dict:
            depth_gt = np.array(Image.open(dataset_dict.pop("depth_label_file_name")))
            depth_gt_1 = (depth_gt // 256).astype(np.uint8)
            depth_gt_2 = (depth_gt  % 256).astype(np.uint8)
        else:
            depth_gt = None

        aug_input = T.AugInput(image, sem_seg=sem_seg_gt)
        transforms = self.augmentations(aug_input)
        image, sem_seg_gt = aug_input.image, aug_input.sem_seg

        template = np.ones_like(vps_seg_gt)
        template = transforms.apply_segmentation(template)
        image[template==0] = 128
        sem_seg_gt[template==0] = 255
        if vps_seg_gt is not None:
            vps_seg_gt = transforms.apply_segmentation(vps_seg_gt)
            vps_seg_gt[template==0] = convert_dict_reverse[32000]
        if depth_gt is not None:
            depth_gt_1 = transforms.apply_segmentation(depth_gt_1)
            depth_gt_2 = transforms.apply_segmentation(depth_gt_2)
            depth_gt = depth_gt_1.astype(np.float64) * 256 + depth_gt_2.astype(np.float64)
            depth_gt = depth_gt / 256.
            del depth_gt_1, depth_gt_2
            depth_gt[template==0] = 0
            for transform in transforms:
                if isinstance(transform, T.ResizeTransform):
                    aug_scale = (transform.w / transform.new_w + transform.h / transform.new_h) / 2
                    if self.depth_bound:
                        depth_gt = np.clip(depth_gt * aug_scale, depth_gt.min(), depth_gt.max())
                    else:
                        depth_gt = depth_gt * aug_scale

        image_shape = image.shape[:2]  # h, w
        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
        if sem_seg_gt is not None:
            dataset_dict["sem_seg"] = torch.as_tensor(sem_seg_gt.astype("long"))
        if depth_gt is not None:
            dataset_dict["depth"] = torch.as_tensor(np.ascontiguousarray(depth_gt).astype("float32"))

        if self.is_train:
            annos = _cityscapes_dps_files_to_dict(vps_seg_gt, convert_dict)

            instances = utils.annotations_to_instances(
                annos, image_shape, mask_format=self.instance_mask_format
            )
            dataset_dict["instances"] = utils.filter_empty_instances(instances)
        else:
            dataset_dict.pop("vps_label_file_name", None)
            dataset_dict.pop("depth_label_file_name", None)
            dataset_dict.pop("next_frame", None)

        return dataset_dict

def _cityscapes_dps_files_to_dict(inst_image, convert_dict):
    trainId_to_contiguous_id = {11: 0, 12: 1, 13: 2, 14: 3, 15: 4, 16: 5, 17: 6, 18: 7}
    """
    Parse cityscapes dps annotation files to a instance segmentation dataset dict.
    Args:
    Returns:
        A dict in Detectron2 Dataset format.
    """
    #from cityscapesscripts.helpers.labels import trainId2label, name2label

    annos = []
    flattened_ids = np.unique(inst_image)

    for instance_vpsId_dense in flattened_ids:
        instance_vpsId = convert_dict[instance_vpsId_dense]
        label_trainId = instance_vpsId // 1000

        if label_trainId not in trainId_to_contiguous_id:
            continue
        iscrowd = instance_vpsId < 1000
        if iscrowd:
            continue

        anno = {}
        anno["category_id"] = trainId_to_contiguous_id[label_trainId]
        anno["vps_id"] = int(instance_vpsId)

        mask = np.asarray(inst_image == instance_vpsId_dense, dtype=np.uint8, order="F")

        inds = np.nonzero(mask)
        ymin, ymax = inds[0].min(), inds[0].max()
        xmin, xmax = inds[1].min(), inds[1].max()
        anno["bbox"] = (int(xmin), int(ymin), int(xmax), int(ymax))
        if xmax <= xmin or ymax <= ymin:
            continue
        anno["bbox_mode"] = BoxMode.XYXY_ABS
        anno["segmentation"] = pycocotools.mask.encode(mask)
        annos.append(anno)

    return annos
