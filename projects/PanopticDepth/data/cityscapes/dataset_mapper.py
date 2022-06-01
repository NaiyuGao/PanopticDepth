# Copyright (c) Facebook, Inc. and its affiliates.
# Adapted from detectron2/data/dataset_mapper.py

import copy
import logging
import numpy as np
from typing import Callable, List, Union
import torch
import pycocotools
from panopticapi.utils import rgb2id

from detectron2.config import configurable
from detectron2.data import MetadataCatalog
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.structures import BoxMode

__all__ = ["CityscapesPanopticDatasetMapper"]


class CityscapesPanopticDatasetMapper:
  """
  A callable which takes a dataset dict in Detectron2 Dataset format,
  and map it into a format used by the model.

  This is the default callable to be used to map your dataset dict into training data.
  You may need to follow it to implement your own one for customized logic,
  such as a different way to read or transform images.
  See :doc:`/tutorials/data_loading` for details.

  The callable currently does the following:

  1. Read the image from "file_name"
  2. Applies cropping/geometric transforms to the image and annotations
  3. Prepare data and annotations to Tensor and :class:`Instances`
  """

  @configurable
  def __init__(
      self,
      cfg,
      is_train: bool,
      *,
      augmentations: List[Union[T.Augmentation, T.Transform]],
      image_format: str,
      use_instance_mask: bool = False,
      instance_mask_format: str = "polygon",
      recompute_boxes: bool = False,
  ):
    """
    NOTE: this interface is experimental.

    Args:
        cfg: config dict
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
    self.cfg = cfg
    dataset_names = self.cfg.DATASETS.TRAIN
    self.meta = MetadataCatalog.get(dataset_names[0])
    self.is_train = is_train
    self.augmentations = T.AugmentationList(augmentations)
    self.image_format = image_format
    self.use_instance_mask = use_instance_mask
    self.instance_mask_format = instance_mask_format
    self.recompute_boxes = recompute_boxes
    # fmt: on
    logger = logging.getLogger(__name__)
    mode = "training" if is_train else "inference"
    logger.info(f"[DatasetMapper] Augmentations used in {mode}: {augmentations}")

  @classmethod
  def from_config(cls, cfg, is_train: bool = True):
    augs = utils.build_augmentation(cfg, is_train)
    if cfg.INPUT.CROP.ENABLED and is_train:
      #augs.insert(1, T.RandomCrop(cfg.INPUT.CROP.TYPE, cfg.INPUT.CROP.SIZE))
      augs.append(T.RandomCrop(cfg.INPUT.CROP.TYPE, cfg.INPUT.CROP.SIZE))
      recompute_boxes = cfg.MODEL.MASK_ON
    else:
      recompute_boxes = False
    if cfg.INPUT.COLOR_AUG and is_train:
      from detectron2.projects.point_rend import ColorAugSSDTransform
      augs.append(ColorAugSSDTransform(img_format=cfg.INPUT.FORMAT))


    ret = {
      "cfg": cfg,
      "is_train": is_train,
      "augmentations": augs,
      "image_format": cfg.INPUT.FORMAT,
      "use_instance_mask": cfg.MODEL.MASK_ON,
      "instance_mask_format": cfg.INPUT.MASK_FORMAT,
      "recompute_boxes": recompute_boxes,
    }

    return ret

  def __call__(self, dataset_dict):
    """
    Args:
        dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

    Returns:
        dict: a format that builtin models in detectron2 accept
    """
    dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
    # USER: Write your own image loading if it's not from a file
    image = utils.read_image(dataset_dict["file_name"], format=self.image_format)
    utils.check_image_size(dataset_dict, image)

    things_classes = list(self.meta.thing_dataset_id_to_contiguous_id.values())
    stuff_classes = np.array(list(self.meta.stuff_dataset_id_to_contiguous_id.values()))

    # USER: Remove if you don't do semantic/panoptic segmentation.
    if "pan_seg_file_name" in dataset_dict:
      pan_seg_gt = utils.read_image(dataset_dict.pop("pan_seg_file_name"))
      pan_seg_gt = pan_seg_gt[:, :, 0] + 256 * pan_seg_gt[:, :, 1] + 256 * 256 * pan_seg_gt[:, :, 2]
    else:
      raise NotImplementedError("Currently only possible if pan seg GT image file name is given")
      # pan_seg_gt = None

    # Create annotations in desired instance segmentation format
    annotations = list()
    for segment in dataset_dict['segments_info']:
      if segment['category_id'] in things_classes:
        annotation = dict()
        annotation['bbox'] = segment['bbox']
        annotation['bbox_mode'] = BoxMode.XYWH_ABS
        annotation['category_id'] = self.meta.contiguous_id_to_thing_train_id[segment['category_id']]
        mask = (pan_seg_gt == segment['id']).astype(np.uint8)
        annotation['segmentation'] = pycocotools.mask.encode(np.asarray(mask, order="F"))
        annotation['iscrowd'] = segment['iscrowd']
        annotations.append(annotation)

    if len(annotations) > 0:
      dataset_dict['annotations'] = annotations

    # USER: Remove if you don't do semantic/panoptic segmentation.
    if "sem_seg_file_name" in dataset_dict:
      sem_seg_gt_tmp = utils.read_image(dataset_dict.pop("sem_seg_file_name"), "L").squeeze(2)
    else:
      raise NotImplementedError("Currently only possible if sem seg GT image file name is given")
      # sem_seg_gt = None

    # For Cityscapes, the contiguous ids for stuff are equal to the stuff train ids. Change for other datasets.
    if self.cfg.MODEL.POSITION_HEAD.STUFF.ALL_CLASSES:
      sem_seg_gt = np.where(np.isin(sem_seg_gt_tmp, stuff_classes), sem_seg_gt_tmp, self.meta.ignore_label)
    else:
      if self.cfg.MODEL.POSITION_HEAD.STUFF.WITH_THING:
        sem_seg_gt = np.where(np.isin(sem_seg_gt_tmp, stuff_classes), sem_seg_gt_tmp + 1, self.meta.ignore_label)
        # Set things class pixels to 0
        sem_seg_gt = np.where(np.isin(sem_seg_gt_tmp, np.array(things_classes)), 0, sem_seg_gt)
      else:
        sem_seg_gt = np.where(np.isin(sem_seg_gt_tmp, stuff_classes), sem_seg_gt_tmp + 1, self.meta.ignore_label)

    aug_input = T.AugInput(image, sem_seg=sem_seg_gt)
    transforms = self.augmentations(aug_input)
    image, sem_seg_gt = aug_input.image, aug_input.sem_seg

    image_shape = image.shape[:2]  # h, w
    # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
    # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
    # Therefore it's important to use torch.Tensor.
    dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
    if sem_seg_gt is not None:
      dataset_dict["sem_seg"] = torch.as_tensor(sem_seg_gt.astype("long"))

    if not self.is_train:
      # USER: Modify this if you want to keep them for some reason.
      dataset_dict.pop("annotations", None)
      dataset_dict.pop("sem_seg_file_name", None)
      return dataset_dict

    if "annotations" in dataset_dict:
      # USER: Modify this if you want to keep them for some reason.
      for anno in dataset_dict["annotations"]:
        if not self.use_instance_mask:
          anno.pop("segmentation", None)

      # USER: Implement additional transformations if you have other types of data
      annos = [
        utils.transform_instance_annotations(
          obj, transforms, image_shape
        )
        for obj in dataset_dict.pop("annotations")
        if obj.get("iscrowd", 0) == 0
      ]
      instances = utils.annotations_to_instances(
        annos, image_shape, mask_format=self.instance_mask_format
      )

      # After transforms such as cropping are applied, the bounding box may no longer
      # tightly bound the object. As an example, imagine a triangle object
      # [(0,0), (2,0), (0,2)] cropped by a box [(1,0),(2,2)] (XYXY format). The tight
      # bounding box of the cropped triangle should be [(1,0),(2,1)], which is not equal to
      # the intersection of original bounding box and the cropping box.

      if self.recompute_boxes and len(instances) > 0:
        instances.gt_boxes = instances.gt_masks.get_bounding_boxes()

      dataset_dict["instances"] = utils.filter_empty_instances(instances)

      if len(dataset_dict['instances']) == 0:
        del dataset_dict["instances"]

    return dataset_dict
