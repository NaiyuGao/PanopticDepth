# Copyright (c) Facebook, Inc. and its affiliates.
import json
import logging
import os

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets.builtin_meta import CITYSCAPES_CATEGORIES
from detectron2.utils.file_io import PathManager

import functools
import multiprocessing as mp
from detectron2.utils.comm import get_world_size
from detectron2.structures import BoxMode

"""
This file contains functions to register the Cityscapes panoptic dataset to the DatasetCatalog.
"""


logger = logging.getLogger(__name__)

def load_cityscapes_dps(image_dir, gt_dir, gt_json, meta):
    assert os.path.exists(
        gt_json
    ), gt_json+" not exists"
    with open(gt_json) as f:
        file_dicts = json.load(f)

    ret = []

    for file_dict in file_dicts:
        ret.append(
            {
                "image_id": "_".join(
                    os.path.splitext(os.path.basename(file_dict["image"]))[0].split("_")[:3]
                    ),
                "height": file_dict["height"],
                "width":file_dict["width"],
                "file_name": os.path.join(image_dir, file_dict["image"]),
                "vps_label_file_name": os.path.join(image_dir, file_dict["seg"]),
                "depth_label_file_name": os.path.join(image_dir, file_dict["depth"]),
            }
        )
    assert len(ret), f"No images found in {image_dir}!"
    logger.info("Loaded {} images from {}".format(len(ret), image_dir))
    return ret

_RAW_CITYSCAPES_DPS_SPLITS = {
    "cityscapes_dps_train": (
        "dvps/cityscapes-dvps/video_sequence/train",
        "dvps/cityscapes-dvps/video_sequence/train",
        "dvps/cityscapes-dvps/video_sequence/dvps_cityscapes_train.json",
    ),
    "cityscapes_dps_val": (
        "dvps/cityscapes-dvps/video_sequence/val",
        "dvps/cityscapes-dvps/video_sequence/val",
        "dvps/cityscapes-dvps/video_sequence/dvps_cityscapes_val.json",
    ),
}


def register_all_cityscapes_dps():
    root = os.getenv("DETECTRON2_DATASETS", "datasets")
    meta = {}
    thing_classes = [k["name"]  for k in CITYSCAPES_CATEGORIES if k["isthing"]==1]
    thing_colors  = [k["color"] for k in CITYSCAPES_CATEGORIES if k["isthing"]==1]
    stuff_classes = [k["name"]  for k in CITYSCAPES_CATEGORIES if k["isthing"]==0]
    stuff_colors  = [k["color"] for k in CITYSCAPES_CATEGORIES if k["isthing"]==0]

    meta["thing_classes"] = thing_classes
    meta["thing_colors"]  = thing_colors
    meta["stuff_classes"] = stuff_classes
    meta["stuff_colors"]  = stuff_colors

    thing_dataset_id_to_contiguous_id = dict()
    stuff_dataset_id_to_contiguous_id = dict()
    contiguous_id_to_thing_train_id = dict()
    contiguous_id_to_stuff_train_id = dict()
    thing_id = 0
    stuff_id = 0

    for k in CITYSCAPES_CATEGORIES:
        stuff_dataset_id_to_contiguous_id[k["id"]] = k["trainId"]
        contiguous_id_to_stuff_train_id[k["trainId"]] = stuff_id
        stuff_id += 1
        if k["isthing"] == 1:
            thing_dataset_id_to_contiguous_id[k["id"]] = k["trainId"]
            contiguous_id_to_thing_train_id[k["trainId"]] = thing_id
            thing_id += 1

    meta["thing_dataset_id_to_contiguous_id"] = thing_dataset_id_to_contiguous_id
    meta["stuff_dataset_id_to_contiguous_id"] = stuff_dataset_id_to_contiguous_id
    meta["contiguous_id_to_thing_train_id"] = contiguous_id_to_thing_train_id
    meta["contiguous_id_to_stuff_train_id"] = contiguous_id_to_stuff_train_id

    meta["thing_train_id2contiguous_id"] = dict(zip(contiguous_id_to_thing_train_id.values(), 
                                               contiguous_id_to_thing_train_id.keys()))
    meta["stuff_train_id2contiguous_id"] = dict(zip(contiguous_id_to_stuff_train_id.values(),
                                               contiguous_id_to_stuff_train_id.keys()))
    for key, (image_dir, gt_dir, gt_json) in _RAW_CITYSCAPES_DPS_SPLITS.items():
        image_dir = os.path.join(root, image_dir)
        gt_dir = os.path.join(root, gt_dir)
        gt_json = os.path.join(root, gt_json)

        DatasetCatalog.register(
                key, lambda x=image_dir, y=gt_dir, z=gt_json: load_cityscapes_dps(x, y, z, meta)
        )
        MetadataCatalog.get(key).set(
            panoptic_root=gt_dir,
            image_root=image_dir,
            gt_dir=gt_dir,
            evaluator_type="cityscapes_dps",
            ignore_label=255,
            label_divisor=1000,
            **meta,
        )

if __name__ == "__main__":
    register_all_cityscapes_dps()
    dataset = DatasetCatalog.get("cityscapes_dps_train")
    print(dataset[-1])
    print(len(dataset))
    print(dataset[-1].keys())
