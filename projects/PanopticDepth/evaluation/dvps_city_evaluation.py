# Copyright (c) Facebook, Inc. and its affiliates.
import glob
import logging
import numpy as np
import os
import tempfile
from collections import OrderedDict
import torch
from PIL import Image

from detectron2.data import MetadataCatalog
from detectron2.utils import comm
from detectron2.utils.file_io import PathManager

from detectron2.evaluation.cityscapes_evaluation import CityscapesEvaluator

from . import eval_cityscapes_dvpq as cityscapes_eval

def make_colors():
    from detectron2.data.datasets.builtin_meta import COCO_CATEGORIES
    colors = []
    for cate in COCO_CATEGORIES:
        colors.append(cate["color"])
    return colors

def vis_seg(seg_map):
    colors = make_colors()
    h,w = seg_map.shape
    color_mask = np.zeros([h,w,3])
    for seg_id in np.unique(seg_map):
        color_mask[seg_map==seg_id] = colors[seg_id%len(colors)]
    return color_mask.astype(np.uint8)


class CityscapesDPSEvaluator(CityscapesEvaluator):
    """
    Evaluate video panoptic segmentation results on cityscapes dataset using cityscapes API.

    Note:
        * It does not work in multi-machine distributed training.
        * It contains a synchronization, therefore has to be used on all ranks.
        * Only the main process runs evaluation.
    """
    def __init__(
        self, dataset_name, output_folder):
        super().__init__(dataset_name)
        self.output_folder = output_folder
        self.evaluator = cityscapes_eval
        self.eval_frames = [1, 2, 3, 4]
        self.depth_thres = [-1, 0.5, 0.25, 0.1]

    def process(self, inputs, outputs):
        save_dir = self._temp_dir
        for input, output in zip(inputs, outputs):
            basename = os.path.basename(input['file_name'])
            pred_depth = output['depth'].to(self._cpu_device).numpy()
            result_panoptic = output['panoptic_2Chn']
            pan_result = torch.stack(
                [result_panoptic[1], 
                 result_panoptic[0], 
                 torch.zeros_like(result_panoptic[0]),
                ], dim=2)

            pred_depth = (pred_depth*256).astype(np.int32)
            pan_result = pan_result.to(self._cpu_device).numpy().astype(np.uint8)

            Image.fromarray(pred_depth).save(
                os.path.join(save_dir, basename.replace("_leftImg8bit.png", "_depth.png")))
            Image.fromarray(pan_result).save(
                os.path.join(save_dir, basename.replace("_leftImg8bit.png", "_panoptic.png")))
            Image.fromarray(vis_seg(pan_result[:,:,1])).save(
                os.path.join(save_dir, basename.replace("_leftImg8bit.png", "_panoptic_vis.png")))

    def evaluate(self):
        comm.synchronize()
        if comm.get_rank() > 0:
            return
        return self.evaluate_dpq()

    def evaluate_dpq(self):
        self._logger.info("Evaluating results under {} ...".format(self._temp_dir))
        dpq = {}
        pred_dir = self._temp_dir
        gt_dir = os.path.join(os.environ['DETECTRON2_DATASETS'], self._metadata.gt_dir)
        for depth_thres in self.depth_thres:
            results = self.evaluator.main(1, pred_dir, gt_dir, depth_thres)
            dpq[depth_thres] = {'dpq':    results['averages'][0],
                                'dpq_th': results['averages'][1],
                                'dpq_st': results['averages'][2]}
        ret = OrderedDict()
        ret.update(dpq)
        ret['averages'] = {
            'dpq':    np.array([dpq[depth_thres]['dpq']    for depth_thres in self.depth_thres if depth_thres > 0]).mean(),
            'dpq_th': np.array([dpq[depth_thres]['dpq_th'] for depth_thres in self.depth_thres if depth_thres > 0]).mean(),
            'dpq_st': np.array([dpq[depth_thres]['dpq_st'] for depth_thres in self.depth_thres if depth_thres > 0]).mean()
            }

        self._working_dir.cleanup()
        return ret
