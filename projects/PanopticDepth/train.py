# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
PanopticDepth Training Script.

This script is a simplified version of the training script in detectron2/tools.
"""

import torch
import os
os.environ["NCCL_LL_THRESHOLD"] = "0"
import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, launch

from detectron2.data import MetadataCatalog, build_detection_train_loader, build_detection_test_loader

from detectron2.solver import get_default_optimizer_params
from detectron2.solver.build import maybe_add_gradient_clipping

from evaluation import CityscapesDPSEvaluator
from detectron2.evaluation import (
    COCOPanopticEvaluator,
    DatasetEvaluators,
    verify_results,
)

from panoptic_depth import add_panoptic_depth_config, build_lr_scheduler

from data.cityscapes.cityscapes_panoptic_separated import register_all_cityscapes_panoptic
from data.cityscapes.dataset_mapper import CityscapesPanopticDatasetMapper 
from data.cityscapes_dps.cityscapes_dps import register_all_cityscapes_dps
from data.cityscapes_dps.dataset_mapper import CityscapesDPSDatasetMapper

class Trainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluator_list = []
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
        if evaluator_type == "cityscapes_panoptic_seg":
            evaluator_list.append(COCOPanopticEvaluator(dataset_name, output_folder))
        if evaluator_type == "cityscapes_dps":
            assert (
                    torch.cuda.device_count() >= comm.get_rank()
                ), "CityscapesEvaluator currently do not work with multiple machines."
            evaluator_list.append(CityscapesDPSEvaluator(dataset_name, output_folder))
        if len(evaluator_list) == 0:
            raise NotImplementedError(
                "no Evaluator for the dataset {} with the type {}".format(
                    dataset_name, evaluator_type
                )
            )
        elif len(evaluator_list) == 1:
            return evaluator_list[0]
        return DatasetEvaluators(evaluator_list)

        return DatasetEvaluators(evaluator_list)

    @classmethod
    def build_optimizer(cls, cfg, model):
        """
        Build an optimizer from config.
        """
        params = get_default_optimizer_params(
            model,
            base_lr=cfg.SOLVER.BASE_LR,
            weight_decay=cfg.SOLVER.WEIGHT_DECAY,
            weight_decay_norm=cfg.SOLVER.WEIGHT_DECAY_NORM,
            bias_lr_factor=cfg.SOLVER.BIAS_LR_FACTOR,
            weight_decay_bias=cfg.SOLVER.WEIGHT_DECAY_BIAS,
            )

        optimizer_type = cfg.SOLVER.OPTIMIZER
        if optimizer_type == "SGD":
            return maybe_add_gradient_clipping(cfg, torch.optim.SGD)(
                params,
                cfg.SOLVER.BASE_LR,
                momentum=cfg.SOLVER.MOMENTUM,
                nesterov=cfg.SOLVER.NESTEROV,
                )
        elif optimizer_type == "ADAM":
            return maybe_add_gradient_clipping(cfg, torch.optim.Adam)(params, cfg.SOLVER.BASE_LR)
        else:
            raise NotImplementedError(f"no optimizer type {optimizer_type}")
    
    @classmethod
    def build_lr_scheduler(cls, cfg, optimizer):
        """
        It now calls :func:`detectron2.solver.build_lr_scheduler`.
        Overwrite it if you'd like a different scheduler.
        """
        return build_lr_scheduler(cfg, optimizer)

    @classmethod
    def build_train_loader(cls, cfg):
        if cfg.DATASETS.NAME == 'CityscapesDPS':
            mapper = CityscapesDPSDatasetMapper(cfg)
            return build_detection_train_loader(cfg, mapper=mapper)
        elif cfg.DATASETS.NAME == 'Cityscapes':
            mapper = CityscapesPanopticDatasetMapper(cfg)
            return build_detection_train_loader(cfg, mapper=mapper)
        else:
            return build_detection_train_loader(cfg)

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        if cfg.DATASETS.NAME == 'CityscapesDPS':
            mapper = CityscapesDPSDatasetMapper(cfg, is_train=False)
            return build_detection_test_loader(cfg, dataset_name, mapper=mapper)
        else:
            return build_detection_test_loader(cfg, dataset_name)

def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_panoptic_depth_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg

def main(args):
    cfg = setup(args)

    if cfg.DATASETS.NAME == 'Cityscapes':
        register_all_cityscapes_panoptic()
    if cfg.DATASETS.NAME == 'CityscapesDPS':
        register_all_cityscapes_dps()

    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)
        if comm.is_main_process():
            verify_results(cfg, res)
        return res

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()

if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
