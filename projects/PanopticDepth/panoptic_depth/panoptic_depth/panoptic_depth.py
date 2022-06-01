import torch
from torch import nn
from torch.nn import functional as F

from PIL import Image
import numpy as np
import json
import os

from detectron2.data import MetadataCatalog
from detectron2.structures import ImageList, Instances, BitMasks
from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY

from ..loss import sigmoid_focal_loss, weighted_dice_loss
from ..backbone_utils import build_semanticfpn, build_backbone
from ..utils import topk_score, multi_apply, gather_feature
from .gt_generate import GenerateGT
from .head import build_position_head, build_kernel_head, build_feature_encoder
from .head import build_thing_generator, build_stuff_generator
from .head import build_depth_kernel_head, build_depth_map_generator,build_depth_feature_encoder

__all__ = ["PanopticDepth"]

@META_ARCH_REGISTRY.register()
class PanopticDepth(nn.Module):
    """
    """
    def __init__(self, cfg):
        super().__init__()
        
        self.device                = torch.device(cfg.MODEL.DEVICE)
        # parameters
        self.cfg                   = cfg
        self.ignore_val            = cfg.MODEL.IGNORE_VALUE
        self.common_stride         = cfg.MODEL.SEMANTIC_FPN.COMMON_STRIDE

        self.center_top_num        = cfg.MODEL.POSITION_HEAD.THING.TOP_NUM
        self.weighted_num          = cfg.MODEL.POSITION_HEAD.THING.POS_NUM
        self.center_thres          = cfg.MODEL.POSITION_HEAD.THING.THRES
        self.sem_thres             = cfg.MODEL.POSITION_HEAD.STUFF.THRES
        self.sem_classes           = cfg.MODEL.POSITION_HEAD.STUFF.NUM_CLASSES
        self.sem_with_thing        = cfg.MODEL.POSITION_HEAD.STUFF.WITH_THING
        self.sem_all_classes       = cfg.MODEL.POSITION_HEAD.STUFF.ALL_CLASSES
        if self.sem_all_classes:
            assert self.sem_with_thing
        self.in_feature            = cfg.MODEL.FEATURE_ENCODER.IN_FEATURES
        self.inst_scale            = cfg.MODEL.KERNEL_HEAD.INSTANCE_SCALES

        self.pos_weight_thing      = cfg.MODEL.LOSS_WEIGHT.POSITION_THING
        self.pos_weight_stuff      = cfg.MODEL.LOSS_WEIGHT.POSITION_STUFF
        self.seg_weight            = cfg.MODEL.LOSS_WEIGHT.SEGMENT
        self.focal_loss_alpha      = cfg.MODEL.LOSS_WEIGHT.FOCAL_LOSS_ALPHA
        self.focal_loss_gamma      = cfg.MODEL.LOSS_WEIGHT.FOCAL_LOSS_GAMMA
        
        self.inst_thres            = cfg.MODEL.INFERENCE.INST_THRES
        self.panoptic_combine      = cfg.MODEL.INFERENCE.COMBINE.ENABLE
        self.panoptic_overlap_thrs = cfg.MODEL.INFERENCE.COMBINE.OVERLAP_THRESH
        self.panoptic_stuff_limit  = cfg.MODEL.INFERENCE.COMBINE.STUFF_AREA_LIMIT
        self.panoptic_inst_thrs    = cfg.MODEL.INFERENCE.COMBINE.INST_THRESH

        self.depth_on              = cfg.MODEL.DEPTH_ON
        self.depth_factor          = cfg.MODEL.DEPTH_HEAD.FACTOR
        self.depth_factor_ins      = cfg.MODEL.DEPTH_HEAD.FACTOR_INS

        # backbone
        self.backbone              = build_backbone(cfg)
        self.semantic_fpn          = build_semanticfpn(cfg, self.backbone.output_shape())
        self.position_head         = build_position_head(cfg)
        self.kernel_head           = build_kernel_head(cfg)
        self.feature_encoder       = build_feature_encoder(cfg)

        if self.depth_on:
            self.depth_encode_head     = build_depth_feature_encoder(cfg) 
            self.depth_kernel_head     = build_depth_kernel_head(cfg)
            self.depth_map_generator   = build_depth_map_generator(cfg)

            self.stuff_mask            = torch.ones(self.sem_classes, device=self.device).bool()
            if not self.sem_all_classes:
                self.stuff_mask[-1] = False
            else:
                self.stuff_mask[-cfg.MODEL.POSITION_HEAD.THING.NUM_CLASSES:] = False

        self.thing_generator       = build_thing_generator(cfg)
        self.stuff_generator       = build_stuff_generator(cfg)
        self.get_ground_truth      = GenerateGT(cfg)

        pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).to(self.device).view(3, 1, 1)
        pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).to(self.device).view(3, 1, 1)
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std

        dataset_names = self.cfg.DATASETS.TRAIN
        self.meta = MetadataCatalog.get(dataset_names[0])

        self.to(self.device)

    def forward(self, batched_inputs):
        images, encode_feat, pred_centers, pred_regions, pred_weights, pred_depth_kernels, encode_depths = self.backbone_and_neck(batched_inputs)
        if self.training:
            gt_dict = self.get_ground_truth.generate(batched_inputs, images, pred_weights, encode_feat)
            return self.losses(pred_centers, pred_regions, pred_weights, pred_depth_kernels, encode_feat, encode_depths, gt_dict)
        else:
            return self.inference(batched_inputs, images, pred_centers, pred_regions, pred_weights, encode_feat, encode_depths, pred_depth_kernels)

    def backbone_and_neck(self, batched_inputs):
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [self.normalizer(x) for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)

        features = self.backbone(images.tensor)
        semantic_fpn_feat = self.semantic_fpn(features)
        encode_feat = self.feature_encoder(semantic_fpn_feat)
        features_in = [features[_feat] for _feat in self.in_feature]
        pred_centers, pred_regions, pred_weights, pred_depth_kernels = multi_apply(self.forward_single_level, features_in)

        encode_depths = self.depth_encode_head(semantic_fpn_feat) if self.depth_on else None
        return images, encode_feat, pred_centers, pred_regions, pred_weights, pred_depth_kernels, encode_depths

    def forward_single_level(self, feature):
        pred_center, pred_region = self.position_head(feature)
        pred_weight = self.kernel_head(feature)
        pred_depth_kernel = self.depth_kernel_head(feature) if self.depth_on else None
        return pred_center, pred_region, pred_weight, pred_depth_kernel

    def losses(self, pred_centers, pred_regions, pred_weights, pred_depth_kernels, encode_feat, encode_depths, gt_dict):
        """
        Calculate losses of prediction with generated gt dict.

        Args:
            pred_centers: prediction for object centers
            pred_regions: prediction for stuff regions
            pred_weights: generated kernel weights for things and stuff
            pred_depth_kernels: generated depth kernel weights for things
            encode_feat: encoded high-resolution feature
            gt_dict(dict): a dict contains all information of gt
            gt_dict = {
                "center": gt gaussian scoremap for things,
                "inst": gt instance target for things,
                "index": gt index for things,
                "index_mask": gt index mask for things,
                "class": gt classes for things,
                "sem_scores": gt semantic score map for stuff,
                "sem_labels":gt masks for stuff,
                "sem_index": gt index for stuff,
                "sem_masks": gt index mask for stuff,
                "depth": gt depth map for the whole image
            }

        Returns:
            loss(dict): a dict contains all information of loss function
            loss = {
                "loss_pos_th": position loss for things,
                "loss_pos_st": position loss for stuff,
                "loss_seg_th": segmentation loss for things,
                "loss_seg_st": segmentation loss for stuff,
            }
        """
        feat_shape = encode_feat.shape
        encode_feat = encode_feat.reshape(*feat_shape[:2], -1)
        loss_pos_ths, loss_pos_sts, idx_feat_th, weighted_values, idx_feat_st, idx_depth_th, idx_depth_st, thing_nums, stuff_nums = \
                        multi_apply(self.loss_single_level, pred_centers,
                                    pred_regions, pred_weights, pred_depth_kernels,
                                    gt_dict["center"], gt_dict["inst"], 
                                    gt_dict["index_mask"], gt_dict["class"], 
                                    gt_dict["sem_scores"], gt_dict["sem_masks"], 
                                    gt_dict["sem_index"])

        thing_num = sum(thing_nums)
        stuff_num = sum(stuff_nums)
        idx_feat_th = torch.cat(idx_feat_th, dim=2)
        idx_depth_th = torch.cat(idx_depth_th, dim=2) if self.depth_on else None
        weighted_values = torch.cat(weighted_values, dim=1)
        idx_feat_st = torch.cat(idx_feat_st, dim=1)
        idx_feat_st = idx_feat_st.reshape(-1, *idx_feat_st.shape[2:])
        idx_depth_st = torch.cat(idx_depth_st, dim=1) if self.depth_on else None

        thing_pred, _, thing_depth_pred = self.thing_generator(encode_feat, feat_shape, idx_feat_th, thing_num, idx_depth_th)
        stuff_pred, _ = self.stuff_generator(encode_feat, feat_shape, idx_feat_st, stuff_num)
        
        # for thing
        thing_gt_idx = [_gt[:,:thing_nums[_idx]] for _idx, _gt in enumerate(gt_dict["index_mask"])]
        thing_gt_idx = torch.cat(thing_gt_idx, dim=1)
        thing_gt_idx = thing_gt_idx.bool()
        thing_gt_num = int(thing_gt_idx.sum())
        thing_gt = [_gt[:,:thing_nums[_idx],...] for _idx, _gt in enumerate(gt_dict["inst"])]
        thing_gt = torch.cat(thing_gt, dim=1)
        # for stuff
        stuff_gt_idx = [_gt[:,:stuff_nums[_idx]] for _idx, _gt in enumerate(gt_dict["sem_index"])]
        stuff_gt_idx = torch.cat(stuff_gt_idx, dim=1)
        stuff_gt_idx = stuff_gt_idx.bool()
        stuff_gt_num = int(stuff_gt_idx.sum())
        stuff_gt = [_gt[:,:stuff_nums[_idx],...] for _idx, _gt in enumerate(gt_dict["sem_labels"])]
        stuff_gt = torch.cat(stuff_gt, dim=1)

        loss_thing = weighted_dice_loss(thing_pred, thing_gt, 
                                        gt_num=thing_gt_num,
                                        index_mask=thing_gt_idx.reshape(-1),
                                        instance_num=thing_num,
                                        weighted_val=weighted_values,
                                        weighted_num=self.weighted_num,
                                        mode="thing",
                                        reduction="sum")

        loss_stuff = weighted_dice_loss(stuff_pred, stuff_gt, 
                                        gt_num=stuff_gt_num,
                                        index_mask=stuff_gt_idx.reshape(-1),
                                        instance_num=stuff_num,
                                        weighted_val=1.0,
                                        weighted_num=1,
                                        mode="stuff",
                                        reduction="sum")

        loss = {}
        # position loss
        loss["loss_pos_th"] = self.pos_weight_thing * sum(loss_pos_ths) / max(thing_gt_num, 1)
        loss["loss_pos_st"] = self.pos_weight_stuff * sum(loss_pos_sts) / max(feat_shape[0],1)
        # segmentation loss
        loss["loss_seg_th"] = self.seg_weight * loss_thing / max(thing_gt_num, 1)
        loss["loss_seg_st"] = self.seg_weight * loss_stuff / max(stuff_gt_num, 1)

        if not self.depth_on:
            return loss
        ######## depth loss ##########
        depth_gt = gt_dict['depth']
        valid_flag = depth_gt > 0

        # instance depth loss for thing
        thing_gt[thing_gt==self.meta.ignore_label] = 0
        select = [thing_gt[_idx][select] * valid_flag[_idx] for _idx, select in enumerate(thing_gt_idx)]
        keep = [s.sum(dim=(1,2)) > 0 for s in select]
        select = [select[_idx][keep_s] for _idx, keep_s in enumerate(keep)]
        keep = torch.cat(keep)
        thing_mask = torch.zeros_like(thing_gt_idx).bool()
        temp = thing_mask[thing_gt_idx]
        temp[keep] = True
        thing_mask[thing_gt_idx] = temp
        del temp, keep

        if thing_mask.sum() > 0:
            thing_depth_pred = thing_depth_pred.reshape(thing_depth_pred.shape[0], -1, self.weighted_num, thing_depth_pred.shape[-1])
            thing_depth_pred, thing_depth_mean = self.depth_map_generator(encode_depths, thing_depth_pred, True)
            thing_depth_pred = thing_depth_pred[thing_mask]

            thing_depth_gt = [d.repeat((s.shape[0],1,1)) for d, s in zip(depth_gt, select)]
            thing_depth_gt = torch.cat(thing_depth_gt, dim=0).unsqueeze(1)
            select = torch.cat(select, dim=0).unsqueeze(1) > 0
            thing_depth_gt = thing_depth_gt * select

            thing_depth_mean_gt_raw = thing_depth_gt.sum(dim=(-1,-2),keepdims=True) / select.sum(dim=(-1,-2),keepdims=True).clamp(min=1)

            thing_depth_mean_gt = torch.zeros_like(thing_depth_mean)
            thing_depth_mean_gt[thing_mask] = thing_depth_mean_gt_raw
            del thing_depth_mean_gt_raw
            thing_mask = thing_mask.repeat((self.weighted_num,1,1)).permute(1,2,0)
            thing_depth_mean_gt = thing_depth_mean_gt.reshape(thing_mask.shape[0],1,-1)
            thing_depth_mean = thing_depth_mean.reshape(thing_mask.shape[0],1,-1)
            thing_mask = thing_mask.reshape(thing_mask.shape[0],1,-1)
            loss["loss_depth_ins"] = self.depth_loss(thing_depth_mean_gt, thing_depth_mean, thing_mask) * self.depth_factor_ins * self.depth_factor
        else:
            thing_depth_gt = None
            loss["loss_depth_ins"] = thing_depth_pred.sum() * 0.

        # instance depth loss for stuff
        stuff_gt[stuff_gt==self.meta.ignore_label] = 0
        gt_sem_one_hot = [s.sum(dim=(-1,-2)) > 0 for s in gt_dict['sem_scores']]
        stuff_mask = [self.stuff_mask.unsqueeze(0) & s for s in gt_sem_one_hot]
        gt_sem_one_hot = torch.cat(gt_sem_one_hot, dim=1)
        stuff_mask = torch.cat(stuff_mask, dim=1)

        select_st = [stuff_gt[_idx][select_st] * valid_flag[_idx] for _idx, select_st in enumerate(stuff_gt_idx)]
        keep = [(s.sum(dim=(1,2)) > 0) & sm[gs > 0] for s, sm, gs in zip(select_st, stuff_mask, gt_sem_one_hot)]
        select_st = [select_st[_idx][keep_s] for _idx, keep_s in enumerate(keep)]

        idx_depth_st = idx_depth_st.squeeze(-1).squeeze(-1).unsqueeze(-2)
        stuff_depth_pred = self.depth_map_generator(encode_depths, idx_depth_st, False)
        stuff_depth_pred = stuff_depth_pred[stuff_gt_idx][torch.cat(keep)]
        stuff_depth_gt = [d.repeat((s.shape[0],1,1)) for d, s in zip(depth_gt, select_st)]
        stuff_depth_gt = torch.cat(stuff_depth_gt, dim=0).unsqueeze(1)
        select_st = torch.cat(select_st, dim=0).unsqueeze(1) > 0.5

        assert len(thing_gt_idx)==1, len(thing_gt_idx)
        stuff_depth_gt = stuff_depth_gt * select_st
        if thing_depth_gt is None:
            thing_stuff_depth_gt   = stuff_depth_gt.reshape(-1)
            thing_stuff_depth_pred = stuff_depth_pred.reshape(-1)
        else:
            thing_stuff_depth_gt   = torch.cat([thing_depth_gt.repeat(1,self.weighted_num,1,1).reshape(-1), \
                                                stuff_depth_gt.reshape(-1)], dim=0)
            thing_stuff_depth_pred = torch.cat([thing_depth_pred.reshape(-1), \
                                                stuff_depth_pred.reshape(-1)], dim=0)
        loss["loss_depth"] = self.depth_loss_flatten(thing_stuff_depth_gt, thing_stuff_depth_pred, thing_stuff_depth_gt > 0) * self.depth_factor

        return loss

    def depth_loss(self, depth_gt, pred_depths, valid_mask):
        eps = 1e-6
        depth_gt = depth_gt.clamp(min=eps)
        pred_depths = pred_depths.clamp(min=eps)
        valid_nums = valid_mask.sum(dim=(-1,-2)).clamp(min=1)

        depth_log_diff = torch.log(depth_gt) - torch.log(pred_depths)
        scale_invar_log_error_1 = ((depth_log_diff ** 2) * valid_mask).sum(dim=(-1,-2)) / valid_nums
        scale_invar_log_error_2 = ((depth_log_diff * valid_mask).sum(dim=(-1,-2)) ** 2) / (valid_nums**2)
        relat_sqrt_error = torch.sqrt((((1. - pred_depths / depth_gt) ** 2) * valid_mask).sum(dim=(-1,-2)) / valid_nums)
        loss = (scale_invar_log_error_1 - scale_invar_log_error_2) * 5. + relat_sqrt_error
        return loss.mean()

    def depth_loss_flatten(self, depth_gt, pred_depths, valid_mask=None):
        if valid_mask is not None:
            pred_depths = pred_depths[valid_mask]
            depth_gt = depth_gt[valid_mask]
        eps = 1e-6
        depth_gt = depth_gt.clamp(min=eps)
        pred_depths = pred_depths.clamp(min=eps)

        depth_log_diff = torch.log(depth_gt) - torch.log(pred_depths)
        scale_invar_log_error_1 = torch.mean(depth_log_diff ** 2)
        scale_invar_log_error_2 = (torch.sum(depth_log_diff) ** 2) / (depth_log_diff.numel()** 2)
        relat_sqrt_error = torch.sqrt(torch.mean((1. - pred_depths / depth_gt) ** 2))
        return (scale_invar_log_error_1 - scale_invar_log_error_2) * 5. + relat_sqrt_error

    def loss_single_level(self, pred_center, pred_region, pred_weights, pred_depth_kernels, \
                          gt_center, gt_inst, gt_index_mask, gt_class, \
                          gt_sem_scores, gt_sem_masks, gt_sem_index):
        # position loss for things
        loss_pos_th = sigmoid_focal_loss(pred_center, gt_center,
                                         mode="thing",
                                         alpha=self.focal_loss_alpha,
                                         gamma=self.focal_loss_gamma,
                                         reduction="sum")
        # position loss for stuff
        loss_pos_st = sigmoid_focal_loss(pred_region, gt_sem_scores,
                                         mode="stuff",
                                         alpha=self.focal_loss_alpha,
                                         gamma=self.focal_loss_gamma,
                                         reduction="sum")
        # generate guided center
        batch_num, _, feat_h, feat_w = pred_center.shape
        guided_inst = F.interpolate(gt_inst, size=(feat_h, feat_w), mode='bilinear', align_corners=False)
        guidence = torch.zeros_like(guided_inst)
        pred_select = []
        for _idx in range(batch_num):
            sub_pred = pred_center[_idx]
            sub_class = gt_class[_idx].to(torch.int64)
            sub_select = torch.index_select(sub_pred, dim=0, index=sub_class)
            pred_select.append(sub_select.sigmoid())

        pred_select = torch.stack(pred_select, dim=0)
        keep = (guided_inst > 0.1) & (guided_inst < 255)
        guidence[keep] = pred_select[keep]

        weighted_values, guided_index = torch.topk(guidence.reshape(*guided_inst.shape[:2], -1), 
                                                   k=self.weighted_num, dim=-1)

        thing_num = int(max(gt_index_mask.sum(dim=1).max(), 1))
        guided_index = guided_index[:,:thing_num, :]
        guided_index = guided_index.reshape(batch_num, -1)
        weighted_values = weighted_values[:,:thing_num, :]
        # pred instance
        weight_shape = pred_weights.shape
        inst_w = pred_weights.reshape(*weight_shape[:2], -1)
        idx_inst = guided_index.unsqueeze(1).expand(*weight_shape[:2], -1)
        idx_feat_th = torch.gather(inst_w, dim=2, index=idx_inst)
        idx_feat_th = idx_feat_th.reshape(*weight_shape[:2], thing_num, self.weighted_num)
        # generate guided sem
        stuff_num = int(max(gt_sem_index.sum(dim=1).max(), 1))
        gt_sem_masks = gt_sem_masks[:, :stuff_num]
        gt_sem_masks = gt_sem_masks.unsqueeze(2)
        idx_feat_st = gt_sem_masks * pred_weights.unsqueeze(1)
        idx_feat_st = idx_feat_st.reshape(-1, *weight_shape[-3:])
        num_class_st = torch.clamp(gt_sem_masks.reshape(-1, 1, *weight_shape[-2:]).sum(dim=(2,3),keepdims=True), min=1)
        idx_feat_st = idx_feat_st.sum(dim=(-1,-2), keepdims=True) / num_class_st
        idx_feat_st = idx_feat_st.reshape(batch_num, -1, weight_shape[1], 1, 1)
   
        if self.depth_on:
            # pred instance depth kernels for thing
            depth_shape = pred_depth_kernels.shape
            inst_d = pred_depth_kernels.reshape(*depth_shape[:2], -1)
            idx_inst = guided_index.unsqueeze(1).expand(*depth_shape[:2], -1)
            idx_depth_th = torch.gather(inst_d, dim=2, index=idx_inst)
            idx_depth_th = idx_depth_th.reshape(*depth_shape[:2], thing_num, self.weighted_num)
            # pred instance depth kernels for stuff
            idx_depth_st = gt_sem_masks * pred_depth_kernels.unsqueeze(1)
            idx_depth_st = idx_depth_st.reshape(-1, *depth_shape[-3:])
            idx_depth_st = idx_depth_st.sum(dim=(2,3),keepdims=True) / num_class_st
            idx_depth_st = idx_depth_st.reshape(batch_num, -1, depth_shape[1], 1, 1)
        else:
            idx_depth_th, idx_depth_st = None, None

        return loss_pos_th, loss_pos_st, idx_feat_th, weighted_values, idx_feat_st, idx_depth_th, idx_depth_st, thing_num, stuff_num

    @torch.no_grad()
    def inference_single_level(self, pred_center, pred_region, pred_weights, pred_depth_kernels, pool_size):
        # pred things
        pred_center = pred_center.sigmoid()
        center_pool = F.avg_pool2d(pred_center, kernel_size=pool_size, 
                                    stride=1, padding=(pool_size-1)//2)
        pred_center = (pred_center + center_pool) / 2.0
        fmap_max = F.max_pool2d(pred_center, 3, stride=1, padding=1)
        keep = (fmap_max == pred_center).float()
        pred_center *= keep

        weight_shape = pred_weights.shape
        center_shape = pred_center.shape
        top_num = min(center_shape[-2]*center_shape[-1], self.center_top_num//2)
        sub_score, sub_index, sub_class, ys, xs = \
                topk_score(pred_center, K=top_num, score_shape=center_shape)
        keep = sub_score > self.center_thres
        score_th = sub_score[keep]
        class_th = sub_class[keep]
        index = sub_index[keep]
        index = index.unsqueeze(0).to(device=self.device, dtype=torch.long)

        thing_num = keep.sum()
        if thing_num > 0:
            inst_w = pred_weights.reshape(*weight_shape[:2], -1)
            idx_inst = index.unsqueeze(1).expand(*weight_shape[:2], -1)
            idx_feat_th = torch.gather(inst_w, dim=2, index=idx_inst)
            idx_feat_th = idx_feat_th.unsqueeze(-1)
            if self.depth_on:
                depth_kernel_th = gather_feature(pred_depth_kernels, sub_index, mask=keep, use_transform=True)
                depth_kernel_th = depth_kernel_th.permute(1,0).unsqueeze(0).unsqueeze(-1)
            else:
                depth_kernel_th = None
        else:
            idx_feat_th, class_th, score_th = [], [], []
            depth_kernel_th = []
        
        # pred stuff
        pred_region = pred_region.sigmoid()
        pred_cate = pred_region.argmax(dim=1)

        class_st, num_class_st = torch.unique(pred_cate, return_counts=True)
        pred_st_mask = F.one_hot(pred_cate, num_classes=self.sem_classes)
        pred_st_mask = pred_st_mask.permute(0, 3, 1, 2).contiguous()
        pred_st_mask = pred_st_mask * (pred_region >= 0.4)

        score_st = (pred_region * pred_st_mask).reshape(1, self.sem_classes, -1)
        score_st = (score_st.sum(dim=-1)[:, class_st] / num_class_st).squeeze(0)
        pred_st_mask = pred_st_mask[:, class_st]
        keep = score_st > self.sem_thres
        stuff_num = keep.sum()
        score_st = score_st[keep]
        class_st = class_st[keep]
        pred_st_mask = pred_st_mask[:, keep]

        pred_st_mask = pred_st_mask.unsqueeze(2)
        idx_feat_st = pred_st_mask * pred_weights.unsqueeze(1)
        idx_feat_st = idx_feat_st.reshape(-1, *weight_shape[-3:])
        idx_feat_st = idx_feat_st.sum(dim=(-1,-2), keepdims=True) / num_class_st[keep].reshape(-1, 1, 1, 1)

        if not (self.sem_all_classes or self.sem_with_thing):
            class_st += 1

        if self.depth_on:
            depth_shape = pred_depth_kernels.shape
            depth_kernel_st = pred_st_mask * pred_depth_kernels.unsqueeze(1)
            depth_kernel_st = depth_kernel_st.reshape(-1, *depth_shape[-3:])
            depth_kernel_st = depth_kernel_st.sum(dim=(-1,-2), keepdims=True) / \
                          torch.clamp(pred_st_mask.reshape(-1, 1, *depth_shape[-2:]).sum(dim=(-1,-2), keepdims=True), min=1)
        else:
            depth_kernel_th, depth_kernel_st = None, None

        return idx_feat_th, class_th, score_th, depth_kernel_th, thing_num, idx_feat_st, score_st, class_st, depth_kernel_st, stuff_num
    
    @torch.no_grad()
    def inference(self, batch_inputs, images, pred_centers, pred_regions, pred_weights, encode_feat, encode_depths, pred_depth_kernels):
        """
        PanopticDepth inference process.

        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper`
            image: ImageList in detectron2.structures
            pred_centers: prediction for object centers
            pred_regions: prediction for stuff regions
            pred_weights: generated kernel weights for things and stuff
            encode_feat: encoded high-resolution feature
        
        Returns:
            processed_results(dict): a dict contains all predicted results
            processed_results={
                "sem_seg": prediction of stuff for semantic segmentation eval, 
                "instances": prediction of things for instance segmentation eval,
                "panoptic_seg": prediction of both for panoptic segmentation eval.
            }
        """
        results = batch_inputs
        processed_results = []
        for img_idx, result_img in enumerate(results):
            if "instances" in result_img.keys():
                img_shape = result_img["instances"].image_size
            else:
                img_shape = result_img["image"].shape[-2:]
            ori_shape = (result_img["height"], result_img["width"])
            encode_feat = encode_feat[img_idx].unsqueeze(0)
            feat_shape = encode_feat.shape
            encode_feat = encode_feat.reshape(*feat_shape[:2], -1)
            result_instance = None

            pred_regions = [_pred[img_idx].unsqueeze(0) for _pred in pred_regions]
            pred_weights = [_pred[img_idx].unsqueeze(0) for _pred in pred_weights]
            if self.depth_on:
                pred_depth_kernels = [_pred[img_idx].unsqueeze(0) for _pred in pred_depth_kernels]
            pred_centers = [_pred[img_idx].unsqueeze(0) for _pred in pred_centers]
            pool_size = [3,3,3,5,5]
            idx_feat_th, class_ths, score_ths, depth_kernel_ths, thing_num, idx_feat_st, score_sts, class_sts, depth_kernel_sts, stuff_num = \
                        multi_apply(self.inference_single_level, pred_centers,\
                            pred_regions, pred_weights, pred_depth_kernels, pool_size)
    
            thing_num = sum(thing_num)
            if thing_num == 0:
                result_instance = Instances(ori_shape, pred_masks=[], pred_boxes=[], 
                                            pred_classes=[], scores=[])
            else:
                score_ths        = [_score        for _score        in score_ths        if len(_score)>0]
                class_ths        = [_class        for _class        in class_ths        if len(_class)>0]
                idx_feat_th      = [_feat         for _feat         in idx_feat_th      if len(_feat)>0]
                class_ths = torch.cat(class_ths, dim=0)
                score_ths = torch.cat(score_ths, dim=0)
                idx_feat_th = torch.cat(idx_feat_th, dim=2)
                keep = torch.argsort(score_ths, descending=True)
                class_ths = class_ths[keep]
                score_ths = score_ths[keep]
                idx_feat_th = idx_feat_th[:,:,keep]
                if self.depth_on:
                    depth_kernel_ths = [_depth_kernel for _depth_kernel in depth_kernel_ths if len(_depth_kernel)>0]
                    depth_kernel_ths = torch.cat(depth_kernel_ths, dim=2)
                    depth_kernel_ths = depth_kernel_ths[:,:,keep]
                else:
                    depth_kernel_ths = None

            stuff_num = sum(stuff_num)
            if stuff_num == 0:
                class_sts, idx_feat_st, score_sts = [], [], []
                depth_kernel_sts = []
            else:
                score_sts = [_score for _score in score_sts if len(_score)>0]
                class_sts = [_cate_sem for _cate_sem in class_sts if len(_cate_sem)>0]
                idx_feat_st = [_feat for _feat in idx_feat_st if len(_feat)>0]
                score_sts = torch.cat(score_sts, dim=0)
                class_sts = torch.cat(class_sts, dim=0)
                idx_feat_st = torch.cat(idx_feat_st, dim=0)
                ###
                if self.depth_on:
                    depth_kernel_sts = [_ for _ in depth_kernel_sts if len(_) > 0]
                    depth_kernel_sts = torch.cat(depth_kernel_sts, dim=0)
                else:
                    depth_kernel_sts = None

            pred_thing, [class_ths, score_ths], depth_kernel_ths = \
                    self.thing_generator(encode_feat, feat_shape, idx_feat_th, thing_num, depth_kernel_ths, class_ths, score_ths)
            pred_stuff, [class_sts, score_sts, depth_kernel_sts] = \
                    self.stuff_generator(encode_feat, feat_shape, idx_feat_st, stuff_num, class_sts, score_sts, depth_kernel_sts)
            pred_stuff = pred_stuff.sigmoid()
            
            if result_instance is None:
                result_instance, pred_inst, class_ths, score_ths, depth_kernels_ths = self.process_inst(
                            class_ths, score_ths, depth_kernel_ths, pred_thing, img_shape, ori_shape)
            else:
                pred_inst, class_ths, score_ths = None, None, None
                depth_kernels_ths = None, None
            if self.sem_with_thing or self.sem_all_classes:
                sem_classes = self.sem_classes
            else:
                sem_classes = self.sem_classes + 1

            pred_stuff = F.interpolate(pred_stuff, scale_factor=self.common_stride, mode="bilinear", 
                                       align_corners=False)[...,:img_shape[0],:img_shape[1]]
            pred_stuff = F.interpolate(pred_stuff, size=ori_shape, mode="bilinear", align_corners=False)[0]
            pred_sem_seg = torch.zeros(sem_classes, *pred_stuff.shape[-2:], device=self.device)
            pred_sem_seg[class_sts] += pred_stuff
            processed_results.append({"sem_seg": pred_sem_seg, "instances": result_instance})

            if self.panoptic_combine:
                if self.depth_on:
                    encode_depth = encode_depths[img_idx].unsqueeze(0)
                    encode_depth = F.interpolate(encode_depth, scale_factor=self.common_stride, mode="bilinear",
                                            align_corners=False)[...,:img_shape[0],:img_shape[1]]
                    encode_depth = F.interpolate(encode_depths, size=ori_shape, mode="bilinear", align_corners=False)[0]
                else:
                    encode_depth = None

                result_panoptic, pred_depth = self.combine_thing_and_stuff(
                    [pred_inst, class_ths, score_ths, depth_kernels_ths],
                    [pred_sem_seg, depth_kernel_sts, class_sts, pred_stuff],
                    self.panoptic_overlap_thrs,
                    self.panoptic_stuff_limit,
                    self.panoptic_inst_thrs,
                    encode_depth, 
                    with_void=not self.depth_on)
                if self.depth_on:
                    pred_depth = torch.clamp(pred_depth.squeeze(0), min=0., max=80.)
                    processed_results[-1]["depth"] = pred_depth
                processed_results[-1]["panoptic_2Chn"] = (result_panoptic[0], result_panoptic[1])
                processed_results[-1]["panoptic_seg"]  = (result_panoptic[1] * self.meta.label_divisor + result_panoptic[0], None)
        return processed_results

    @torch.no_grad()
    def process_inst(self, classes, scores, depth_kernels, pred_inst, img_shape, ori_shape):
        """
        Simple process generate prediction of Things.

        Args:
            classes: predicted classes of Things
            scores: predicted scores of Things
            pred_inst: predicted instances of Things
            img_shape: input image shape
            ori_shape: original image shape
        
        Returns:
            result_instance: preserved results for Things
            pred_mask: preserved binary masks for Things
            classes: preserved object classes
            scores: processed object scores
        """
        pred_inst = pred_inst.sigmoid()[0]
        pred_mask = pred_inst > self.inst_thres
        # object rescore.
        sum_masks = pred_mask.sum((1, 2)).float() + 1e-6
        seg_score = (pred_inst * pred_mask.float()).sum((1, 2)) / sum_masks
        scores *= seg_score

        keep = torch.argsort(scores, descending=True)
        pred_inst = pred_inst[keep]
        pred_mask = pred_mask[keep]
        scores = scores[keep]
        classes = classes[keep]
        depth_kernels = depth_kernels[keep] if self.depth_on else None
        sum_masks = sum_masks[keep]
        
        # object score filter.
        keep = scores >= 0.05
        if keep.sum() == 0:
            result_instance = Instances(ori_shape, pred_masks=[], pred_boxes=[], 
                                        pred_classes=[], scores=[])
            return result_instance, pred_mask, None, None, None
        pred_inst = pred_inst[keep]
        scores = scores[keep]
        classes = classes[keep]
        depth_kernels = depth_kernels[keep] if self.depth_on else None

        # sort and keep top_k
        keep = torch.argsort(scores, descending=True)
        keep = keep[:self.center_top_num]
        pred_inst = pred_inst[keep]
        scores = scores[keep].reshape(-1)
        classes = classes[keep].reshape(-1).to(torch.int32)
        depth_kernels = depth_kernels[keep] if self.depth_on else None
        
        pred_inst = F.interpolate(pred_inst.unsqueeze(0), 
                                  scale_factor=self.common_stride, 
                                  mode="bilinear", 
                                  align_corners=False)[...,:img_shape[0],:img_shape[1]]
        pred_inst = F.interpolate(pred_inst, 
                                  size=ori_shape, 
                                  mode="bilinear", 
                                  align_corners=False)[0]

        pred_bitinst = BitMasks(pred_inst > self.inst_thres)
        pred_box = pred_bitinst.get_bounding_boxes()
        result_instance = Instances(ori_shape,
                                    pred_masks=pred_bitinst,
                                    pred_boxes=pred_box,
                                    pred_classes=classes,
                                    scores=scores)
        return result_instance, pred_inst, classes, scores, depth_kernels

    @torch.no_grad()
    def combine_thing_and_stuff(
        self,
        thing_results,
        stuff_results,
        overlap_threshold,
        stuff_area_limit,
        inst_threshold,
        encode_depth,
        with_void = False,
    ):
        """
        """
        pred_thing, thing_cate, thing_score, depth_kernels = thing_results
        stuff_results_logits, depth_kernel_sts, class_sts, _ = stuff_results
        stuff_results = stuff_results_logits.argmax(dim=0)

        panoptic_seg = torch.zeros_like(stuff_results, dtype=torch.int32)
        semantic_seg = torch.ones_like(stuff_results, dtype=torch.int32) * self.meta.ignore_label

        panoptic_logits = None
        depth_cat = None
        panoptic_cats = []
        panoptic_ids = []

        pred_depth   = torch.zeros_like(stuff_results, dtype=torch.float32) if self.depth_on else None
        current_segment_id = self.sem_classes+1
        segments_info = []
        if thing_cate is not None:
            keep = thing_score >= inst_threshold
            if keep.sum() > 0:
                pred_thing = pred_thing[keep]
                thing_cate = thing_cate[keep]
                thing_score = thing_score[keep]

                if self.depth_on:
                    depth_kernels = depth_kernels[keep]
                    depth_th_map = self.depth_map_generator(encode_depth.unsqueeze(0), depth_kernels.unsqueeze(1).unsqueeze(0)).squeeze(0).squeeze(1)
                else:
                    depth_th_map = None
                img_h, img_w = stuff_results.shape

                # Add instances one-by-one, check for overlaps with existing ones
                for _idx, (_mask_logit, _cate, _score) in enumerate(zip(pred_thing, thing_cate, thing_score)):
                    _mask = _mask_logit >self.inst_thres
                    mask_area = _mask.sum().item()
                    intersect = _mask & (panoptic_seg > 0)
                    intersect_area = intersect.sum().item()
                    if mask_area==0 or intersect_area * 1.0 / mask_area > overlap_threshold:
                        continue
                    if intersect_area > 0:
                        _mask = _mask & (panoptic_seg == 0)
                    current_segment_id += 1
                    panoptic_seg[_mask] = current_segment_id
                    thing_category_id = _cate.item()
                    category_id = self.meta.thing_train_id2contiguous_id[thing_category_id]
                    semantic_seg[_mask] = category_id
                    if self.depth_on and with_void:
                        pred_depth[_mask] = depth_th_map[_idx][_mask]

                    if panoptic_logits is None:
                        panoptic_logits = _mask_logit.unsqueeze(0)
                        depth_cat = depth_th_map[_idx].unsqueeze(0) if self.depth_on else None
                    else:
                        panoptic_logits = torch.cat([panoptic_logits, _mask_logit.unsqueeze(0)], dim=0)
                        depth_cat = torch.cat([depth_cat, depth_th_map[_idx].unsqueeze(0)], dim=0) if self.depth_on else None

                    panoptic_cats.append(category_id)
                    panoptic_ids.append(current_segment_id)


        if self.depth_on:
            depth_st_map = self.depth_map_generator(encode_depth.unsqueeze(0), depth_kernel_sts.unsqueeze(1).unsqueeze(0)).squeeze(0).squeeze(1)
        else:
            depth_st_map = None

        stuff_labels = torch.unique(stuff_results)
        for stuff_label in stuff_labels:
            stuff_category_id = stuff_label.item()
            category_id = self.meta.stuff_train_id2contiguous_id[stuff_category_id]
            if self.sem_with_thing:
                if self.sem_all_classes:
                    if category_id in self.meta.thing_train_id2contiguous_id.values():
                        continue
                else:
                    if stuff_label == 0:  # 0 is a special "thing" class
                        continue
            _mask = (stuff_results == stuff_label) & (panoptic_seg == 0)
            mask_area = _mask.sum()
            if mask_area < stuff_area_limit:
                continue
            panoptic_seg[_mask] = stuff_category_id+1
            semantic_seg[_mask] = category_id
            _idx = class_sts == stuff_label

            if panoptic_logits is None:
                panoptic_logits = stuff_results_logits[stuff_label].unsqueeze(0)
                depth_cat = depth_st_map[_idx].mean(dim=0).unsqueeze(0) if self.depth_on else None
            else:
                panoptic_logits = torch.cat([panoptic_logits, stuff_results_logits[stuff_label].unsqueeze(0)], dim=0)
                depth_cat = torch.cat([depth_cat, depth_st_map[_idx].mean(dim=0).unsqueeze(0)], dim=0) if self.depth_on else None
            panoptic_cats.append(category_id)
            panoptic_ids.append(stuff_category_id+1)

        if with_void:
            panoptic_seg[semantic_seg==self.meta.ignore_label] = -1
            semantic_seg[semantic_seg==self.meta.ignore_label] = 0
            return (panoptic_seg, semantic_seg), pred_depth
        else:
            panoptic_cats = torch.Tensor(panoptic_cats).to(self.device)
            panoptic_ids  = torch.Tensor(panoptic_ids ).to(self.device)
            panoptic_seg = panoptic_logits.argmax(dim=0)
            pred_depth = torch.gather(depth_cat, dim=0, index=panoptic_seg.unsqueeze(0)).squeeze(0) if self.depth_on else None
            semantic_seg = panoptic_cats[panoptic_seg]
            panoptic_seg = panoptic_ids[panoptic_seg]
        return (panoptic_seg, semantic_seg), pred_depth
