from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn


class HeatmapMSELoss(nn.Module):
    def __init__(self, use_target_weight):
        super(HeatmapMSELoss, self).__init__()
        self.criterion = nn.MSELoss(size_average=True, reduce=True)
        self.use_target_weight = use_target_weight

    def forward(self, output, target, target_weight):
        batch_size = output.size(0)
        num_joints = output.size(1)
        heatmaps_pred = output.reshape((batch_size, 1, -1)).split(1, 1)
        heatmaps_gt = target.reshape((batch_size, 1, -1)).split(1, 1)
        loss = 0

        heatmap_pred = heatmaps_pred[0].squeeze()
        heatmap_gt = heatmaps_gt[0].squeeze()
        if self.use_target_weight:
            loss += 0.5 * self.criterion(
                heatmap_pred.mul(target_weight),
                heatmap_gt.mul(target_weight)
            )
        else:
            loss += 0.5 * self.criterion(heatmap_pred, heatmap_gt)

        return loss



class HybridLoss(nn.Module):
    def __init__(self, roi_weight, regress_weight, use_target_weight):
        super(HybridLoss, self).__init__()
        self.heatmap_mse = HeatmapMSELoss(use_target_weight)
        self.smooth_l1 = nn.SmoothL1Loss(size_average=True, reduce=True)
        self.roi_weight = roi_weight
        self.regress_weight = regress_weight

    def forward(self, pred_ds, target_ds, pred_roi, target_roi, pred_offset, target_offset, target_weight):
        heatmap_ds_loss = self.heatmap_mse(pred_ds, target_ds, target_weight)
        heatmap_roi_loss = self.heatmap_mse(pred_roi, target_roi, target_weight)

        pred_offset = pred_offset.mul(target_weight)
        target_offset = target_offset.mul(target_weight)
        regress_loss = self.smooth_l1(pred_offset, target_offset)

        heatmap_roi_loss = heatmap_roi_loss * self.roi_weight
        regress_loss = regress_loss * self.regress_weight
        hybrid_loss = heatmap_ds_loss + heatmap_roi_loss + regress_loss

        return hybrid_loss, {'heatmap_ds': heatmap_ds_loss,
                             'heatmap_roi': heatmap_roi_loss,
                             'offset': regress_loss}



