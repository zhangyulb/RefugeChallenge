from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import numpy as np
import torch
import torch.nn.functional as F

from utils.transforms import transform_preds


def get_max_preds(batch_heatmaps):
    '''
    get predictions from score maps
    heatmaps: numpy.ndarray([batch_size, 1, height, width])
    '''
    assert isinstance(batch_heatmaps, np.ndarray), \
        'batch_heatmaps should be numpy.ndarray'
    assert batch_heatmaps.ndim == 4, 'batch_images should be 4-ndim'

    batch_size = batch_heatmaps.shape[0]
    num_joints = batch_heatmaps.shape[1]
    width = batch_heatmaps.shape[3]
    heatmaps_reshaped = batch_heatmaps.reshape((batch_size, 1, -1))
    idx = np.argmax(heatmaps_reshaped, 2)
    maxvals = np.amax(heatmaps_reshaped, 2)

    maxvals = maxvals.reshape((batch_size, 1, 1))
    idx = idx.reshape((batch_size, 1, 1))

    preds = np.tile(idx, (1, 1, 2)).astype(np.float32)

    preds[:, :, 0] = (preds[:, :, 0]) % width
    preds[:, :, 1] = np.floor((preds[:, :, 1]) / width)

    pred_mask = np.tile(np.greater(maxvals, 0.0), (1, 1, 2))
    pred_mask = pred_mask.astype(np.float32)

    preds *= pred_mask
    return preds, maxvals


def get_final_preds(config, batch_heatmap_ds, batch_heatmap_roi, offsets_in_roi, meta):
    coords_ds, maxvals_ds = get_max_preds(batch_heatmap_ds)
    coords_roi, maxvals_roi = get_max_preds(batch_heatmap_roi)
    region_size = 2 * config.MODEL.REGION_RADIUS
    offsets_in_roi = offsets_in_roi * region_size
    # coords: [N, 1, 2] -> [N, 2]
    coords_ds = coords_ds[:, 0, :]
    coords_roi = coords_roi[:, 0, :]
    coords_lr = coords_ds * config.MODEL.DS_FACTOR
    coords_hr = coords_roi + meta['roi_center'].cpu().numpy() - config.MODEL.REGION_RADIUS
    coords_final = coords_hr + offsets_in_roi
    coords_roi_final = coords_roi + offsets_in_roi
    return coords_lr, coords_hr, coords_final, coords_roi, coords_roi_final


