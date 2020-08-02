from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import logging
import random

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from utils.transforms import get_affine_transform
from utils.transforms import affine_transform
from utils.transforms import fliplr_coord
from utils.transforms import img_crop_with_pad
from utils.transforms import crop_and_resize


logger = logging.getLogger(__name__)


class FoveaDataset(Dataset):
    def __init__(self, cfg, root, image_set, is_train, transform=None):
        self.is_train = is_train
        self.root = root
        self.image_set = image_set

        self.output_path = cfg.OUTPUT_DIR

        self.rotation_factor = cfg.DATASET.ROT_FACTOR
        self.shift_factor = cfg.DATASET.SHIFT_FACTOR
        self.flip = cfg.DATASET.FLIP
        self.scale_factor = cfg.DATASET.SCALE_FACTOR

        self.target_type = cfg.MODEL.TARGET_TYPE
        self.image_size = np.array(cfg.MODEL.IMAGE_SIZE)
        self.crop_size = np.array(cfg.MODEL.CROP_SIZE)
        self.patch_size = np.array(cfg.MODEL.PATCH_SIZE)
        self.ds_factor = np.array(cfg.MODEL.DS_FACTOR)
        self.sigma = cfg.MODEL.SIGMA
        self.max_ds_offset = cfg.MODEL.MAX_DS_OFFSET
        self.max_offset = cfg.MODEL.MAX_OFFSET
        self.region_radius = cfg.MODEL.REGION_RADIUS

        self.transform = transform
        self.db = []

    def evaluate(self, preds, output_dir):
        raise NotImplementedError

    def _get_db(self,):
        raise NotImplementedError

    def __len__(self,):
        return len(self.db)

    def __getitem__(self, idx):
        db_rec = copy.deepcopy(self.db[idx])

        # load data
        image_file = db_rec['image']
        data_numpy = cv2.imread(
            image_file, cv2.IMREAD_COLOR
        )
        if data_numpy is None:
            logger.error('=> fail to read {}'.format(image_file))
            raise ValueError('Fail to read {}'.format(image_file))
        if 'fovea' in db_rec.keys():
            fovea = np.array(db_rec['fovea'])
        else:
            fovea = np.array([-1, -1])

        dh, dw = data_numpy.shape[:2]
        if dh != self.image_size[1] or dw != self.image_size[0]:
            data_numpy = cv2.resize(data_numpy, dsize=(self.image_size[0], self.image_size[1]), interpolation=cv2.INTER_LINEAR)
            h_ratio = self.image_size[1] * 1.0 / dh
            w_ratio = self.image_size[0] * 1.0 / dw
            fovea[0] *= w_ratio
            fovea[1] *= h_ratio

        if self.is_train:
            if self.scale_factor > 0 and np.random.rand() > 0.5:
                sign = 1 if np.random.rand() > 0.5 else -1
                scale_factor = 1.0 + np.random.rand() * self.scale_factor * sign
                dh, dw = data_numpy.shape[:2]
                nh, nw = int(dh * scale_factor), int(dw * scale_factor)
                data_numpy = cv2.resize(data_numpy, dsize=(nw, nh), interpolation=cv2.INTER_LINEAR)
                fovea[0] *= (nw * 1.0 / dw)
                fovea[1] *= (nh * 1.0 / dh)
                if sign > 0: # crop
                    ph = (nh - self.image_size[1]) // 2
                    pw = (nw - self.image_size[0]) // 2
                    data_numpy = data_numpy[ph:ph+self.image_size[1], pw:pw+self.image_size[0], :]
                    fovea[0] -= pw
                    fovea[1] -= ph
                else: # pad
                    ph = (self.image_size[1] - nh) // 2
                    pw = (self.image_size[0] - nw) // 2
                    data_numpy = np.pad(data_numpy, ((ph, self.image_size[1]-nh-ph),
                                                     (pw, self.image_size[0]-nw-pw), (0, 0)), mode='constant')
                    fovea[0] += pw
                    fovea[1] += ph

        image_size = self.image_size
        # crop image from center
        crop_size = self.crop_size
        pw = (image_size[0] - crop_size[0]) // 2
        ph = (image_size[1] - crop_size[1]) // 2
        data_numpy = data_numpy[ph:ph+crop_size[1], pw:pw+crop_size[0], :]
        image_size = crop_size
        fovea[0] -= pw
        fovea[1] -= ph

        # get image transform for augmentation
        c = image_size * 0.5
        r = 0
        s = 0

        if self.is_train:
            rf = self.rotation_factor
            sf = self.shift_factor
            sign = 1 if np.random.randn() > 0.5 else -1
            r = np.clip(sign*np.random.randn()*rf, -rf*2, rf*2) \
                if random.random() <= 0.6 else 0
            sign = 1 if np.random.randn() > 0.5 else -1
            s = sign * np.random.rand() * sf

            if self.flip and random.random() <= 0.5:
                data_numpy = data_numpy[:, ::-1, :]
                fovea = fliplr_coord(fovea, data_numpy.shape[1])
                c[0] = data_numpy.shape[1] - c[0] - 1

        trans = get_affine_transform(c, r, image_size, shift=s)
        input = cv2.warpAffine(
            data_numpy,
            trans,
            (int(image_size[0]), int(image_size[1])),
            flags=cv2.INTER_LINEAR)

        fovea = affine_transform(fovea, trans)

        if self.is_train:
            patch_size = self.patch_size.astype(np.int32)
            pw = np.random.randint(0, int(image_size[0] - patch_size[0] + 1))
            ph = np.random.randint(0, int(image_size[1] - patch_size[1] + 1))
            input = input[ph:ph+patch_size[1], pw:pw+patch_size[0], :]
            fovea[0] -= pw
            fovea[1] -= ph

        if self.transform:
            input = self.transform(input)

        meta = {'fovea': fovea,
                'image': image_file}

        if self.is_train:
            heatmap_ds, heatmap_roi, roi_center, pixel_in_roi, offset_in_roi, fovea, fovea_in_roi, target_weight = \
                self.generate_target(input, fovea)

            # crop ROI
            input_roi = crop_and_resize(input.unsqueeze(0),
                                        torch.from_numpy(roi_center).unsqueeze(0),
                                        output_size=2*self.region_radius, scale=1.0)[0]

            heatmap_ds = torch.from_numpy(heatmap_ds).float()
            heatmap_roi = torch.from_numpy(heatmap_roi).float()

            roi_center = torch.from_numpy(roi_center).float()
            pixel_in_roi = torch.from_numpy(pixel_in_roi).float()
            offset_in_roi = torch.from_numpy(offset_in_roi).float()
            fovea = torch.from_numpy(fovea).float()
            fovea_in_roi = torch.from_numpy(fovea_in_roi).float()

            meta.update({
                'roi_center': roi_center,
                'pixel_in_roi': pixel_in_roi,
                'fovea_in_roi': fovea_in_roi
            })

            return input, input_roi, heatmap_ds, heatmap_roi, offset_in_roi, target_weight, meta
        else:
            return input, meta



    def generate_target(self, image, fovea):
        assert self.target_type == 'gaussian', \
            'Only support gaussian map now!'

        if self.target_type == 'gaussian':
            if self.is_train:
                image_size = self.patch_size
                image_ds_size = self.patch_size / self.ds_factor
            else:
                image_size = self.crop_size
                image_ds_size = self.crop_size / self.ds_factor
            image_size = image_size.astype(np.int32)
            image_ds_size = image_ds_size.astype(np.int32)
            heatmap_ds = np.zeros((1,
                                   image_ds_size[1],
                                   image_ds_size[0]),
                                   dtype=np.float32)
            target_weight = np.array([1.], np.float32)

            tmp_size = self.sigma * 3

            feat_stride = image_size / image_ds_size
            mu_x = int(fovea[0] / feat_stride[0] + 0.5)
            mu_y = int(fovea[1] / feat_stride[1] + 0.5)
            # Check that any part of the gaussian is in-bounds
            ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
            br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
            if ul[0] >= image_ds_size[0] or ul[1] >= image_ds_size[1] \
                    or br[0] < 0 or br[1] < 0 or fovea[0] < 0 or fovea[1] < 0:
                # If not, just return the image as is
                target_weight = np.array([0.], np.float32)
                region_size = 2 * self.region_radius
                heatmap_roi = np.zeros((1, region_size, region_size), dtype=np.float32)
                roi_center = np.array([-1, -1], np.float32)
                pixel_in_roi = np.array([-1, -1], np.float32)
                offset_in_roi = np.array([-1, -1], np.float32)
                fovea_in_roi = np.array([-1, -1], np.float32)
                return heatmap_ds, heatmap_roi, roi_center, pixel_in_roi, offset_in_roi, fovea, fovea_in_roi, target_weight

            # # Generate gaussian
            size = 2 * tmp_size + 1
            x = np.arange(0, size, 1, np.float32)
            y = x[:, np.newaxis]
            x0 = y0 = size // 2
            # The gaussian is not normalized, we want the center value to equal 1
            g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * self.sigma ** 2))

            # Usable gaussian range
            g_x = max(0, -ul[0]), min(br[0], image_ds_size[0]) - ul[0]
            g_y = max(0, -ul[1]), min(br[1], image_ds_size[1]) - ul[1]
            # Image range
            img_x = max(0, ul[0]), min(br[0], image_ds_size[0])
            img_y = max(0, ul[1]), min(br[1], image_ds_size[1])

            heatmap_ds[0][img_y[0]:img_y[1], img_x[0]:img_x[1]] = \
                g[g_y[0]:g_y[1], g_x[0]:g_x[1]]

            # sample a noisily centered ROI region for high-resolution target
            offset = self.max_ds_offset
            region_size = self.region_radius * 2
            sign_x = 1 if np.random.rand() > 0.5 else -1
            sign_y = 1 if np.random.rand() > 0.5 else -1
            ox = np.random.rand() * offset
            oy = np.random.rand() * offset
            cx = np.clip(sign_x * ox + fovea[0] / feat_stride[0], 0, image_size[0] - 1)
            cy = np.clip(sign_y * oy + fovea[1] / feat_stride[1], 0, image_size[1] - 1)
            cx = (cx * feat_stride[0]).astype(np.int32)
            cy = (cy * feat_stride[1]).astype(np.int32)

            # get fovea location in this ROI
            fovea_in_roi = fovea.copy()
            fovea_in_roi[0] -= (cx - self.region_radius)
            fovea_in_roi[1] -= (cy - self.region_radius)

            # Re-apply the target generation process
            image_size = np.array([region_size, region_size], np.int32)
            heatmap_roi = np.zeros((1,
                                   region_size,
                                   region_size),
                                   dtype=np.float32)
            target_weight = np.array([1.], np.float32)

            tmp_size = self.sigma * 3

            mu_x = int(fovea_in_roi[0] + 0.5)
            mu_y = int(fovea_in_roi[1] + 0.5)
            # Check that any part of the gaussian is in-bounds
            ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
            br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
            if ul[0] >= region_size or ul[1] >= region_size \
                    or br[0] < 0 or br[1] < 0 or fovea_in_roi[0] < 0 or fovea_in_roi[1] < 0:
                # If not, just return the image as is
                target_weight = np.array([0.], np.float32)
                heatmap_roi = np.zeros((1, region_size, region_size), dtype=np.float32)
                roi_center = np.array([-1, -1], np.float32)
                pixel_in_roi = np.array([-1, -1], np.float32)
                offset_in_roi = np.array([-1, -1], np.float32)
                fovea_in_roi = np.array([-1, -1], np.float32)
                return heatmap_ds, heatmap_roi, roi_center, pixel_in_roi, offset_in_roi, fovea, fovea_in_roi, target_weight

            # # Generate gaussian
            size = 2 * tmp_size + 1
            x = np.arange(0, size, 1, np.float32)
            y = x[:, np.newaxis]
            x0 = y0 = size // 2
            # The gaussian is not normalized, we want the center value to equal 1
            g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * self.sigma ** 2))

            # Usable gaussian range
            g_x = max(0, -ul[0]), min(br[0], region_size) - ul[0]
            g_y = max(0, -ul[1]), min(br[1], region_size) - ul[1]
            # Image range
            img_x = max(0, ul[0]), min(br[0], region_size)
            img_y = max(0, ul[1]), min(br[1], region_size)

            heatmap_roi[0][img_y[0]:img_y[1], img_x[0]:img_x[1]] = \
                g[g_y[0]:g_y[1], g_x[0]:g_x[1]]

            roi_center = np.array([cx, cy], np.float32)

            # sample a noisy prediction for offset regression
            offset = self.max_offset
            sign_x = 1 if np.random.rand() > 0.5 else -1
            sign_y = 1 if np.random.rand() > 0.5 else -1
            ox = np.random.rand() * offset
            oy = np.random.rand() * offset
            cx = np.clip(sign_x * ox + fovea_in_roi[0], 0, region_size - 1)
            cy = np.clip(sign_y * oy + fovea_in_roi[1], 0, region_size - 1)

            offset_in_roi = np.array([fovea_in_roi[0] - cx, fovea_in_roi[1] - cy], np.float32)
            pixel_in_roi = np.array([cx, cy], np.float32)
            offset_in_roi /= region_size

        return heatmap_ds, heatmap_roi, roi_center, pixel_in_roi, offset_in_roi, fovea, fovea_in_roi, target_weight

