# Copied from https://github.com/chainer/chainercv/blob/master/chainercv/links/model/ssd/multibox_loss.py
from __future__ import division

import numpy as np

import torch
import torch.nn.functional as F


def _hard_negative(x, positive, k):
    _, idx = (x * (positive.type_as(x) - 1)).sort(dim=1)
    _, rank = idx.sort(dim=1)
    hard_negative = rank < (positive.sum(dim=1) * k)[:, None]
    return hard_negative


def multibox_loss(mb_locs, mb_confs, gt_mb_locs, gt_mb_labels, k):
    """Computes multibox losses.
    This is a loss function used in [#]_.
    This function returns :obj:`loc_loss` and :obj:`conf_loss`.
    :obj:`loc_loss` is a loss for localization and
    :obj:`conf_loss` is a loss for classification.
    The formulas of these losses can be found in
    the equation (2) and (3) in the original paper.
    .. [#] Wei Liu, Dragomir Anguelov, Dumitru Erhan,
       Christian Szegedy, Scott Reed, Cheng-Yang Fu, Alexander C. Berg.
       SSD: Single Shot MultiBox Detector. ECCV 2016.
    Args:
        mb_locs (tensor): The offsets and scales
            for predicted bounding boxes.
            Its shape is :math:`(B, K, 4)`,
            where :math:`B` is the number of samples in the batch and
            :math:`K` is the number of default bounding boxes.
        mb_confs (tensor): The classes of predicted
            bounding boxes.
            Its shape is :math:`(B, K, n\_class)`.
            This function assumes the first class is background (negative).
        gt_mb_locs (tensor): The offsets and scales
            for ground truth bounding boxes.
            Its shape is :math:`(B, K, 4)`.
        gt_mb_labels (tensor): The classes of ground truth
            bounding boxes.
            Its shape is :math:`(B, K)`.
        k (float): A coefficient which is used for hard negative mining.
            This value determines the ratio between the number of positives
            and that of mined negatives. The value used in the original paper
            is :obj:`3`.

    Returns:
        tuple of tensors:
        This function returns two tensors :obj:`loc_loss` and
        :obj:`conf_loss`.
    """

    positive = gt_mb_labels > 0
    n_positive = positive.sum().item()

    if n_positive == 0:
        z = torch.zeros(())
        return z, z

    loc_loss = F.smooth_l1_loss(mb_locs, gt_mb_locs, reduction='none').sum(dim=-1)
    loc_loss *= positive.type_as(loc_loss)
    loc_loss = torch.sum(loc_loss) / n_positive

    conf_loss = F.cross_entropy(mb_confs.reshape(-1, mb_confs.shape[-1]),
                                torch.flatten(gt_mb_labels),
                                reduction='none').reshape(mb_confs.shape[0], -1)
    hard_negative = _hard_negative(conf_loss, positive, k)

    conf_loss *= (positive | hard_negative).type_as(conf_loss)
    conf_loss = torch.sum(conf_loss) / n_positive

    return loc_loss, conf_loss
