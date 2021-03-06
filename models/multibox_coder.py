# Adapted from https://github.com/chainer/chainercv/blob/master/chainercv/links/model/ssd/multibox_coder.py
from __future__ import division

import itertools
import numpy as np

from utils import bbox_iou, non_maximum_suppression


class MultiboxCoder(object):
    """A helper class to encode/decode bounding boxes.
    This class encodes :obj:`(bbox, label)` to :obj:`(mb_loc, mb_label)`
    and decodes :obj:`(mb_loc, mb_conf)` to :obj:`(bbox, label, score)`.
    These encoding/decoding are used in Single Shot Multibox Detector [#]_.
    * :obj:`mb_loc`: An array representing offsets and scales \
         from the default bounding boxes. \
         Its shape is :math:`(K, 4)`, where :math:`K` is the number of \
         the default bounding boxes. \
         The second axis is composed by \
         :math:`(\Delta x, \Delta y, \Delta w, \Delta h)`. \
         These values are computed by the following formulas.
         * :math:`\Delta x = (b_x - m_x) / (m_w * v_0)`
         * :math:`\Delta y = (b_y - m_y) / (m_h * v_0)`
         * :math:`\Delta w = log(b_w / m_w) / v_1`
         * :math:`\Delta h = log(b_h / m_h) / v_1`
         :math:`(m_x, m_y)` and :math:`(m_w, m_h)` are \
         center coodinates and size of a default bounding box. \
         :math:`(b_x, b_y)` and :math:`(b_w, b_h)` are \
         center coodinates and size of \
         a given bounding boxes that is assined to the default bounding box. \
         :math:`(v_0, v_1)` are coefficients that can be set \
         by argument :obj:`variance`.
    * :obj:`mb_label`: An array representing classes of \
         ground truth bounding boxes. Its shape is :math:`(K,)`.
    * :obj:`mb_conf`: An array representing classes of \
         predicted bounding boxes. Its shape is :math:`(K, n\_fg\_class + 1)`.
    .. [#] Wei Liu, Dragomir Anguelov, Dumitru Erhan, Christian Szegedy,
       Scott Reed, Cheng-Yang Fu, Alexander C. Berg.
       SSD: Single Shot MultiBox Detector. ECCV 2016.
    Args:
        grids (iterable of ints): An iterable of integers.
            Each integer indicates the size of a feature map.
        aspect_ratios (iterable of tuples of ints):
            An iterable of tuples of integers
            used to compute the default bouding boxes.
            Each tuple indicates the aspect ratios of
            the default bounding boxes at each feature maps.
            The length of this iterable should be :obj:`len(grids)`.
        steps (iterable of floats): The step size for each feature map.
            The length of this iterable should be :obj:`len(grids)`.
        sizes (iterable of floats): The base size of default bounding boxes
            for each feature map.
            The length of this iterable should be :obj:`len(grids) + 1`.
        variance (tuple of floats): Two coefficients for encoding/decoding
            the locations of bounding boxes. The first value is used to
            encode/decode coordinates of the centers.
            The second value is used to encode/decode the sizes of
            bounding boxes.
    """

    def __init__(self, grids, aspect_ratios, steps, sizes, variance):
        if not len(aspect_ratios) == len(grids):
            raise ValueError('The length of aspect_ratios is wrong.')
        if not len(steps) == len(grids):
            raise ValueError('The length of steps is wrong.')
        if not len(sizes) == len(grids) + 1:
            raise ValueError('The length of sizes is wrong.')

        default_bbox = []

        for k, grid in enumerate(grids):
            for u, v in itertools.product(range(grid), repeat=2):
                cx = (u + 0.5) * steps[k]
                cy = (v + 0.5) * steps[k]

                s = sizes[k]
                default_bbox.append((cx, cy, s, s))

                s = np.sqrt(sizes[k] * sizes[k + 1])
                default_bbox.append((cx, cy, s, s))

                s = sizes[k]
                for ar in aspect_ratios[k]:
                    default_bbox.append(
                        (cx, cy, s / np.sqrt(ar), s * np.sqrt(ar)))
                    default_bbox.append(
                        (cx, cy, s * np.sqrt(ar), s / np.sqrt(ar)))

        # (center_x, center_y, width, height)
        self._default_bbox = np.stack(default_bbox)
        self._variance = variance

    def encode(self, bbox, label, iou_thresh=0.5):
        """Encodes coordinates and classes of bounding boxes.
        This method encodes :obj:`bbox` and :obj:`label` to :obj:`mb_loc`
        and :obj:`mb_label`, which are used to compute multibox loss.
        Args:
            bbox (array): A float array of shape :math:`(R, 4)`,
                where :math:`R` is the number of bounding boxes in an image.
                Each bouding box is organized by
                :math:`(x_{min}, y_{min}, x_{max}, y_{max})`
                in the second axis.
            label (array) : An integer array of shape :math:`(R,)`.
                Each value indicates the class of the bounding box.
            iou_thresh (float): The threshold value to determine
                a default bounding box is assigned to a ground truth
                or not. The default value is :obj:`0.5`.
        Returns:
            tuple of two arrays:
            This method returns a tuple of two arrays,
            :obj:`(mb_loc, mb_label)`.
            * **mb_loc**: A float array of shape :math:`(K, 4)`, \
                where :math:`K` is the number of default bounding boxes.
            * **mb_label**: An integer array of shape :math:`(K,)`.
        """
        if len(bbox) == 0:
            return (
                np.zeros(self._default_bbox.shape, dtype=np.float32),
                np.zeros(self._default_bbox.shape[0], dtype=np.int32))


        iou = bbox_iou(
            np.hstack((
                self._default_bbox[:, :2] - self._default_bbox[:, 2:] / 2,
                self._default_bbox[:, :2] + self._default_bbox[:, 2:] / 2)),
            bbox)

        index = np.empty(len(self._default_bbox), dtype=int)
        # -1 is for background
        index[:] = -1

        masked_iou = iou.copy()
        while True:
            i, j = np.unravel_index(masked_iou.argmax(), masked_iou.shape)
            if masked_iou[i, j] <= 1e-6:
                break
            index[i] = j
            masked_iou[i, :] = 0
            masked_iou[:, j] = 0

        mask = np.logical_and(index < 0, iou.max(axis=1) >= iou_thresh)
        index[mask] = iou[mask].argmax(axis=1)

        mb_bbox = bbox[index].copy()
        # (x_min, y_min, x_max, y_max) -> (x_min, y_min, width, height)
        mb_bbox[:, 2:] -= mb_bbox[:, :2]
        # (x_min, y_min, width, height) -> (center_x, center_y, width, height)
        mb_bbox[:, :2] += mb_bbox[:, 2:] / 2

        mask = (mb_bbox[:, 2:] > 0).all(axis=1)
        mb_loc = np.zeros_like(mb_bbox)
        mb_loc[mask, :2] = (mb_bbox[mask, :2] - self._default_bbox[mask, :2]) / \
            (self._variance[0] * self._default_bbox[mask, 2:])
        mb_loc[mask, 2:] = np.log(mb_bbox[mask, 2:] / self._default_bbox[mask, 2:]) / \
            self._variance[1]

        # [0, n_fg_class - 1] -> [1, n_fg_class]
        mb_label = label[index] + 1
        # 0 is for background
        mb_label[index < 0] = 0

        return mb_loc.astype(np.float32), mb_label.astype(np.long)

    def decode(self, mb_loc, mb_conf, nms_thresh=0.45, score_thresh=0.6):
        """Decodes back to coordinates and classes of bounding boxes.
        This method decodes :obj:`mb_loc` and :obj:`mb_conf` returned
        by a SSD network back to :obj:`bbox`, :obj:`label` and :obj:`score`.
        Args:
            mb_loc (array): A float array whose shape is
                :math:`(K, 4)`, :math:`K` is the number of
                default bounding boxes.
            mb_conf (array): A float array whose shape is
                :math:`(K, n\_fg\_class + 1)`.
            nms_thresh (float): The threshold value
                for :func:`~utils.non_maximum_suppression`.
                The default value is :obj:`0.45`.
            score_thresh (float): The threshold value for confidence score.
                If a bounding box whose confidence score is lower than
                this value, the bounding box will be suppressed.
                The default value is :obj:`0.6`.
        Returns:
            tuple of three arrays:
            This method returns a tuple of three arrays,
            :obj:`(bbox, label, score)`.
            * **bbox**: A float array of shape :math:`(R, 4)`, \
                where :math:`R` is the number of bounding boxes in a image. \
                Each bouding box is organized by \
                :math:`(x_{min}, y_{min}, x_{max}, y_{max})` \
                in the second axis.
            * **label** : An integer array of shape :math:`(R,)`. \
                Each value indicates the class of the bounding box.
            * **score** : A float array of shape :math:`(R,)`. \
                Each value indicates how confident the prediction is.
        """
        # (center_x, center_y, width, height)
        mb_bbox = self._default_bbox.copy()
        mb_bbox[:, :2] += mb_loc[:, :2] * self._variance[0] \
            * self._default_bbox[:, 2:]
        mb_bbox[:, 2:] *= np.exp(mb_loc[:, 2:] * self._variance[1])

        # (center_x, center_y, width, height) -> (x_min, y_min, width, height)
        mb_bbox[:, :2] -= mb_bbox[:, 2:] / 2
        # (x_min, y_min, width, height) -> (x_min, y_min, x_max, y_max)
        mb_bbox[:, 2:] += mb_bbox[:, :2]

        # softmax
        mb_score = np.exp(mb_conf)
        mb_score /= mb_score.sum(axis=1, keepdims=True)

        bbox = []
        label = []
        score = []
        for l in range(mb_conf.shape[1] - 1):
            bbox_l = mb_bbox
            # the l-th class corresponds for the (l + 1)-th column.
            score_l = mb_score[:, l + 1]

            mask = score_l >= score_thresh
            bbox_l = bbox_l[mask]
            score_l = score_l[mask]

            if nms_thresh is not None:
                indices = non_maximum_suppression(bbox_l, nms_thresh, score_l)
                bbox_l = bbox_l[indices]
                score_l = score_l[indices]

            bbox.append(bbox_l)
            label.append(np.array((l,) * len(bbox_l)))
            score.append(score_l)

        bbox = np.vstack(bbox).astype(np.float32)
        label = np.hstack(label).astype(np.int32)
        score = np.hstack(score).astype(np.float32)

        return bbox, label, score
