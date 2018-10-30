import random
import sys

if sys.version_info[0] == 2:
    from pathlib2 import Path
else:
    from pathlib import Path


import numpy as np
import cv2

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

from torch.utils.data import DataLoader

import vision_transforms as vt
vt.set_image_backend("opencv")

from datasets import VOCDetection, TransformedDataset, to_target


def get_dataloader(path, mode, dtf, coder, batch_size=8, num_workers=8):
    assert Path(path).exists(), "Path '{}' is not found".format(path)
    assert mode in ("train", "trainval", "val")
    assert callable(dtf)
    assert hasattr(coder, "encode")
    voc_ds = VOCDetection(path, mode)

    def _xy_transform(x, y):
        x, y = to_target(x, y)
        x, y = dtf((x, y), random.getstate())
        y = coder.encode(*y)

    det_aug_ds = TransformedDataset(voc_ds, xy_transform=_xy_transform)
    dataloader = DataLoader(det_aug_ds,
                            batch_size=batch_size,
                            shuffle=True,
                            num_workers=num_workers,
                            drop_last=True)
    return dataloader


class DataAugTransform(vt.BaseTransform):

    def __init__(self):

        translate_scale_params = {
            'translate': (0.2, 0.2),
            'scale': (0.7, 1.3)
        }
        self.random_affine = vt.RandomAffine(degrees=0, **translate_scale_params, resample=cv2.INTER_CUBIC)
        self.bbox_random_affine = vt.BBoxRandomAffine(**translate_scale_params)

        self.random_crop = vt.RandomCrop(size=300, padding=50)
        self.bbox_random_crop = vt.BBoxRandomCrop(size=300, padding=50)

        self.img_geom = vt.Sequential(
            self.random_affine,
            self.random_crop,
        )
        self.bbox_geom = vt.Sequential(
            self.bbox_random_affine,
            self.bbox_random_crop,
        )
        self.img_color = vt.ColorJitter(hue=0.1, saturation=0.2)

    def __call__(self, datapoint, rng):

        x, y = datapoint
        img_rgb = x
        bboxes, labels = list(zip(*y))
        bboxes = np.array(bboxes)

        t_img_rgb = self.img_geom(img_rgb, rng)
        t_img_rgb = self.img_color(t_img_rgb)
        t_bboxes = self.bbox_geom(bboxes, rng, input_canvas_size=img_rgb.shape[:2])

        # to CHW normalized float32:
        x = np.transpose(t_img_rgb, (2, 0, 1))
        x = x.astype(np.float32) / 255.0
        x -= 0.5

        y = (t_bboxes, np.array(labels, dtype=np.long))
        return x, y
