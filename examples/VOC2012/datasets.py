import sys

if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
    from pathlib2 import Path
else:
    import xml.etree.ElementTree as ET
    from pathlib import Path

import numpy as np
import cv2

from torch.utils.data import Dataset


VOC_CLASSES = (  # always index 0
    'aeroplane', 'bicycle', 'bird', 'boat',
    'bottle', 'bus', 'car', 'cat', 'chair',
    'cow', 'diningtable', 'dog', 'horse',
    'motorbike', 'person', 'pottedplant',
    'sheep', 'sofa', 'train', 'tvmonitor')


def read_image(path, data_id):
    img_path = Path(path) / "JPEGImages" / "{}.jpg".format(data_id)
    img = cv2.imread(img_path.as_posix())
    if img is None:
        raise RuntimeError("Failed to read the image id {}".format(data_id))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def read_annotations(path, data_id):
    annotations_path = Path(path) / "Annotations" / "{}.xml".format(data_id)
    ann_xml = ET.parse(annotations_path.as_posix()).getroot()

    annotations = []
    for obj in ann_xml.iter('object'):

        label = obj.find('name').text.lower().strip()
        bbox_obj = obj.find('bndbox')
        is_difficult = int(obj.find('difficult').text) == 1
        is_truncated = int(obj.find('truncated').text) == 1
        is_occluded = int(obj.find('occluded').text) == 1

        bbox = [
            int(bbox_obj.find('xmin').text) - 1,
            int(bbox_obj.find('ymin').text) - 1,
            int(bbox_obj.find('xmax').text) - 1,
            int(bbox_obj.find('ymax').text) - 1
        ]
        annotations.append((bbox, label,
                            {'is_difficult': is_difficult,
                             'is_truncated': is_truncated,
                             'is_occluded': is_occluded}))
    return annotations


class VOCDetection(Dataset):
    """VOC Detection Dataset Object

    Arguments:
        path (string): filepath to VOC20XY folder.
        image_set (string): image set, filename from the folder "VOC20XY/ImageSets/Main/",
            e.g. train, trainval, val, test, aeroplane_train, etc.
    """

    def __init__(self, path, image_set):
        assert Path(path).exists(), "VOC dataset path '{}' is not found".format(path)
        self.path = Path(path)
        self.ids = []
        for line in (self.path / "ImageSets" / "Main" / "{}.txt".format(image_set)).open():
            self.ids.append(line.strip())

    def __getitem__(self, index):
        data_id = self.ids[index]

        img = read_image(self.path, data_id)
        annotations = read_annotations(self.path, data_id)

        return img, annotations

    def __len__(self):
        return len(self.ids)


class TransformedDataset(Dataset):

    def __init__(self, dataset, xy_transform):
        assert isinstance(dataset, Dataset)
        assert callable(xy_transform)

        self.dataset = dataset
        self.xy_transform = xy_transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.xy_transform(*self.dataset[index])


def to_target(x, y):
    return x, [(np.array(a[0]), VOC_CLASSES.index(a[1])) for a in y
            if not a[-1]['is_difficult']]
