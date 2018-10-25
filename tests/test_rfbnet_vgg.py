
import torch

import pytest

from models.rfbnet_vgg import RFBPyramidNet, RFBNetVGG


def test_rfb_head():

    x = torch.rand(4, 1024, 32, 32)
    in_planes = x.shape[1]
    net = RFBPyramidNet(in_planes=in_planes)
    out = net(x)
    assert isinstance(out, list) and len(out) == len(RFBPyramidNet.out_planes_list)
    for i, y in enumerate(out):
        assert y.shape[1] == RFBPyramidNet.out_planes_list[i]


def test_rfb_net_vgg():

    x = torch.rand(4, 3, 512, 512)
    num_classes = 12
    net = RFBNetVGG(num_classes=num_classes)
    loc, conf = net(x)

    assert isinstance(loc, torch.Tensor) and loc.shape[-1] == 4
    assert isinstance(conf, torch.Tensor) and conf.shape[-1] == num_classes


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Skip if no GPU")
def test_rfb_net_vgg_cuda():

    x = torch.rand(4, 3, 512, 512).to('cuda')
    num_classes = 12
    net = RFBNetVGG(num_classes=num_classes).to('cuda')
    loc, conf = net(x)

    assert isinstance(loc, torch.Tensor) and loc.shape[-1] == 4
    assert isinstance(conf, torch.Tensor) and conf.shape[-1] == num_classes
    shape_1 = loc.shape[1]

    x = torch.rand(4, 3, 768, 768).to('cuda')
    num_classes = 12
    net = RFBNetVGG(num_classes=num_classes).to('cuda')
    loc, conf = net(x)

    assert isinstance(loc, torch.Tensor) and loc.shape[-1] == 4
    assert isinstance(conf, torch.Tensor) and conf.shape[-1] == num_classes
    shape_2 = loc.shape[1]

    assert shape_1 != shape_2
