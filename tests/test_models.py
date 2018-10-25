
import torch

from models import BasicConv, RFB3, RFB4, SSDDetectionHead


def test_basic_conv():

    x = torch.rand(4, 16, 12, 12)
    in_planes = x.shape[1]
    out_planes = in_planes * 2
    kernel_size = 3
    conv = BasicConv(in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, with_relu=True)
    y = conv(x)
    assert isinstance(y, torch.Tensor) and y.shape[1] == out_planes


def test_rfb3():

    x = torch.rand(4, 16, 12, 12)
    in_planes = x.shape[1]
    out_planes = in_planes * 2
    rfb = RFB3(in_planes, out_planes, stride=1, scale=0.1, visual=1)
    y = rfb(x)
    assert isinstance(y, torch.Tensor) and y.shape[1] == out_planes


def test_rfb4():

    x = torch.rand(4, 16, 12, 12)
    in_planes = x.shape[1]
    out_planes = in_planes * 2
    rfb = RFB4(in_planes, out_planes, stride=1, scale=0.1)
    y = rfb(x)
    assert isinstance(y, torch.Tensor) and y.shape[1] == out_planes


def test_ssd_detection_head():

    inputs = [
        torch.rand(4, 32, 16, 16),
        torch.rand(4, 64, 8, 8),
        torch.rand(4, 128, 4, 4),
        torch.rand(4, 128, 2, 2),
    ]

    in_planes_list = [x.shape[1] for x in inputs]
    num_anchors_list = [6 for _ in inputs]
    num_classes = 12
    head = SSDDetectionHead(in_planes_list, num_anchors_list, num_classes)
    loc, conf = head(inputs)
    assert isinstance(loc, torch.Tensor) and loc.shape[-1] == 4
    assert isinstance(conf, torch.Tensor) and conf.shape[-1] == num_classes
