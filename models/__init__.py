
import torch
import torch.nn as nn


class BasicConv(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, with_relu=True):
        super(BasicConv, self).__init__()
        self.block = nn.Sequential()
        self.block.add_module(
            "conv", nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation)
        )
        self.block.add_module(
            "bn", nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True)
        )
        if with_relu:
            self.block.add_module(
                "relu", nn.ReLU(inplace=True)
            )

    def forward(self, x):
        return self.block(x)


class RFB3(nn.Module):
    """
    Basic RFBlock of 3 branches and a shorcut
    """
    def __init__(self, in_planes, out_planes, stride=1, scale=0.1, visual=1):
        super(RFB3, self).__init__()
        self.scale = scale
        inter_planes = in_planes // 8
        self.branch0 = nn.Sequential(
            BasicConv(in_planes, 2 * inter_planes, kernel_size=1, stride=stride),
            BasicConv(2 * inter_planes, 2 * inter_planes,
                      kernel_size=3, stride=1, padding=visual, dilation=visual, with_relu=False)
        )
        self.branch1 = nn.Sequential(
            BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
            BasicConv(inter_planes, 2 * inter_planes, kernel_size=3, stride=stride, padding=1),
            BasicConv(2 * inter_planes, 2 * inter_planes,
                      kernel_size=3, stride=1, padding=visual + 1, dilation=visual + 1, with_relu=False)
        )
        self.branch2 = nn.Sequential(
            BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
            BasicConv(inter_planes, (inter_planes // 2) * 3, kernel_size=3, stride=1, padding=1),
            BasicConv((inter_planes // 2) * 3, 2 * inter_planes, kernel_size=3, stride=stride, padding=1),
            BasicConv(2 * inter_planes, 2 * inter_planes,
                      kernel_size=3, stride=1, padding=2 * visual + 1, dilation=2 * visual + 1, with_relu=False)
        )

        self.out_conv = BasicConv(6 * inter_planes, out_planes, kernel_size=1, stride=1, with_relu=False)
        self.shortcut = BasicConv(in_planes, out_planes, kernel_size=1, stride=stride, with_relu=False)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)

        out = torch.cat((x0, x1, x2), 1)
        out = self.out_conv(out)
        short = self.shortcut(x)
        out = out * self.scale + short
        out = self.relu(out)
        return out


class RFB4(nn.Module):
    """
    Basic RFBlock of 4 branches and a shorcut
    """

    def __init__(self, in_planes, out_planes, stride=1, scale=0.1):
        super(RFB4, self).__init__()
        self.scale = scale
        self.out_channels = out_planes
        inter_planes = in_planes // 4

        self.branch0 = nn.Sequential(
            BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
            BasicConv(inter_planes, inter_planes, kernel_size=3, stride=1, padding=1, with_relu=False)
        )
        self.branch1 = nn.Sequential(
            BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
            BasicConv(inter_planes, inter_planes, kernel_size=(3, 1), stride=1, padding=(1, 0)),
            BasicConv(inter_planes, inter_planes, kernel_size=3, stride=1, padding=3, dilation=3, with_relu=False)
        )
        self.branch2 = nn.Sequential(
            BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
            BasicConv(inter_planes, inter_planes, kernel_size=(1, 3), stride=stride, padding=(0, 1)),
            BasicConv(inter_planes, inter_planes, kernel_size=3, stride=1, padding=3, dilation=3, with_relu=False)
        )
        self.branch3 = nn.Sequential(
            BasicConv(in_planes, inter_planes // 2, kernel_size=1, stride=1),
            BasicConv(inter_planes // 2, (inter_planes // 4) * 3, kernel_size=(1, 3), stride=1, padding=(0, 1)),
            BasicConv((inter_planes // 4) * 3, inter_planes, kernel_size=(3, 1), stride=stride, padding=(1, 0)),
            BasicConv(inter_planes, inter_planes, kernel_size=3, stride=1, padding=5, dilation=5, with_relu=False)
        )

        self.out_conv = BasicConv(4 * inter_planes, out_planes, kernel_size=1, stride=1, with_relu=False)
        self.shortcut = BasicConv(in_planes, out_planes, kernel_size=1, stride=stride, with_relu=False)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)

        out = torch.cat((x0, x1, x2, x3), 1)
        out = self.out_conv(out)
        short = self.shortcut(x)
        out = out * self.scale + short
        out = self.relu(out)
        return out


class SSDDetectionHead(nn.Module):

    def __init__(self, in_planes_list, num_anchors_list, num_classes):
        super(SSDDetectionHead, self).__init__()

        if not isinstance(in_planes_list, (list, tuple)):
            raise TypeError("Argument in_planes_list should be a list/tuple")

        if not isinstance(num_anchors_list, (list, tuple)):
            raise TypeError("Argument num_anchors_list should be a list/tuple")

        if len(in_planes_list) != len(num_anchors_list):
            raise ValueError("{} != {}".format(len(in_planes_list), len(num_anchors_list)))

        self.num_classes = num_classes
        self.loc_layers = []
        self.conf_layers = []
        for s, n in zip(in_planes_list, num_anchors_list):
            self.loc_layers.append(nn.Conv2d(s, n * 4, kernel_size=3, padding=1))
            self.conf_layers.append(nn.Conv2d(s, n * num_classes, kernel_size=3, padding=1))

        self.loc_layers = nn.ModuleList(self.loc_layers)
        self.conf_layers = nn.ModuleList(self.conf_layers)

    def forward(self, detection_inputs):
        if not isinstance(detection_inputs, (list, tuple)):
            raise TypeError("Input should be a list/tuple")

        if len(detection_inputs) != len(self.loc_layers):
            raise ValueError("{} != {}".format(len(self.detection_inputs), len(self.loc_layers)))

        loc = []
        conf = []

        for x, l, c in zip(detection_inputs, self.loc_layers, self.conf_layers):
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())

        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)

        loc = loc.view(loc.size(0), -1, 4)
        conf = conf.view(conf.size(0), -1, self.num_classes)
        return loc, conf
