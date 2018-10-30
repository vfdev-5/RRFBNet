
import torch.nn as nn

from torchvision.models.vgg import vgg16

from models import BasicConv, RFB3, RFB4, SSDDetectionHead


def get_num_anchors_list(aspect_ratios=((2, 3), (2, 3), (2, 3), (2, 3), (2, 3), (2,), (2,))):
    return [(2 * len(a) + 2) for a in aspect_ratios]


class RFBNetVGG(nn.Module):
    """
    Receptive-Field Block Network

    input image should be D x D, RGB
    """
    def __init__(self, num_classes, num_anchors_list=(6, 6, 6, 6, 6, 4, 4)):
        super(RFBNetVGG, self).__init__()
        self.num_classes = num_classes

        net = vgg16(pretrained=False)
        self.blocks1234 = net.features[0:23]

        self.mp_block5 = net.features[23:]
        self.mp_block5[-1] = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)

        self.block67 = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, kernel_size=1),
            nn.ReLU(inplace=True)
        )

        self.rfb4 = RFB4(512, 512, stride=1, scale=1.0)
        self.rfb_pyramid = RFBPyramidNet(in_planes=1024)

        in_planes_list = (512, ) + self.rfb_pyramid.out_planes_list
        self.detection_head = SSDDetectionHead(in_planes_list, num_anchors_list, num_classes)

    def forward_encoder(self, x):
        detection_inputs = []
        x = self.blocks1234(x)
        detection_inputs.append(self.rfb4(x))
        x = self.mp_block5(x)
        x = self.block67(x)
        detection_inputs += self.rfb_pyramid(x)
        return detection_inputs

    def forward(self, x):
        detection_inputs = self.forward_encoder(x)
        return self.detection_head(detection_inputs)


class RFBPyramidNet(nn.Module):

    out_planes_list = (1024, 512, 512, 256, 256, 256)

    def __init__(self, in_planes):
        super(RFBPyramidNet, self).__init__()

        self.blocks = nn.ModuleList([
            RFB3(in_planes, self.out_planes_list[0], scale=1.0, visual=2),
            RFB3(self.out_planes_list[0], self.out_planes_list[1], stride=2, scale=1.0, visual=2),
            RFB3(self.out_planes_list[1], self.out_planes_list[2], stride=2, scale=1.0, visual=2),
            RFB3(self.out_planes_list[2], self.out_planes_list[3], stride=2, scale=1.0, visual=1),
            RFB3(self.out_planes_list[3], self.out_planes_list[4], stride=2, scale=1.0, visual=1),
        ])

        self.last_block = nn.Sequential(
            BasicConv(self.out_planes_list[4], self.out_planes_list[4] // 2, kernel_size=1, stride=1),
            BasicConv(self.out_planes_list[4] // 2, self.out_planes_list[5], kernel_size=4, stride=1, padding=1)
        )

    def forward(self, x):
        output = []
        for b in self.blocks:
            x = b(x)
            output.append(x)
        output.append(self.last_block(x))
        return output
