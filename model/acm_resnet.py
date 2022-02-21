from typing import Type, Any, Callable, Union, List, Optional

import torch
import torch.nn as nn
from ..builder import BACKBONES
from torch import Tensor

__all__ = [
    "ACMResNet",
    "acmresnet50",
    "acmresnet101",
]


# model_urls = {
#     "resnet18": "https://download.pytorch.org/models/resnet18-f37072fd.pth",
#     "resnet34": "https://download.pytorch.org/models/resnet34-b627a593.pth",
#     "resnet50": "https://download.pytorch.org/models/resnet50-0676ba61.pth",
#     "resnet101": "https://download.pytorch.org/models/resnet101-63fe2227.pth",
#     "resnet152": "https://download.pytorch.org/models/resnet152-394f9c45.pth",
#     "resnext50_32x4d": "https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth",
#     "resnext101_32x8d": "https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth",
#     "wide_resnet50_2": "https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth",
#     "wide_resnet101_2": "https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth",
# }


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv5x5(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=5,
        stride=stride,
        padding=2 * dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
            self,
            inplanes: int,
            planes: int,
            stride: int = 1,
            downsample: Optional[nn.Module] = None,
            groups: int = 1,
            base_width: int = 64,
            dilation: int = 1,
            norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion: int = 4

    def __init__(
            self,
            inplanes: int,
            planes: int,
            stride: int = 1,
            downsample: Optional[nn.Module] = None,
            groups: int = 1,
            base_width: int = 64,
            dilation: int = 1,
            norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class AC5_Module(nn.Module):
    expansion: int = 4

    def __init__(
            self,
            inplanes: int,
            planes: int,
            stride: int = 1,
            downsample: Optional[nn.Module] = None,
            groups: int = 1,
            base_width: int = 64,
            norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        self.conv1_d1 = conv1x1(inplanes, width)
        self.bn1_d1 = norm_layer(width)
        self.conv2_d1 = conv5x5(width, width, stride, groups, dilation=1)
        self.bn2_d1 = norm_layer(width)
        self.conv3_d1 = conv1x1(width, planes * self.expansion)
        self.bn3_d1 = norm_layer(planes * self.expansion)
        self.conv1_d2 = conv1x1(inplanes, width)
        self.bn1_d2 = norm_layer(width)
        self.conv2_d2 = conv5x5(width, width, stride, groups, dilation=2)
        self.bn2_d2 = norm_layer(width)
        self.conv3_d2 = conv1x1(width, planes * self.expansion)
        self.bn3_d2 = norm_layer(planes * self.expansion)
        self.conv1_d3 = conv1x1(inplanes, width)
        self.bn1_d3 = norm_layer(width)
        self.conv2_d3 = conv5x5(width, width, stride, groups, dilation=4)
        self.bn2_d3 = norm_layer(width)
        self.conv3_d3 = conv1x1(width, planes * self.expansion)
        self.bn3_d3 = norm_layer(planes * self.expansion)

        self.conv_fusion = conv1x1(3 * planes * self.expansion, planes * self.expansion)
        self.bn_fusion = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        # self.elu = nn.ELU()
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        acm1 = self.conv1_d1(x)
        acm1 = self.bn1_d1(acm1)
        acm1 = self.relu(acm1)
        acm1 = self.conv2_d1(acm1)
        acm1 = self.bn2_d1(acm1)
        acm1 = self.relu(acm1)
        acm1 = self.conv3_d1(acm1)
        acm1 = self.bn3_d1(acm1)
        acm2 = self.conv1_d2(x)
        acm2 = self.bn1_d2(acm2)
        acm2 = self.relu(acm2)
        acm2 = self.conv2_d2(acm2)
        acm2 = self.bn2_d2(acm2)
        acm2 = self.relu(acm2)
        acm2 = self.conv3_d2(acm2)
        acm2 = self.bn3_d2(acm2)
        acm3 = self.conv1_d3(x)
        acm3 = self.bn1_d3(acm3)
        acm3 = self.relu(acm3)
        acm3 = self.conv2_d3(acm3)
        acm3 = self.bn2_d3(acm3)
        acm3 = self.relu(acm3)
        acm3 = self.conv3_d3(acm3)
        acm3 = self.bn3_d3(acm3)
        # print(acm1.shape)
        # print(acm2.shape)
        # print(acm3.shape)
        out = self.conv_fusion(torch.cat([acm1, acm2, acm3], dim=1))
        out = self.bn_fusion(out)
        out = self.relu(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


@BACKBONES.register_module()
class ACMResNet(nn.Module):
    def __init__(
            self,
            block: Type[Union[BasicBlock, Bottleneck]] = Bottleneck,
            layers: List[int] = [3, 4, 6, 3],
            num_classes: int = 1000,
            zero_init_residual: bool = False,
            groups: int = 1,
            width_per_group: int = 64,
            replace_stride_with_dilation: Optional[List[bool]] = None,
            norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(AC5_Module, 64, layers[0], acm=True)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(
            self,
            block: Type[Union[BasicBlock, Bottleneck]],
            planes: int,
            blocks: int,
            stride: int = 1,
            dilate: bool = False,
            acm: bool = False
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        if acm:
            layers.append(
                block(self.inplanes, planes, stride, downsample, self.groups, self.base_width, norm_layer)
            )
            self.inplanes = planes * block.expansion
            for _ in range(1, blocks):
                layers.append(
                    block(
                        self.inplanes,
                        planes,
                        groups=self.groups,
                        base_width=self.base_width,
                        norm_layer=norm_layer,
                    )
                )
        else:
            layers.append(
                block(
                    self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation,
                    norm_layer
                )
            )
            self.inplanes = planes * block.expansion
            for _ in range(1, blocks):
                layers.append(
                    block(
                        self.inplanes,
                        planes,
                        groups=self.groups,
                        base_width=self.base_width,
                        dilation=self.dilation,
                        norm_layer=norm_layer,
                    )
                )

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        outs = []
        print(x.shape)
        x = self.layer1(x)
        outs.append(x)
        print(x.shape)
        x = self.layer2(x)
        outs.append(x)
        print(x.shape)
        x = self.layer3(x)
        outs.append(x)
        print(x.shape)
        x = self.layer4(x)
        outs.append(x)
        print(outs)
        return tuple(outs)

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


def _acm_resnet(
        arch: str,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        pretrained: bool,
        progress: bool,
        **kwargs: Any,
) -> ACMResNet:
    model = ACMResNet(block, layers, **kwargs)
    if pretrained:
        print("pretrained model not available for now.")
        # state_dict = load_state_dict_from_url(model_urls[arch], progress=progress)
        # model.load_state_dict(state_dict)
    return model


def acmresnet50(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ACMResNet:
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _acm_resnet("resnet50", Bottleneck, [3, 4, 6, 3], pretrained, progress, **kwargs)


def acmresnet101(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ACMResNet:
    r"""ResNet-101 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _acm_resnet("resnet101", Bottleneck, [3, 4, 23, 3], pretrained, progress, **kwargs)


if __name__ == '__main__':
    x = torch.randn((4, 3, 254, 254))
    model = acmresnet50()
    y = model(x)
