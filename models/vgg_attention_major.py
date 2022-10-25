import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F

cfg_base = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [32, 32, 'M', 64, 64, 'M', 128, 128, 128, 'M', 256, 256, 256, 'M'],  # , 512, 512, 512,      'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
}

cfg_major = [512, 512, 'M']

cfg_minor = [512, 512, 512, 'M']


class VGG_MM(nn.Module):

    def __init__(self, features, num_class=13, num_major=15):
        super().__init__()
        self.features = features[0]
        self.major_branch = features[1]
        self.minor_branch = features[2]
        self.major_avgpool = features[3]
        self.minor_avgpool = features[4]
        self.cbam = CBAM(512)
        self.classifier_major = nn.Sequential(
            nn.Linear(512 * 7 * 7, num_major),
        )
        self.classifier_minor = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_class)
        )

    def forward(self, x):
        x = self.features(x)

        x_major_cbam = self.cbam(x)
        x_minor_cbam = self.cbam(x)
        x_major = self.major_branch(x_major_cbam)
        x_major = self.cbam(x_major)
        x_major = self.major_avgpool(x_major)
        x_major = torch.flatten(x_major, 1)

        pred_major = self.classifier_major(x_major)
        pred_major = pred_major.view(pred_major.size()[0], -1)

        x_minor = self.minor_branch(x_minor_cbam)
        x_minor = self.cbam(x_minor)
        x_minor = self.minor_avgpool(x_minor)
        x_minor = torch.flatten(x_minor, 1)

        x_minor = x_minor.view(x_minor.size()[0], -1)
        x_major = x_major.view(x_major.size()[0], -1)
        x = x_minor + x_major
        # 1*1
        pred_minor = self.classifier_minor(x)

        return pred_major
        # return pred_minor
        # return pred_major, pred_minor


def make_layers(cfg, batch_norm=False):
    layers = nn.Sequential(*list(torchvision.models.vgg16_bn(pretrained=True).children())[0][:-10]).train()
    for p in layers.parameters():
        p.requires_grad = False

    major_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((7, 7)))
    minor_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((7, 7)))

    bch_major = nn.Sequential(
        nn.Conv2d(512, cfg_major[0], kernel_size=3, padding=1),
        nn.BatchNorm2d(cfg_major[0]),
        nn.ReLU(inplace=True),
        nn.Conv2d(cfg_major[0], cfg_major[1], kernel_size=3, padding=1),
        nn.BatchNorm2d(cfg_major[1]),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2, stride=2),
        # nn.AdaptiveAvgPool2d(output_size=(5, 5))
    )

    bch_minor = nn.Sequential(
        nn.Conv2d(512, cfg_minor[0], kernel_size=3, padding=1),
        nn.BatchNorm2d(cfg_minor[0]),
        nn.ReLU(inplace=True),
        nn.Conv2d(cfg_minor[0], cfg_minor[1], kernel_size=3, padding=1),
        nn.BatchNorm2d(cfg_minor[1]),
        nn.ReLU(inplace=True),
        nn.Conv2d(cfg_minor[1], cfg_minor[2], kernel_size=3, padding=1),
        nn.BatchNorm2d(cfg_minor[2]),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2, stride=2),
        # nn.AdaptiveAvgPool2d(output_size=(7,7))
    )

    return [layers, bch_major, bch_minor, major_avg_pool, minor_avg_pool]


def vgg16_bn(num_cls=12, num_mj=15):
    # print(VGG_MM(make_layers(cfg_base['D'], batch_norm=True), num_class=num_cls, num_major=num_mj))
    return VGG_MM(make_layers(cfg_base['D'], batch_norm=True), num_class=num_cls, num_major=num_mj)


class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True,
                 bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
        )
        self.pool_types = pool_types

    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type == 'avg':
                avg_pool = F.avg_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(avg_pool)
            elif pool_type == 'max':
                max_pool = F.max_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(max_pool)
            elif pool_type == 'lp':
                lp_pool = F.lp_pool2d(x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(lp_pool)
            elif pool_type == 'lse':
                # LSE pool only
                lse_pool = logsumexp_2d(x)
                channel_att_raw = self.mlp(lse_pool)

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = F.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale


def logsumexp_2d(tensor):
    tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
    s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
    outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
    return outputs


class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)


class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2, relu=False)

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = F.sigmoid(x_out)  # broadcasting
        return x * scale


class CBAM(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False):
        super(CBAM, self).__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
        self.no_spatial = no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate()

    def forward(self, x):
        x_out = self.ChannelGate(x)
        if not self.no_spatial:
            x_out = self.SpatialGate(x_out)
        return x_out


# net = vgg16_bn(12, 15)
# # net.named_modules()
# # for idx, m in enumerate(net.named_modules()):
# #     print(idx, '->', m)
# from torchsummaryX import summary
#
#
# def cbam_forward_hook(module, input, output):  # cbam의 forward연산일 때 실행, (cbam이름, 피쳐맵 입력, 연산 출력)
#     if isinstance(input, tuple): # input가 튜플이면 리스트로 바꿔줌
#         input = input[0]
#     # inp가 forward일때 특징맵
#     input_featuremap = np.array(input.detach())  # 연산에는 지장없이 input feature 값을 가져옴.
#     # print(input_featuremap[0].shape)
#     # print(input_featuremap[0,0].shape)
#     # print(len(input_featuremap[0,0]))
#     # out이 cham 출력으로 나오는 특징맵
#     # output_featuremap = output.detach()  # output feature 값 가져옴
#     sum = np.zeros((len(input_featuremap[0,0]),len(input_featuremap[0,0])), dtype=float)
#     for i in range(len(input_featuremap[0])):
#         sum += input_featuremap[0, i]
#     sum = sum/len(input_featuremap[0])
#     plt.pcolor(sum)
#     plt.colorbar()
#     plt.show()
#
# for name, module in net.named_modules():
#     if '__main__.CBAM' in str(type(module)):  # cbam에 해당하는 모듈이 있으면
#         module.register_forward_hook(cbam_forward_hook)  # 해당 모듈의 forward연산에 hook 함수 실행

# summary(net, torch.zeros((1,3,224,224)))
# net(torch.zeros((1, 3, 224, 224)))
# x = np.array((512,7,7))
