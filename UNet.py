import torch.nn as nn
import torch
import torch.nn.functional as F


class unet(nn.Module):

    def __init__(self, in_chl, feature_scale=4, num_blocks=3, is_deconv=True, is_batchnorm=True, skip_connect='None'):
        super(unet, self).__init__()
        self.is_deconv = is_deconv
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale
        self.skip_connect = skip_connect
        self.num_blocks = num_blocks

        filters = [128, 256, 512, 1024]
        filters = [int(x / self.feature_scale) for x in filters]

        # self.inFilter = unetConv2(in_chl, filters[0], self.is_batchnorm)

        down_layers = []
        ichls = in_chl * 2
        for i in range(num_blocks):
            down_layers.append(unetConv2(ichls, filters[i], self.is_batchnorm))
            ichls = filters[i]

        up_layers = []
        for i in range(num_blocks):
            if skip_connect == 'NONE':
                up_layers.append(unetUpNoCat(filters[i + 1], filters[i], True))
            else:
                up_layers.append(unetUp(filters[i + 1], filters[i], True))
        skip_layers = []

        if skip_connect == 'All':
            for i in range(num_blocks):
                skip_layers.append(unetConv2(filters[i], filters[i], self.is_batchnorm))

        self.down_layers = nn.ModuleList(down_layers)
        self.up_layers = nn.ModuleList(up_layers)
        self.skip_layers = nn.ModuleList(skip_layers)

        self.center_layer = unetConv2(filters[num_blocks - 1], filters[num_blocks], self.is_batchnorm)
        self.outconv = nn.Conv2d(filters[0], in_chl, 1)

    def forward(self, x1, x2):

        features = []
        # inp = self.inFilter(x)
        inp = torch.cat([x1, x2], 1)
        for i in range(self.num_blocks):
            inp = self.down_layers[i](inp)
            features.append(inp)
            inp = F.max_pool2d(inp, 2)

        inp = self.center_layer(inp)

        for i in reversed(range(self.num_blocks)):
            if self.skip_connect == 'NONE':
                inp = self.up_layers[i](inp)
            elif self.skip_connect == 'One':
                inp = self.up_layers[i](features[i], inp)
            elif self.skip_connect == 'All':
                inp2 = self.skip_layers[i](features[i])
                inp = self.up_layers[i](inp2, inp)
        inp = self.outconv(inp)

        return inp


class MultiUnet(nn.Module):
    def __init__(self, in_chl, feature_scale=4, num_blocks=3, is_deconv=True, is_batchnorm=True, skip_connect='None'):
        super(MultiUnet, self).__init__()
        self.unet = unet(in_chl, feature_scale, num_blocks, is_deconv, is_batchnorm, skip_connect)

    def forward(self, left, right, length):
        results = []
        for i in range(length):
            left = self.unet(left, right)
            results.append(left)
        return results


class unetConv2(nn.Module):
    def __init__(self, in_size, out_size, is_batchnorm):
        super(unetConv2, self).__init__()

        if is_batchnorm:
            self.conv1 = nn.Sequential(nn.Conv2d(in_size, out_size, 3, 1, 1),
                                       nn.BatchNorm2d(out_size),
                                       nn.ReLU(), )
            self.conv2 = nn.Sequential(nn.Conv2d(out_size, out_size, 3, 1, 1),
                                       nn.BatchNorm2d(out_size),
                                       nn.ReLU(), )
        else:
            self.conv1 = nn.Sequential(nn.Conv2d(in_size, out_size, 3, 1, 1),
                                       nn.ReLU(), )
            self.conv2 = nn.Sequential(nn.Conv2d(out_size, out_size, 3, 1, 1),
                                       nn.ReLU(), )

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        return outputs


class unetUp(nn.Module):
    def __init__(self, in_size, out_size, is_deconv):
        super(unetUp, self).__init__()
        self.conv = unetConv2(in_size, out_size, False)
        if is_deconv:
            self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2, padding=0)
        else:
            self.up = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, inputs1, inputs2):
        outputs2 = self.up(inputs2)
        offset = outputs2.size()[2] - inputs1.size()[2]
        padding = 2 * [offset // 2, offset // 2]
        outputs1 = F.pad(inputs1, padding)
        return self.conv(torch.cat([outputs1, outputs2], 1))


class unetUpNoCat(nn.Module):
    def __init__(self, in_size, out_size, is_deconv):
        super(unetUpNoCat, self).__init__()
        self.conv = unetConv2(in_size, out_size, False)
        if is_deconv:
            self.up = nn.ConvTranspose2d(in_size, in_size, kernel_size=2, stride=2, padding=0)
        else:
            self.up = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, inputs):
        outputs = self.up(inputs)
        return self.conv(outputs)


if __name__ == '__main__':
    a = unet(32, skip_connect='NONE')
    b = torch.randn(1, 32, 64, 64)
    b = torch.autograd.Variable(b)
    print(a(b).size())
