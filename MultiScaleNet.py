import torch
import torch.nn as nn
from StyleTransferNet import AttentionModule
import numpy as np
from NetworkDeconv import DeConv2d
import math


class PixelNorm(nn.Module):
    def __init__(self):
        super(PixelNorm, self).__init__()
        self.eps = 1e-8

    def forward(self, x):
        mean = torch.mean(x * x, dim=1, keepdim=True)
        dom = torch.rsqrt(mean + self.eps)
        x = x * dom
        return x


class FilterResponseNormLayer(nn.Module):
    def __init__(self, num_features, eps=1e-6):
        super(FilterResponseNormLayer, self).__init__()
        self.num_features = num_features
        self.tau = torch.nn.Parameter(torch.Tensor(1, num_features, 1, 1), requires_grad=True)
        self.beta = torch.nn.Parameter(torch.Tensor(1, num_features, 1, 1), requires_grad=True)
        self.gamma = torch.nn.Parameter(torch.Tensor(1, num_features, 1, 1), requires_grad=True)
        self.eps = torch.nn.Parameter(torch.Tensor([eps]), requires_grad=True)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.zeros_(self.tau)
        nn.init.zeros_(self.beta)
        nn.init.ones_(self.gamma)

    def forward(self, inputs):
        nu2 = torch.mean(inputs**2, dim=(2, 3), keepdim=True, out=None)
        inputs = inputs * torch.rsqrt(nu2 + torch.abs(self.eps))
        return torch.max(self.gamma * inputs + self.beta, self.tau)

    def extra_repr(self):
        return '{}'.format(
            self.num_features
        )


class Empty(nn.Module):
    def __init__(self):
        super(Empty, self).__init__()

    def forward(self, x):
        return x


class MinBatchStd(nn.Module):
    def __init__(self):
        super(MinBatchStd, self).__init__()
        self.eps = 1e-8

    def forward(self, x):
        x_s = x.shape
        x_mean = torch.mean(x, dim=0, keepdim=True)
        x_std = torch.sqrt(torch.mean((x-x_mean)**2, dim=0) + self.eps)
        mean_std = torch.mean(x_std)
        mean_std = mean_std.expand(x_s[0], 1, x_s[2], x_s[3])
        out = torch.cat([x, mean_std], 1)
        return out


class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features), requires_grad=True)
        self.b_2 = nn.Parameter(torch.zeros(features), requires_grad=True)
        self.eps = eps

    def forward(self, x):
        mean = x.mean((2,3), keepdim=True)
        std = x.std((2,3), keepdim=True)
        x = self.a_2 * (x - mean) / (std + self.eps) + self.b_2
        return x


def select_norm(norm_mode, channels):
    if norm_mode == 'none':
        return Empty()
    elif norm_mode == 'pixelnorm':
        return PixelNorm()
    elif norm_mode == 'batchnorm':
        return nn.BatchNorm2d(channels)
    elif norm_mode == 'groupnorm':
        return nn.GroupNorm(num_groups=4,num_channels=channels)
    elif norm_mode == 'syncbatchnorm':
        return nn.SyncBatchNorm(channels, track_running_stats=True)
    elif norm_mode == 'instancenorm':
        return nn.InstanceNorm2d(channels, track_running_stats=True)
    elif norm_mode == 'localresponse':
        return nn.LocalResponseNorm(8)
    elif norm_mode == 'filterresponse':
        return FilterResponseNormLayer(channels)
    elif norm_mode == 'layernorm':
        return LayerNorm(channels)
    else:
        return nn.BatchNorm2d(channels)


def select_conv2d(in_channels, out_channels, use_spectral_norm, conv_type, add_bias=False, padding=1, stride=1, kernel_size=3):
    if conv_type == 'conv':
        if use_spectral_norm:
            conv = nn.utils.spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=add_bias))
        else:
            conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=add_bias)
    elif conv_type == 'deconv':
        if use_spectral_norm:
            conv = nn.utils.spectral_norm(DeConv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=add_bias, num_groups=min(in_channels,16)))
        else:
            conv = DeConv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=add_bias, num_groups=min(in_channels,16))
    elif conv_type == 'channelwise':
        if use_spectral_norm:
            conv = nn.utils.spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=add_bias, groups=math.gcd(in_channels, out_channels)))
        else:
            conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=add_bias, groups=math.gcd(in_channels, out_channels))
    else:
        if use_spectral_norm:
            conv = nn.utils.spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=add_bias))
        else:
            conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=add_bias)

    nn.init.xavier_normal_(conv.weight)

    return conv


def select_nonlinearity(nonlinearity_mode):
    if nonlinearity_mode == 'elu':
        return nn.ELU(alpha=1.0, inplace=True)
    elif nonlinearity_mode == 'relu':
        return nn.ReLU(inplace=True)
    elif nonlinearity_mode == 'selu':
        return nn.SELU(inplace=True)
    elif nonlinearity_mode == 'celu':
        return nn.CELU(alpha=1.0, inplace=True)
    elif nonlinearity_mode == 'leaky_relu':
        return nn.LeakyReLU(negative_slope=0.2, inplace=True)
    elif nonlinearity_mode == 'tanh':
        return nn.Tanh()
    elif nonlinearity_mode == 'none':
        return Empty()
    else:
        return nn.ReLU(inplace=True)


class MultiScaleGeneratorInBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_spectral_norm, conv_type, norm_mode, nonlinearity_mode, use_bias, num_internal_layers):
        super(MultiScaleGeneratorInBlock, self).__init__()

        self.op1 = select_conv2d(in_channels, out_channels, use_spectral_norm, conv_type, use_bias)
        self.op2 = select_nonlinearity(nonlinearity_mode)
        self.op3 = select_norm(norm_mode, out_channels)

        ops = []
        for i in range(num_internal_layers):
            ops.append(select_conv2d(out_channels, out_channels, use_spectral_norm, conv_type, use_bias))
            ops.append(select_nonlinearity(nonlinearity_mode))
            ops.append(select_norm(norm_mode, out_channels))

        self.op4 = nn.Sequential(*ops)

    def forward(self, x):
        x = self.op1(x)
        x = self.op2(x)
        x = self.op3(x)
        x = self.op4(x)
        return x


class MultiScaleGeneratorBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_spectral_norm, conv_type, norm_mode, nonlinearity_mode, use_bias, num_internal_layers):
        super(MultiScaleGeneratorBlock, self).__init__()
        self.op1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.op2 = select_conv2d(in_channels, out_channels, use_spectral_norm, conv_type, use_bias)
        self.op3 = select_nonlinearity(nonlinearity_mode)
        self.op4 = select_norm(norm_mode, out_channels)

        ops = []
        for i in range(num_internal_layers):
            ops.append(select_conv2d(out_channels, out_channels, use_spectral_norm, conv_type, use_bias))
            ops.append(select_nonlinearity(nonlinearity_mode))
            ops.append(select_norm(norm_mode, out_channels))

        self.op5 = nn.Sequential(*ops)

    def forward(self, x):
        x = self.op1(x)
        x = self.op2(x)
        x = self.op3(x)
        x = self.op4(x)
        x = self.op5(x)
        return x


class MultiScaleGeneratorOutBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_spectral_norm, conv_type, use_bias, num_internal_layers):
        super(MultiScaleGeneratorOutBlock, self).__init__()
        self.op1 = select_conv2d(in_channels, out_channels, use_spectral_norm, conv_type, True, padding=0, kernel_size=1)
        self.op2 = nn.Tanh()

    def forward(self, x):
        x = self.op1(x)
        x = self.op2(x)
        return x


class MultiScaleGenerator(nn.Module):
    def __init__(self, latent_size,
                 latent_layers,
                 out_resolution,
                 out_channels,
                 use_spectral_norm,
                 conv_type,
                 norm_mode,
                 use_self_attention,
                 nonlinearity_mode,
                 use_bias,
                 num_internal_layers,
                 internal_channels):
        super(MultiScaleGenerator, self).__init__()

        self.internal_channels = internal_channels

        if use_spectral_norm is True:
            latent_array = [nn.utils.spectral_norm(nn.Linear(latent_size, internal_channels))]
            latent_array.append(select_nonlinearity(nonlinearity_mode))
            for i in range(latent_layers-2):
                latent_array.append(nn.utils.spectral_norm(nn.Linear(internal_channels, internal_channels)))
                latent_array.append(select_nonlinearity(nonlinearity_mode))
            latent_array.append(nn.utils.spectral_norm(nn.Linear(internal_channels, internal_channels)))
            latent_array.append(select_nonlinearity(nonlinearity_mode))
        else:
            latent_array = [nn.Linear(latent_size, internal_channels)]
            latent_array.append(select_nonlinearity(nonlinearity_mode))
            for i in range(latent_layers - 2):
                latent_array.append(nn.Linear(internal_channels, internal_channels))
                latent_array.append(select_nonlinearity(nonlinearity_mode))
            latent_array.append(nn.Linear(internal_channels, internal_channels))
            latent_array.append(select_nonlinearity(nonlinearity_mode))

        self.latent = nn.Sequential(*latent_array)

        levels = int(np.log2(out_resolution)) - 4

        self.ops = nn.ModuleList()
        self.ops_out = nn.ModuleList()

        self.ops.append(MultiScaleGeneratorInBlock(internal_channels//16, internal_channels, use_spectral_norm, conv_type, norm_mode, nonlinearity_mode, use_bias, num_internal_layers))  # [N,512,1,1] -> [N,512,4,4]
        self.ops_out.append(MultiScaleGeneratorOutBlock(internal_channels, out_channels, use_spectral_norm, conv_type, use_bias, num_internal_layers))                  # [N,512,4,4] -> [N,3,4,4]

        for i in range(levels):
            self.ops.append(MultiScaleGeneratorBlock(internal_channels,internal_channels, use_spectral_norm, conv_type, norm_mode, nonlinearity_mode, use_bias, num_internal_layers))
            self.ops_out.append(MultiScaleGeneratorOutBlock(internal_channels, out_channels, use_spectral_norm, conv_type, use_bias, num_internal_layers))

        self.ops.append(MultiScaleGeneratorBlock(internal_channels, internal_channels//2, use_spectral_norm, conv_type, norm_mode, nonlinearity_mode, use_bias, num_internal_layers))
        self.ops_out.append(MultiScaleGeneratorOutBlock(internal_channels//2, out_channels, use_spectral_norm, conv_type, use_bias, num_internal_layers))
        self.ops.append(MultiScaleGeneratorBlock(internal_channels//2, internal_channels//4, use_spectral_norm, conv_type, norm_mode, nonlinearity_mode, use_bias, num_internal_layers))
        self.ops_out.append(MultiScaleGeneratorOutBlock(internal_channels//4, out_channels, use_spectral_norm, conv_type, use_bias, num_internal_layers))

        self.levels = int(np.log2(out_resolution)) - 1

        if use_self_attention is True:
            self.has_attention = True
            self.attention = AttentionModule(internal_channels)
        else:
            self.has_attention = False

    def forward(self, x):
        x = self.latent(x)
        x = x.view(x.shape[0], self.internal_channels//16, 4, 4)
        #x = x.expand(x.shape[0], 512, 6, 6)
        outs = []

        for i in range(self.levels):
            x = self.ops[i](x)
            outs.append(self.ops_out[i](x))
            if i == 3 and self.has_attention is True:
                x = self.attention(x)

        return outs


class MultiScaleDiscriminatorInBlock(nn.Module):
    def __init__(self, in_channels1, in_channels2, out_channels, use_spectral_norm, conv_type, norm_mode, use_minbatch_std, nonlinearity_mode, use_bias):
        super(MultiScaleDiscriminatorInBlock, self).__init__()

        self.op1 = select_conv2d(in_channels1, in_channels2, use_spectral_norm, conv_type, use_bias, padding=0, kernel_size=1)
        if use_minbatch_std is True:
            self.op2 = nn.Sequential(MinBatchStd(), select_conv2d(in_channels2 + 1, in_channels2, use_spectral_norm, conv_type, use_bias))
        else:
            self.op2 = select_conv2d(in_channels2, in_channels2, use_spectral_norm, conv_type, use_bias)

        self.op3 = select_nonlinearity(nonlinearity_mode)
        self.op4 = select_norm(norm_mode, in_channels2)
        self.op5 = select_conv2d(in_channels2, out_channels, use_spectral_norm, conv_type, use_bias)
        self.op6 = select_nonlinearity(nonlinearity_mode)
        self.op7 = select_norm(norm_mode, out_channels)
        self.op8 = nn.AvgPool2d(2)

    def forward(self, x):
        x = self.op1(x)
        x = self.op2(x)
        x = self.op3(x)
        x = self.op4(x)
        x = self.op5(x)
        x = self.op6(x)
        x = self.op7(x)
        x = self.op8(x)
        return x


class MultiScaleDiscriminator2InBlock(nn.Module):
    def __init__(self, in_channels1, in_channels2, out_channels, use_spectral_norm, conv_type, norm_mode, cat_mode, use_minbatch_std, nonlinearity_mode, use_bias):
        super(MultiScaleDiscriminator2InBlock, self).__init__()

        if use_minbatch_std is True:
            self.op1 = MinBatchStd()
            if cat_mode == 'simple':
                self.simple = True
                self.op2 = select_conv2d(in_channels2 + in_channels1 + 1, in_channels2, use_spectral_norm, conv_type, use_bias)
            else:
                self.simple = False
                self.op0 = select_conv2d(in_channels1, in_channels2 // 2, use_spectral_norm, conv_type, use_bias, padding=0, kernel_size=1)
                self.op2 = select_conv2d(in_channels2 + in_channels2//2 + 1, in_channels2, use_spectral_norm, conv_type, use_bias)
        else:
            self.op1 = Empty()
            if cat_mode == 'simple':
                self.simple = True
                self.op2 = select_conv2d(in_channels2 + in_channels1, in_channels2, use_spectral_norm, conv_type, use_bias)
            else:
                self.simple = False
                self.op0 = select_conv2d(in_channels1, in_channels2 // 2, use_spectral_norm, conv_type, use_bias, padding=0, kernel_size=1)
                self.op2 = select_conv2d(in_channels2 + in_channels2 // 2, in_channels2, use_spectral_norm, conv_type, use_bias)
        self.op3 = select_nonlinearity(nonlinearity_mode)
        self.op4 = select_norm(norm_mode, in_channels2)
        self.op5 = select_conv2d(in_channels2, out_channels, use_spectral_norm, conv_type, use_bias)
        self.op6 = select_nonlinearity(nonlinearity_mode)
        self.op7 = select_norm(norm_mode, out_channels)
        self.op8 = nn.AvgPool2d(2)

    def forward(self, x1, x2):
        if self.simple is True:
            x = torch.cat([x1, x2], dim=1)
        else:
            x = self.op0(x1)
            x = torch.cat([x, x2], dim=1)
        x = self.op1(x)
        x = self.op2(x)
        x = self.op3(x)
        x = self.op4(x)
        x = self.op5(x)
        x = self.op6(x)
        x = self.op7(x)
        x = self.op8(x)
        return x


class MultiScaleDiscriminatorOutBlock(nn.Module):
    def __init__(self, in_channels1, in_channels2, out_channels, use_spectral_norm, conv_type, norm_mode, cat_mode, use_minbatch_std, nonlinearity_mode, use_bias):
        super(MultiScaleDiscriminatorOutBlock, self).__init__()

        if use_minbatch_std is True:
            self.op1 = MinBatchStd()
            if cat_mode == 'simple':
                self.simple = True
                self.op2 = select_conv2d(in_channels2 + in_channels1 + 1, out_channels, use_spectral_norm, conv_type, use_bias)
            else:
                self.simple = False
                self.op0 = select_conv2d(in_channels1, in_channels2 // 2, use_spectral_norm, conv_type, use_bias, padding=0, kernel_size=1)
                self.op2 = select_conv2d(in_channels2 + in_channels2 // 2 + 1, out_channels, use_spectral_norm, conv_type, use_bias)
        else:
            self.op1 = Empty()
            if cat_mode == 'simple':
                self.simple = True
                self.op2 = select_conv2d(in_channels2 + in_channels1, out_channels, use_spectral_norm, conv_type, use_bias)
            else:
                self.simple = False
                self.op0 = select_conv2d(in_channels1, in_channels2 // 2, use_spectral_norm, conv_type, use_bias, padding=0, kernel_size=1)
                self.op2 = select_conv2d(in_channels2 + in_channels2 // 2, out_channels, use_spectral_norm, conv_type, use_bias)
        self.op3 = select_nonlinearity(nonlinearity_mode)
        self.op4 = select_norm(norm_mode, out_channels)
        self.op5 = nn.Conv2d(out_channels, out_channels, kernel_size=4, stride=1, padding=0)
        self.op6 = select_nonlinearity(nonlinearity_mode)
        #self.op7 = select_norm(norm_mode, out_channels)

    def forward(self, x1, x2):
        if self.simple is True:
            x = torch.cat([x1, x2], dim=1)
        else:
            x = self.op0(x1)
            x = torch.cat([x, x2], dim=1)
        x = self.op1(x)
        x = self.op2(x)
        x = self.op3(x)
        x = self.op4(x)
        x = self.op5(x)
        x = self.op6(x)
        #x = self.op7(x)
        return x


class MultiScaleDiscriminator(nn.Module):
    def __init__(self, in_channels,
                 in_resolution,
                 out_channels,
                 use_spectral_norm,
                 conv_type,
                 norm_mode,
                 cat_mode,
                 use_self_attention,
                 use_minbatch_std,
                 nonlinearity_mode,
                 use_bias,
                 internal_size):
        super(MultiScaleDiscriminator, self).__init__()

        levels = int(np.log2(in_resolution)) - 5

        self.ops = nn.ModuleList()

        self.ops.append(MultiScaleDiscriminatorInBlock(in_channels, internal_size//8, internal_size//4, use_spectral_norm, conv_type, norm_mode, use_minbatch_std, nonlinearity_mode, use_bias))  # [N,3,256,256] -> [N,64,128,128]
        self.ops.append(MultiScaleDiscriminator2InBlock(in_channels, internal_size//4, internal_size//2, use_spectral_norm, conv_type, norm_mode, cat_mode, use_minbatch_std, nonlinearity_mode, use_bias))  # [N,3,128,128], [N,128,128,128] -> [N,256,64,64]
        self.ops.append(MultiScaleDiscriminator2InBlock(in_channels, internal_size//2, internal_size, use_spectral_norm, conv_type, norm_mode, cat_mode, use_minbatch_std, nonlinearity_mode, use_bias))  # [N,3,64,64], [N,256,64,64] -> [N,512,32,32]

        for i in range(levels):
            self.ops.append(MultiScaleDiscriminator2InBlock(in_channels, internal_size, internal_size, use_spectral_norm, conv_type, norm_mode, cat_mode, use_minbatch_std, nonlinearity_mode, use_bias))  # [N,3,32,32], [N,512,32,32] -> [N,512,16,16]

        self.ops.append(MultiScaleDiscriminatorOutBlock(in_channels, internal_size, internal_size, use_spectral_norm, conv_type, norm_mode, cat_mode, use_minbatch_std, nonlinearity_mode, use_bias))  # [N,3,4,4], [N,512,4,4] -> [N,512,1,1]

        self.ops.append(select_conv2d(internal_size, out_channels, use_spectral_norm, conv_type, True, padding=0, kernel_size=1))  # [N,512,1,1] -> [N,1,1,1]

        self.levels = int(np.log2(in_resolution))

        if use_self_attention is True and self.levels-5 >= 0:
            sizes = [internal_size//8, internal_size//4, internal_size//2, internal_size, internal_size, internal_size, internal_size, internal_size, internal_size, internal_size, internal_size]
            self.use_attention = True
            self.attention = AttentionModule(sizes[self.levels-5])
        else:
            self.use_attention = False



    def forward(self, xs):
        x = self.ops[0](xs[self.levels-2])
        for i in range(self.levels-2):
            x = self.ops[i+1](xs[self.levels-3-i], x)
            if i == self.levels-7 and self.use_attention is True:
                x = self.attention(x)
        x = self.ops[self.levels-1](x)
        return x.view(x.shape[0], x.shape[1])


class DeGeneratorBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_spectral_norm, conv_type, norm_mode, nonlinearity_mode, use_bias, num_internal_layers):
        super(DeGeneratorBlock, self).__init__()

        self.op1 = select_conv2d(in_channels, out_channels, use_spectral_norm, conv_type, use_bias)
        self.op2 = select_nonlinearity(nonlinearity_mode)
        self.op3 = select_norm(norm_mode, out_channels)

        ops = []
        for i in range(num_internal_layers):
            ops.append(select_conv2d(out_channels, out_channels, use_spectral_norm, conv_type, use_bias))
            ops.append(select_nonlinearity(nonlinearity_mode))
            ops.append(select_norm(norm_mode, out_channels))

        self.op4 = nn.Sequential(*ops)

        self.op5 = nn.MaxPool2d(kernel_size=2,stride=2)

    def forward(self, x):
        x = self.op1(x)
        x = self.op2(x)
        x = self.op3(x)
        x = self.op4(x)
        x = self.op5(x)
        return x

class DeGenerator(nn.Module):
    def __init__(self, in_channels,
                 in_resolution,
                 out_channels,
                 use_spectral_norm,
                 conv_type,
                 norm_mode,
                 nonlinearity_mode,
                 use_bias,
                 internal_size):
        super(DeGenerator, self).__init__()

        sizes = [in_channels, internal_size//8, internal_size//4, internal_size//2, internal_size, internal_size, internal_size, internal_size, internal_size, internal_size, internal_size, internal_size, internal_size, internal_size]
        self.levels = int(np.log2(in_resolution)) - 2

        self.ops = nn.ModuleList()
        for i in range(self.levels):
            self.ops.append(DeGeneratorBlock(sizes[i], sizes[i+1], use_spectral_norm, conv_type, norm_mode, nonlinearity_mode, use_bias, internal_size))  # [N,3,256,256] -> [N,64,128,128]

        final_res = int(in_resolution/(2 ** (self.levels)))


        if use_spectral_norm is True:
            self.out = nn.Sequential(
                nn.utils.spectral_norm(nn.Linear(final_res*final_res*sizes[self.levels], internal_size)),
                select_nonlinearity(nonlinearity_mode),
                nn.utils.spectral_norm(nn.Linear(internal_size, out_channels)))
        else:
            self.out = nn.Sequential(
                nn.Linear(final_res*final_res*sizes[self.levels], internal_size),
                select_nonlinearity(nonlinearity_mode),
                nn.Linear(internal_size, out_channels))

    def forward(self, x):
        for i in range(self.levels):
            x = self.ops[i](x)
        B, C, H, W = x.size()
        x = x.view(B,C*H*W)
        x = self.out(x)
        return x