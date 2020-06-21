import torch
import torch.nn as nn

network_use_bias = True

class AttentionModule(nn.Module):
    def __init__(self, inout_channels):
        super(AttentionModule, self).__init__()
        self.f = nn.Conv2d(inout_channels, int(inout_channels/8), kernel_size=1, stride=1, padding=0, bias=False)
        self.g = nn.Conv2d(inout_channels, int(inout_channels/8), kernel_size=1, stride=1, padding=0, bias=False)
        self.h = nn.Conv2d(inout_channels, inout_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.softmax = nn.Softmax(dim=-1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        B, C, H, W = x.size()
        fx = self.f(x).view(B,-1,W*H).permute(0,2,1) # [B,WH, C/8]
        gx = self.g(x).view(B,-1,W*H) # [B,C/8,WH]
        hx = self.h(x).view(B,-1,W*H) # [B,C,WH]
        beta = torch.bmm(fx,gx) # [B,WH,WH]
        beta = self.softmax(beta) # [B,WH,WH]
        beta = beta.permute(0,2,1)
        o = torch.bmm(hx, beta)
        o = o.view(B,C,H,W)
        y = self.gamma * o + x
        return y

class fromRGB_7(nn.Module):
    def __init__(self, out_c):
        super(fromRGB_7, self).__init__()
        self.pad = nn.ReflectionPad2d(3)
        self.conv = nn.Conv2d(3, out_c, kernel_size=7, stride=1, padding=0, bias=network_use_bias)
        self.norm = nn.InstanceNorm2d(out_c)
        self.act = nn.ReLU(True)
        nn.init.xavier_normal_(self.conv.weight)

    def forward(self, x):
        x = self.pad(x)
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x

class fromRGB_3(nn.Module):
    def __init__(self, out_c):
        super(fromRGB_3, self).__init__()
        self.pad = nn.ReflectionPad2d(1)
        self.conv = nn.Conv2d(3, out_c, kernel_size=3, stride=1, padding=0, bias=network_use_bias)
        self.norm = nn.InstanceNorm2d(out_c)
        self.act = nn.ReLU(True)
        nn.init.xavier_normal_(self.conv.weight)

    def forward(self, x):
        x = self.pad(x)
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x

class fromRGB_1(nn.Module):
    def __init__(self, out_c):
        super(fromRGB_1, self).__init__()
        self.conv = nn.Conv2d(3, out_c, kernel_size=1, stride=1, padding=0, bias=network_use_bias)
        self.norm = nn.InstanceNorm2d(out_c)
        self.act = nn.ReLU(True)
        nn.init.xavier_normal_(self.conv.weight)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x

class toRGB_1(nn.Module):
    def __init__(self, in_c):
        super(toRGB_1, self).__init__()
        self.conv = nn.Conv2d(in_c, 3, kernel_size=1, stride=1, padding=0, bias=network_use_bias)
        self.norm = nn.InstanceNorm2d(3)
        self.act = nn.Tanh()
        nn.init.xavier_normal_(self.conv.weight)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x

class Block_IN(nn.Module):
    def __init__(self, in_c, out_c, kernel_size, stride):
        super(Block_IN, self).__init__()
        self.pad = nn.ReflectionPad2d(kernel_size//2)
        self.conv = nn.Conv2d(in_c, out_c, kernel_size=kernel_size, stride=stride, padding=0, bias=network_use_bias)
        self.norm = nn.InstanceNorm2d(out_c)
        self.act = nn.ReLU(True)
        nn.init.xavier_normal_(self.conv.weight)

    def forward(self, x):
        x = self.pad(x)
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x

class Block_BN(nn.Module):
    def __init__(self, in_c, out_c, kernel_size, stride):
        super(Block_BN, self).__init__()
        self.pad = nn.ReflectionPad2d(kernel_size//2)
        self.conv = nn.Conv2d(in_c, out_c, kernel_size=kernel_size, stride=stride, padding=0, bias=network_use_bias)
        self.norm = nn.BatchNorm2d(out_c)
        self.act = nn.ReLU(True)
        nn.init.xavier_normal_(self.conv.weight)

    def forward(self, x):
        x = self.pad(x)
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x

class Block_2IN(nn.Module):
    def __init__(self, in_c, out_c, kernel_size, stride):
        super(Block_2IN, self).__init__()
        self.pad1 = nn.ReflectionPad2d(kernel_size//2)
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=kernel_size, stride=1, padding=0, bias=network_use_bias)
        self.norm1 = nn.InstanceNorm2d(out_c)
        self.act1 = nn.ReLU(True)
        self.pad2 = nn.ReflectionPad2d(kernel_size // 2)
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=kernel_size, stride=stride, padding=0, bias=network_use_bias)
        self.norm2 = nn.InstanceNorm2d(out_c)
        self.act2 = nn.ReLU(True)
        nn.init.xavier_normal_(self.conv1.weight)
        nn.init.xavier_normal_(self.conv2.weight)

    def forward(self, x):
        x = self.pad1(x)
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act1(x)
        x = self.pad2(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.act2(x)
        return x

class Block_2INConVTranspose(nn.Module):
    def __init__(self, in_c, out_c, kernel_size, stride):
        super(Block_2IN, self).__init__()
        self.pad1 = nn.ReflectionPad2d(kernel_size//2)
        self.conv1 = nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2, bias=network_use_bias)
        self.norm1 = nn.InstanceNorm2d(out_c)
        self.act1 = nn.ReLU(True)
        self.pad2 = nn.ReflectionPad2d(kernel_size // 2)
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=kernel_size, stride=stride, padding=0, bias=network_use_bias)
        self.norm2 = nn.InstanceNorm2d(out_c)
        self.act2 = nn.ReLU(True)
        nn.init.xavier_normal_(self.conv1.weight)
        nn.init.xavier_normal_(self.conv2.weight)

    def forward(self, x):
        x = self.pad1(x)
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act1(x)
        x = self.pad2(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.act2(x)
        return x

class Block_2BN(nn.Module):
    def __init__(self, in_c, out_c, kernel_size, stride):
        super(Block_2BN, self).__init__()
        self.pad1 = nn.ReflectionPad2d(kernel_size//2)
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=kernel_size, stride=1, padding=0, bias=network_use_bias)
        self.norm1 = nn.BatchNorm2d(out_c)
        self.act1 = nn.ReLU(True)
        self.pad2 = nn.ReflectionPad2d(kernel_size//2)
        self.conv2 = nn.Conv2d(in_c, out_c, kernel_size=kernel_size, stride=stride, padding=0, bias=network_use_bias)
        self.norm2 = nn.BatchNorm2d(out_c)
        self.act2 = nn.ReLU(True)
        nn.init.xavier_normal_(self.conv1.weight)
        nn.init.xavier_normal_(self.conv2.weight)

    def forward(self, x):
        x = self.pad1(x)
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act1(x)
        x = self.pad2(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.act2(x)
        return x

class DownBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super(DownBlock, self).__init__()
        self.frgb = fromRGB_1(out_c)
        self.conv = Block_2IN(in_c, out_c, 3, 2)

    def forward(self, i, x):
        x = self.conv(x)
        i = nn.functional.interpolate(i, [x.shape[2],x.shape[3]], mode='bilinear', align_corners=True)
        it = self.frgb(i)
        x = x + it
        return i, x

class UpBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super(UpBlock, self).__init__()
        self.trgb = toRGB_1(out_c)
        self.conv = Block_2IN(in_c*2, out_c, 3, 1)

    def forward(self, i, x, unet_x):
        x = torch.cat([x, unet_x], dim=1)
        x = nn.functional.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = self.conv(x)
        xt = self.trgb(x)
        i = nn.functional.interpolate(i, scale_factor=2, mode='bilinear', align_corners=True)
        it = i + xt
        return it, x

class UpBlockConvTranspose(nn.Module):
    def __init__(self, in_c, out_c):
        super(UpBlockConvTranspose, self).__init__()
        self.trgb = toRGB_1(out_c)
        self.conv = Block_2INConVTranspose(in_c, out_c, 3, 1)

    def forward(self, i, x, unet_x):
        x = torch.cat([x, unet_x], dim=1)
        x = self.conv(x)
        xt = self.trgb(x)
        i = nn.functional.interpolate(i, scale_factor=2, mode='bilinear', align_corners=True)
        it = i + xt
        return it, x

class DBX_UNET(nn.Module):
    def __init__(self, levels):
        super(DBX_UNET, self).__init__()
        channels = [64, 128, 256, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512]

        self.levels = levels

        self.frgb_S = fromRGB_7(64)
        self.frgb_T = fromRGB_7(64)
        self.downblock_S = nn.ModuleList()
        self.downblock_T = nn.ModuleList()

        for i in range(levels):
            self.downblock_S.append(DownBlock(channels[i], channels[i+1]))
            self.downblock_T.append(DownBlock(channels[i], channels[i+1]))

        self.trgb_S = toRGB_1(512)
        self.trgb_T = toRGB_1(512)

        self.upblock_S = nn.ModuleList()
        self.upblock_T = nn.ModuleList()
        for i in range(levels):
            self.upblock_S.append(UpBlock(channels[levels - i], channels[levels - i - 1]))
            self.upblock_T.append(UpBlock(channels[levels - i], channels[levels - i - 1]))


    def forward_encoder_S(self, i):
        x = self.frgb_S(i)
        outs = []
        for j in range(self.levels):
            i, x = self.downblock_S[j](i, x)
            outs.append(x)
        #latent = x
        return x, outs

    def forward_decoder_S(self, x, unet_x):
        ir = self.trgb_S(x)
        for j in range(self.levels):
            ir, x = self.upblock_S[j](ir, x, unet_x[-j-1])
        return ir

    def forward_encoder_T(self, i):
        x = self.frgb_T(i)
        outs = []
        for j in range(self.levels):
            i, x = self.downblock_T[j](i, x)
            outs.append(x)
        #latent = x
        return x, outs

    def forward_decoder_T(self, x, unet_x):
        ir = self.trgb_T(x)
        for j in range(self.levels):
            ir, x = self.upblock_T[j](ir, x, unet_x[-j-1])
        return ir


class DiscriminatorBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super(DiscriminatorBlock, self).__init__()
        self.pad1 = nn.ReflectionPad2d(1)
        self.conv1 = nn.utils.spectral_norm(nn.Conv2d(in_c, out_c, kernel_size=3, stride=1, padding=0, bias=network_use_bias))
        self.norm1 = nn.InstanceNorm2d(out_c)
        self.act1 = nn.ReLU(True)
        self.pad2 = nn.ReflectionPad2d(1)
        self.conv2 = nn.utils.spectral_norm(nn.Conv2d(out_c, out_c, kernel_size=3, stride=2, padding=0, bias=network_use_bias))
        self.norm2 = nn.InstanceNorm2d(out_c)
        self.act2 = nn.ReLU(True)
        self.resconv = nn.utils.spectral_norm(nn.Conv2d(in_c, out_c, kernel_size=1, stride=1, padding=0, bias=network_use_bias))
        nn.init.xavier_normal_(self.conv1.weight)
        nn.init.xavier_normal_(self.conv2.weight)
        nn.init.xavier_normal_(self.resconv.weight)

    def forward(self, x):
        i = x
        i = self.resconv(i)
        x = self.pad1(x)
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act1(x)
        x = self.pad2(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.act2(x)
        i = nn.functional.interpolate(i, [x.shape[2], x.shape[3]], mode='bilinear', align_corners=True)
        x = i + x
        return x

class SmallDiscriminatorBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super(SmallDiscriminatorBlock, self).__init__()
        self.pad1 = nn.ReflectionPad2d(1)
        self.conv1 = nn.utils.spectral_norm(nn.Conv2d(in_c, out_c, kernel_size=3, stride=2, padding=0, bias=network_use_bias))
        self.norm1 = nn.InstanceNorm2d(out_c)
        self.act1 = nn.ReLU(True)
        self.resconv = nn.utils.spectral_norm(nn.Conv2d(in_c, out_c, kernel_size=1, stride=1, padding=0, bias=network_use_bias))
        nn.init.xavier_normal_(self.conv1.weight)
        nn.init.xavier_normal_(self.resconv.weight)

    def forward(self, x):
        i = x
        i = self.resconv(i)
        x = self.pad1(x)
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act1(x)
        i = nn.functional.interpolate(i, [x.shape[2], x.shape[3]], mode='bilinear', align_corners=True)
        x = i + x
        return x


class DBX_Discriminator(nn.Module):
    def __init__(self, input_size, levels):
        super(DBX_Discriminator, self).__init__()

        self.frgb = fromRGB_1(64)
        channels = [64, 128, 256, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512]
        self.downblock = nn.ModuleList()
        for i in range(levels):
            self.downblock.append(DiscriminatorBlock(channels[i], channels[i + 1]))
            input_size = input_size//2
        self.out = nn.Linear(input_size*input_size*512, 1)
        self.levels = levels

    def forward(self, x):
        x = self.frgb(x)
        for i in range(self.levels):
            x = self.downblock[i](x)
        x = x.view(x.shape[0],-1)
        x = self.out(x)
        return x

class DBX_SmallerDiscriminator(nn.Module):
    def __init__(self, input_size, levels):
        super(DBX_SmallerDiscriminator, self).__init__()

        self.frgb = fromRGB_1(64)
        channels = [64, 128, 256, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512]
        self.downblock = nn.ModuleList()
        for i in range(levels):
            self.downblock.append(DiscriminatorBlock(channels[i], channels[i + 1]))
            input_size = input_size//2
        self.down = nn.AvgPool2d(kernel_size=input_size)
        self.out = nn.Linear(512, 1)
        self.levels = levels

    def forward(self, x):
        x = self.frgb(x)
        for i in range(self.levels):
            x = self.downblock[i](x)
        x = self.down(x)
        x = x.view(x.shape[0],-1)
        x = self.out(x)
        return x

class DBX_SmallestDiscriminator(nn.Module):
    def __init__(self, input_size, levels):
        super(DBX_SmallestDiscriminator, self).__init__()

        self.frgb = fromRGB_1(64)
        channels = [64, 128, 256, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512]
        self.downblock = nn.ModuleList()
        for i in range(levels):
            self.downblock.append(SmallDiscriminatorBlock(channels[i], channels[i + 1]))
            input_size = input_size//2
        self.down = nn.AvgPool2d(kernel_size=input_size)
        self.out = nn.Linear(512, 1)
        self.levels = levels

    def forward(self, x):
        x = self.frgb(x)
        for i in range(self.levels):
            x = self.downblock[i](x)
        x = self.down(x)
        x = x.view(x.shape[0],-1)
        x = self.out(x)
        return x








'''
class Unet_Block(nn.Module):
    def __init__(self, in_channels, out_channels, use_spectral_norm):
        super(Unet_Block, self).__init__()
        if use_spectral_norm:
            self.op1 = nn.utils.spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=0, bias=False))
        else:
            self.op1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=0, bias=False)
        self.op2 = nn.ReLU(True)
        if use_spectral_norm:
            self.op3 = nn.GroupNorm(8, out_channels, affine=False)
        else:
            self.op3 = nn.BatchNorm2d(out_channels)
        if use_spectral_norm:
            self.op4 = nn.utils.spectral_norm(nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=0, bias=False))
        else:
            self.op4 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=0, bias=False)
        self.op5 = nn.ReLU(True)
        if use_spectral_norm:
            self.op6 = nn.GroupNorm(8, out_channels, affine=False)
        else:
            self.op6 = nn.BatchNorm2d(out_channels)
        self.op7 = nn.ReflectionPad2d(2)

    def forward(self, x):
        x = self.op1(x)
        x = self.op2(x)
        x = self.op3(x)
        x = self.op4(x)
        x = self.op5(x)
        x = self.op6(x)
        x = self.op7(x)
        return x

class Unet_DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_spectral_norm, use_conv_downsample):
        super(Unet_DownBlock, self).__init__()
        self.conv_downsample = use_conv_downsample
        if use_conv_downsample is False:
            self.op1 = nn.MaxPool2d(2)
            self.op2 = Unet_Block(in_channels,out_channels, use_spectral_norm)
        else:
            self.op1 = nn.Conv2d(in_channels, in_channels, kernel_size=2, stride=2, padding=0, bias=False)
            self.op2 = Unet_Block(in_channels, out_channels, use_spectral_norm)

    def forward(self, x):
        x = self.op1(x)
        x = self.op2(x)
        return x

class Unet_UpBlock(nn.Module):
    def __init__(self, in_channels, in_channels_skip,  out_channels, use_spectral_norm, use_deconv_upsample):
        super(Unet_UpBlock, self).__init__()
        self.deconv_upsample = use_deconv_upsample
        if use_deconv_upsample:
            self.op1 = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2)
            self.op2 = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0, bias=False)
            self.op3 = Unet_Block(in_channels+in_channels_skip,out_channels,use_spectral_norm)
        else:
            self.op1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.op2 = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0, bias=False)
            self.op3 = Unet_Block(in_channels+in_channels_skip,out_channels,use_spectral_norm)

    def forward(self, x, y):
        x = self.op1(x)
        x = self.op2(x)
        t = torch.cat([y,x],dim=1)
        o = self.op3(t)
        return o

class Unet_Out(nn.Module):
    def __init__(self, in_channels, out_channels, output_bias):
        super(Unet_Out, self).__init__()
        self.op = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=output_bias)

    def forward(self, x):
        return self.op(x)

class ArtitsTransfer_UNET(nn.Module):
    def __init__(self, in_channels, use_spectral_norm, output_bias, use_conv_deconv_for_downsample_upsample):
        super(ArtitsTransfer_UNET, self).__init__()
        self.input = Unet_Block(in_channels, 64, use_spectral_norm)                                         # [N,3,256,256] -> [N,64,256,256]
        self.down1 = Unet_DownBlock(64, 128, use_spectral_norm, use_conv_deconv_for_downsample_upsample)    # [N,64,256,256] -> [N,128,128,128]
        self.down2 = Unet_DownBlock(128, 256, use_spectral_norm, use_conv_deconv_for_downsample_upsample)   # [N,128,128,128] -> [N,256,64,64]
        self.attn1 = AttentionModule(256)
        self.down3 = Unet_DownBlock(256, 512, use_spectral_norm, use_conv_deconv_for_downsample_upsample)   # [N,256,64,64] -> [N,512,32,32]
        self.attn2 = AttentionModule(512)
        self.down4 = Unet_DownBlock(512, 512, use_spectral_norm, use_conv_deconv_for_downsample_upsample)   # [N,512,32,32] -> [N,512,16,16]
        self.up1 = Unet_UpBlock(512,512, 256, use_spectral_norm, use_conv_deconv_for_downsample_upsample)   # [N,512,16,16],[N,512,32,32] -> [N,256,32,32]
        self.up2 = Unet_UpBlock(256,256, 128, use_spectral_norm, use_conv_deconv_for_downsample_upsample)   # [N,256,32,32],[N,256,64,64] -> [N,128,64,64]
        self.up3 = Unet_UpBlock(128, 128, 64, use_spectral_norm, use_conv_deconv_for_downsample_upsample)   # [N,128,64,64],[N,128,1284,128] -> [N,64,128,128]
        self.up4 = Unet_UpBlock(64, 64, 64, use_spectral_norm, use_conv_deconv_for_downsample_upsample)     # [N,64,128,128],[N,64,256,256] -> [N,64,256,256]
        self.output = Unet_Out(64, in_channels, output_bias)                                                # [N,64,256,256] -> [N,3,256,256]
        self.tanh = nn.Tanh()

    def forward(self, x):
        x1 = self.input(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        o = self.down4(x4)
        x4 = self.attn2(x4) #before concatenating pass it throug attention module
        o = self.up1(o, x4)
        x3 = self.attn1(x3) #before concatenating pass it throug attention module
        o = self.up2(o, x3)
        o = self.up3(o, x2)
        o = self.up4(o, x1)
        out = self.output(o)
        out = self.tanh(out)
        return out

class DiscriminatorBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_spectral_norm):
        super(DiscriminatorBlock, self).__init__()
        self.has_spectral_norm = use_spectral_norm
        if use_spectral_norm:
            self.op1 = nn.utils.spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=0, bias=False))
        else:
            self.op1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=0, bias=False)
            self.op1_n = nn.GroupNorm(8, out_channels)
        self.op2 = nn.LeakyReLU(0.1, True)
        if use_spectral_norm:
            self.op3 = nn.utils.spectral_norm(nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=0, bias=False))
        else:
            self.op3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=0, bias=False)
            self.op3_n = nn.GroupNorm(8, out_channels)
        self.op4 = nn.LeakyReLU(0.1, True)

    def forward(self, x):
        x = self.op1(x)
        if self.has_spectral_norm is False:
            x = self.op1_n(x)
        x = self.op2(x)
        x = self.op3(x)
        if self.has_spectral_norm is False:
            x = self.op3_n(x)
        x = self.op4(x)

        return x

class DiscriminatorLastBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DiscriminatorLastBlock, self).__init__()
        self.op3 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x):
        x = self.op3(x)
        return x

class ArtitsTransfer_Discriminator(nn.Module):
    def __init__(self, in_channels, out_channels, use_spectral_norm):
        super(ArtitsTransfer_Discriminator, self).__init__()
        self.down1 = DiscriminatorBlock(in_channels, 64, use_spectral_norm)  # [N,3,256,256] -> [N,64,128,128]
        self.down2 = DiscriminatorBlock(64, 128, use_spectral_norm)  # [N,64,128,128] -> [N,128,64,64]
        self.down3 = DiscriminatorBlock(128, 256, use_spectral_norm)  # [N,128,64,64] -> [N,256,32,32]
        self.down4 = DiscriminatorBlock(256, 256, use_spectral_norm)  # [N,256,32,32] -> [N,256,16,16]
        self.down5 = DiscriminatorBlock(256, 512, use_spectral_norm)  # [N,256,16,16] -> [N,512,8,8]
        self.down6 = DiscriminatorBlock(512, 512, use_spectral_norm)  # [N,512,8,8] -> [N,512,4,4]
        self.out = DiscriminatorLastBlock(512, out_channels)  # [N,512,4,4] -> [N,512,1,1] ->  [N,1,1,1]

    def forward(self, x):
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        x = self.down4(x)
        x = self.down5(x)
        x = self.down6(x)
        x = self.out(x)
        return x.view(x.shape[0], x.shape[1])
'''