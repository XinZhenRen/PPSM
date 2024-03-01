import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn import functional as F
#from models.modules import LCA,ASM,GCM_up,GCM,CrossNonLocalBlock
import math

""" Local Context Attention Module"""


def scaled_dot_product(q, k, v, mask=None):
    d_k = q.size()[-1]
    attn_logits = torch.matmul(q, k.transpose(-2, -1))
    attn_logits = attn_logits / math.sqrt(d_k)
    if mask is not None:
        attn_logits = attn_logits.masked_fill(mask == 0, -9e15)
    attention = F.softmax(attn_logits, dim=-1)
    values = torch.matmul(attention, v)
    return values, attention


class LCA(nn.Module):
    def __init__(self):
        super(LCA, self).__init__()

    def forward(self, x, pred):
        residual = x
        out = residual

        return out


""" Global Context Module"""


class GCM(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCM, self).__init__()
        pool_size = [1, 3, 5]
        out_channel_list = [256, 128, 64, 64]
        upsampe_scale = [2, 4, 8, 16]
        GClist = []
        GCoutlist = []
        for ps in pool_size:
            GClist.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(ps),
                nn.Conv2d(in_channels, out_channels, 1, 1),
                nn.ReLU(inplace=True)))
        GClist.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, 1),
            nn.ReLU(inplace=True),
            NonLocalBlock(out_channels)))
        self.GCmodule = nn.ModuleList(GClist)
        for i in range(4):
            GCoutlist.append(nn.Sequential(nn.Conv2d(out_channels * 4, out_channel_list[i], 3, 1, 1),
                                           nn.ReLU(inplace=True),
                                           nn.Upsample(scale_factor=upsampe_scale[i], mode='bilinear')))
        self.GCoutmodel = nn.ModuleList(GCoutlist)

    def forward(self, x):
        xsize = x.size()[2:]
        global_context = []
        for i in range(len(self.GCmodule) - 1):
            global_context.append(F.interpolate(self.GCmodule[i](x), xsize, mode='bilinear', align_corners=True))
        global_context.append(self.GCmodule[-1](x))
        global_context = torch.cat(global_context, dim=1)

        output = []
        for i in range(len(self.GCoutmodel)):
            output.append(self.GCoutmodel[i](global_context))

        return output


""" Adaptive Selection Module"""


class ASM(nn.Module):
    def __init__(self, in_channels, all_channels):
        super(ASM, self).__init__()
        self.non_local = NonLocalBlock(in_channels)

    def forward(self, lc, fuse, gc):
        fuse = self.non_local(fuse)
        fuse = torch.cat([lc, fuse, gc], dim=1)

        return fuse


"""
Squeeze and Excitation Layer

https://arxiv.org/abs/1709.01507

"""


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


"""
Non Local Block

https://arxiv.org/abs/1711.07971
"""


class NonLocalBlock(nn.Module):
    def __init__(self, in_channels, inter_channels=None, sub_sample=True, bn_layer=True):
        super(NonLocalBlock, self).__init__()

        self.sub_sample = sub_sample

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        self.g = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0)

        if bn_layer:
            self.W = nn.Sequential(
                nn.Conv2d(in_channels=self.inter_channels, out_channels=self.in_channels,
                          kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(self.in_channels)
            )
            nn.init.constant_(self.W[1].weight, 0)
            nn.init.constant_(self.W[1].bias, 0)
        else:
            self.W = nn.Conv2d(in_channels=self.inter_channels, out_channels=self.in_channels,
                               kernel_size=1, stride=1, padding=0)
            nn.init.constant_(self.W.weight, 0)
            nn.init.constant_(self.W.bias, 0)

        self.theta = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                               kernel_size=1, stride=1, padding=0)
        self.phi = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)

        if sub_sample:
            self.g = nn.Sequential(self.g, nn.MaxPool2d(kernel_size=(2, 2)))
            self.phi = nn.Sequential(self.phi, nn.MaxPool2d(kernel_size=(2, 2)))

    def forward(self, x):

        batch_size = x.size(0)

        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
        f = torch.matmul(theta_x, phi_x)
        f_div_C = F.softmax(f, dim=-1)

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x

        return z


# AGCM Module
class CrossNonLocalBlock(nn.Module):
    def __init__(self, in_channels_source, in_channels_target, inter_channels, sub_sample=False, bn_layer=True):
        super(CrossNonLocalBlock, self).__init__()

        self.sub_sample = sub_sample

        self.in_channels_source = in_channels_source
        self.in_channels_target = in_channels_target
        self.inter_channels = inter_channels

        """
        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1
        """
        self.g = nn.Conv2d(in_channels=self.in_channels_source, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0)
        self.theta = nn.Conv2d(in_channels=self.in_channels_source, out_channels=self.inter_channels,
                               kernel_size=1, stride=1, padding=0)
        self.phi = nn.Conv2d(in_channels=self.in_channels_target, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)

        if bn_layer:
            self.W = nn.Sequential(
                nn.Conv2d(in_channels=self.inter_channels, out_channels=self.in_channels_target,
                          kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(self.in_channels_target)
            )
            nn.init.constant_(self.W[1].weight, 0)
            nn.init.constant_(self.W[1].bias, 0)

        if sub_sample:
            self.g = nn.Sequential(self.g, nn.MaxPool2d(kernel_size=(2, 2)))
            self.phi = nn.Sequential(self.phi, nn.MaxPool2d(kernel_size=(2, 2)))

    def forward(self, x, l):

        batch_size = x.size(0)
        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)  # source
        theta_x1 = self.theta(x)
        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)  # source
        phi_x = self.phi(l).view(batch_size, self.inter_channels, -1)  # target
        f = torch.matmul(theta_x, phi_x)
        f_div_C = F.softmax(f, dim=-1)
        f_div_C = f_div_C.permute(0, 2, 1)
        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *l.size()[2:])
        W_y = self.W(y)
        z = W_y + l

        return z


# SFEM module
class NonLocalBlock_PatchWise(nn.Module):

    def __init__(self, in_channel, inter_channel, patch_factor):
        super(NonLocalBlock_PatchWise, self).__init__()
        "Embedding dimension must be 0 modulo number of heads."
        self.in_channel = in_channel
        self.patch_factor = patch_factor
        self.patch_width = int(8 / self.patch_factor)
        self.patch_height = int(8 / self.patch_factor)
        self.stride_width = int(8 / self.patch_factor)
        self.stride_height = int(8 / self.patch_factor)
        self.unfold = nn.Unfold(kernel_size=(self.patch_width, self.patch_height),
                                stride=(self.stride_width, self.stride_height))

        self.adp = nn.AdaptiveAvgPool2d(8)
        self.bottleneck = nn.Conv2d(64, inter_channel, kernel_size=(1, 1))
        self.non_block = NonLocalBlock(self.in_channel)
        self.adp_post = nn.AdaptiveAvgPool2d((8, 8))

    def forward(self, x):
        batch_size = x.size(0)
        x_up = self.adp(x)
        x_up = self.unfold(x)
        batch_size, p_dim, p_size = x_up.size()
        x_up = x_up.view(batch_size, -1, self.in_channel, p_size)
        final_output = torch.tensor([]).cuda()
        index = torch.arange(0, p_size, 1, dtype=torch.int64).cuda()
        for i in range(int(p_size)):
            divide = torch.index_select(x_up, 3, index[i])
            divide = divide.view(batch_size, -1, self.in_channel)
            patch_width = int(divide.size(1) ** 0.5)
            divide = divide.reshape(batch_size, self.in_channel, patch_width, patch_width)  # tensor to operate on
            attn = self.non_block(divide)
            output = attn.view(batch_size, -1, self.in_channel, 1)
            final_output = torch.cat((final_output, output), dim=3)

        final_output = final_output.view(batch_size, self.in_channel, 8, 8)

        return final_output


class GCM_up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCM_up, self).__init__()
        self.adp = nn.AdaptiveAvgPool2d((8, 8))
        self.patch1 = NonLocalBlock_PatchWise(in_channels, out_channels, 2)
        self.patch2 = NonLocalBlock_PatchWise(in_channels, out_channels, 4)
        self.patch3 = NonLocalBlock(256, 64)
        self.fuse = SELayer(3 * 256)
        self.conv = nn.Conv2d(3 * 256, out_channels, 1, 1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        b, c, h, w = x.size()
        x = self.adp(x)
        patch1 = self.patch1(x)
        patch2 = self.patch2(x)
        patch3 = self.patch3(x)
        global_cat = torch.cat((patch1, patch2, patch3), dim=1)
        fuse = self.relu(self.conv(self.fuse(global_cat)))
        adp_post = nn.AdaptiveAvgPool2d((h, w))
        fuse = adp_post(fuse)
        return fuse


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size=3, stride=1, padding=1):
        super(DecoderBlock, self).__init__()

        self.conv1 = ConvBlock(in_channels, in_channels // 4, kernel_size=kernel_size,
                               stride=stride, padding=padding)

        self.conv2 = ConvBlock(in_channels // 4, out_channels, kernel_size=kernel_size,
                               stride=stride, padding=padding)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.upsample(x)
        return x


class SideoutBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(SideoutBlock, self).__init__()

        self.conv1 = ConvBlock(in_channels, in_channels // 4, kernel_size=kernel_size,
                               stride=stride, padding=padding)

        self.dropout = nn.Dropout2d(0.1)

        self.conv2 = nn.Conv2d(in_channels // 4, out_channels, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.dropout(x)
        x = self.conv2(x)

        return x


class EUNet(nn.Module):
    def __init__(self, num_classes:int = 1):
        super(EUNet, self).__init__()

        resnet = models.resnet34(pretrained=True)
       
        # Encoder
        self.encoder1_conv = resnet.conv1
        self.encoder1_bn = resnet.bn1
        self.encoder1_relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.encoder2 = resnet.layer1
        self.encoder3 = resnet.layer2
        self.encoder4 = resnet.layer3
        self.encoder5 = resnet.layer4


        # Decoder
        self.decoder5 = DecoderBlock(in_channels=512, out_channels=512)
        self.decoder4 = DecoderBlock(in_channels=1024, out_channels=256)
        self.decoder3 = DecoderBlock(in_channels=512, out_channels=128)
        self.decoder2 = DecoderBlock(in_channels=256, out_channels=64)
        self.decoder1 = DecoderBlock(in_channels=192, out_channels=64)

        self.outconv = nn.Sequential(ConvBlock(64, 32, kernel_size=3, stride=1, padding=1),
                                        nn.Dropout2d(p=0.1),
                                      nn.Conv2d(in_channels=32, out_channels=num_classes, kernel_size=1))

        self.outenc = ConvBlock(512,256,kernel_size=1, stride=1,padding=0)

        # Sideout
        self.sideout2 = SideoutBlock(64, 1)
        self.sideout3 = SideoutBlock(128, 1)
        self.sideout4 = SideoutBlock(256, 1)
        self.sideout5 = SideoutBlock(512, 1)

   

        # global context module
        self.gcm_up = GCM_up(256,64)
        self.gcm_e5 = GCM_up(256, 256)#3
        
        self.gcm_e4 = GCM_up(256, 128)#2
        self.gcm_e3 = GCM_up(256, 64)#1
        self.gcm_e2 = GCM_up(256, 64)#0


        # adaptive selection module
        self.asm4 = ASM(512, 1024)
        self.asm3 = ASM(256, 512)
        self.asm2 = ASM(128, 256)
        self.asm1 = ASM(64, 192)

        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True) 
        self.up2 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True) 
        self.up3 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True) 
        self.up4 = nn.Upsample(scale_factor=16, mode='bilinear', align_corners=True)

        self.lca_cross_1 = CrossNonLocalBlock(512,256,256)
        self.lca_cross_2 = CrossNonLocalBlock(1024,128,128)
        self.lca_cross_3 = CrossNonLocalBlock(512,64,64)
        self.lca_cross_4 = CrossNonLocalBlock(256,64,64)

    def forward(self, x):
        e1 = self.encoder1_conv(x) 
        e1 = self.encoder1_bn(e1)
        e1 = self.encoder1_relu(e1)
        e1_pool = self.maxpool(e1)  
        e2 = self.encoder2(e1_pool)
        e3 = self.encoder3(e2) 
        e4 = self.encoder4(e3)  
        e5 = self.encoder5(e4)  
        e_ex = self.outenc(e5)
        
        global_contexts_up = self.gcm_up(e_ex)

        
        d5 = self.decoder5(e5)  
        out5 = self.sideout5(d5)
        lc4 = self.lca_cross_1(d5,e4)
        gc4 = self.gcm_e5(e_ex)
        gc4 = self.up1(gc4)
        
        
        comb4 = self.asm4(lc4, d5, gc4)

        d4 = self.decoder4(comb4) 
        out4 = self.sideout4(d4)
        lc3 = self.lca_cross_2(comb4,e3)
        gc3 = self.gcm_e4(e_ex)
        gc3 = self.up2(gc3)
        
 
        comb3 = self.asm3(lc3, d4, gc3)
        

        d3 = self.decoder3(comb3)
        out3 = self.sideout3(d3)
        lc2= self.lca_cross_3(comb3,e2)
        gc2 = self.gcm_e3(e_ex)
        gc2 = self.up3(gc2)
        
        comb2 = self.asm2(lc2, d3, gc2)

        d2 = self.decoder2(comb2)  
        out2 = self.sideout2(d2)
        lc1 = self.lca_cross_4(comb2,e1)
        gc1 = self.gcm_e2(e_ex)
        gc1 = self.up4(gc1)
       
        comb1 = self.asm1(lc1, d2, gc1)

        d1 = self.decoder1(comb1) 
        out1 = self.outconv(d1)  

        # return torch.sigmoid(out1), torch.sigmoid(out2), torch.sigmoid(out3), \
        #     torch.sigmoid(out4), torch.sigmoid(out5)
        return out1


if __name__ == '__main__':
    x=torch.randn((4,3,352,352)).cuda()
    Net=EUNet(1).cuda()
    y=Net(x)
    print(y.shape)