from functools import partial
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
from Models import pvt_v2
from timm.models.vision_transformer import _cfg


class RB(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.in_layers = nn.Sequential(
            nn.GroupNorm(32, in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        )

        self.out_layers = nn.Sequential(
            nn.GroupNorm(32, out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        )

        if out_channels == in_channels:
            self.skip = nn.Identity()
        else:
            self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        h = self.in_layers(x)
        h = self.out_layers(h)
        return h + self.skip(x)


class FCB(nn.Module):
    def __init__(
        self,
        in_channels=3,
        min_level_channels=32,
        min_channel_mults=[1, 1, 2, 2, 4, 4],
        n_levels_down=6,
        n_levels_up=6,
        n_RBs=2,
        in_resolution=352,
    ):
        super().__init__()
        self.enc_blocks = nn.ModuleList(
            [nn.Conv2d(in_channels, min_level_channels, kernel_size=3, padding=1)]
        )
        ch = min_level_channels
        enc_block_chans = [min_level_channels]
        for level in range(n_levels_down):#图是down sample+RB+RB 但是编程是 RB+RB+down sample
            min_channel_mult = min_channel_mults[level]
            for block in range(n_RBs):
                self.enc_blocks.append(
                    nn.Sequential(RB(ch, min_channel_mult * min_level_channels))
                )
                ch = min_channel_mult * min_level_channels
                enc_block_chans.append(ch)
            if level != n_levels_down - 1:
                self.enc_blocks.append(
                    nn.Sequential(nn.Conv2d(ch, ch, kernel_size=3, padding=1, stride=2))#相当于down sample行和列的步距都是2
                )
                enc_block_chans.append(ch)

        self.middle_block = nn.Sequential(RB(ch, ch), RB(ch, ch))

        self.dec_blocks = nn.ModuleList([])
        for level in range(n_levels_up):
            min_channel_mult = min_channel_mults[::-1][level]

            for block in range(n_RBs + 1):
                layers = [
                    RB(
                        ch + enc_block_chans.pop(),
                        min_channel_mult * min_level_channels,
                    )
                ]
                ch = min_channel_mult * min_level_channels
                if level < n_levels_up - 1 and block == n_RBs:
                    layers.append(
                        nn.Sequential(
                            nn.Upsample(scale_factor=2, mode="nearest"),
                            nn.Conv2d(ch, ch, kernel_size=3, padding=1),
                        )
                    )
                self.dec_blocks.append(nn.Sequential(*layers))

    def forward(self, x):
        hs = []
        h = x
        for module in self.enc_blocks:
            h = module(h)
            hs.append(h)
        h = self.middle_block(h)
        for module in self.dec_blocks:
            cat_in = torch.cat([h, hs.pop()], dim=1)
            h = module(cat_in)
        return h
class TB(nn.Module):
    def __init__(self):

        super().__init__()

        backbone = pvt_v2.PyramidVisionTransformerV2(
            patch_size=4,
            embed_dims=[64, 128, 320, 512],
            num_heads=[1, 2, 5, 8],
            mlp_ratios=[8, 8, 4, 4],
            qkv_bias=True,
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            depths=[3, 4, 18, 3],
            sr_ratios=[8, 4, 2, 1],
        )

        checkpoint = torch.load("pvt_v2_b3.pth")
        backbone.default_cfg = _cfg()
        backbone.load_state_dict(checkpoint)
        self.backbone = torch.nn.Sequential(*list(backbone.children()))[:-1]

        for i in [1, 4, 7, 10]:
            self.backbone[i] = torch.nn.Sequential(*list(self.backbone[i].children()))

        self.LE = nn.ModuleList([])
        for i in range(4):
            self.LE.append(
                nn.Sequential(
                    RB([64, 128, 320, 512][i], 64), RB(64, 64), nn.Upsample(size=88)
                )
            )

        self.SFA = nn.ModuleList([])
        for i in range(3):
            self.SFA.append(nn.Sequential(RB(128, 64), RB(64, 64)))

    def get_pyramid(self, x):
        pyramid = []
        B = x.shape[0]
        for i, module in enumerate(self.backbone):
            if i in [0, 3, 6, 9]:
                x, H, W = module(x)
            elif i in [1, 4, 7, 10]:
                for sub_module in module:
                    x = sub_module(x, H, W)
            else:
                x = module(x)
                x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
                pyramid.append(x)

        return pyramid

    def forward(self, x):
        pyramid = self.get_pyramid(x)
        pyramid_emph = []
        for i, level in enumerate(pyramid):
            pyramid_emph.append(self.LE[i](pyramid[i]))

        l_i = pyramid_emph[-1]
        for i in range(2, -1, -1):
            l = torch.cat((pyramid_emph[i], l_i), dim=1)
            l = self.SFA[i](l)
            l_i = l

        return l

class PE(nn.Module):
    def __init__(self,in_channels=0,promt_label=[0,0,0,0]):#promt_label:Edge Polygon Scribble Point
        super().__init__()
        match sum(promt_label):
            case 0:
                self.conv1=nn.Conv2d(in_channels=1, )
            case 1:
                pass
            case 2:
                pass
            case 3:
                pass
            case 4:
                pass


    def forward(self, x):
        pyramid = self.get_pyramid(x)
        pyramid_emph = []
        for i, level in enumerate(pyramid):
            pyramid_emph.append(self.LE[i](pyramid[i]))

        l_i = pyramid_emph[-1]
        for i in range(2, -1, -1):
            l = torch.cat((pyramid_emph[i], l_i), dim=1)
            l = self.SFA[i](l)
            l_i = l

        return l

class FCBFormer(nn.Module):
    def __init__(self, size=352):
        super().__init__()
        self.TB = TB()
        self.FCB = FCB(in_resolution=size)
        self.PH = nn.Sequential(
            RB(64 + 32, 64), RB(64, 64), nn.Conv2d(64, 1, kernel_size=1)
        )
        self.up_tosize = nn.Upsample(size=size)
    def forward(self, x):
        x1 = self.TB(x)
        x2 = self.FCB(x)
        x1 = self.up_tosize(x1)
        x = torch.cat((x1, x2), dim=1)
        out = self.PH(x)
        return out
class DoubleConv(nn.Sequential):
    def __init__(self,in_channels,out_channels,mid_channels=None):
        self.in_channels=in_channels
        self.out_channels=out_channels
        self.mid_channels=mid_channels
        if mid_channels is None:
            self.mid_channels=self.out_channels
        super(DoubleConv,self).__init__(
            nn.Conv2d(self.in_channels,self.mid_channels,3,1,1,bias=False),
            nn.BatchNorm2d(self.mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.mid_channels,self.out_channels,3,1,1,bias=False),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU(inplace=True)
        )

class FEEMIn(nn.Sequential):
    def __init__(self,in_channels,out_channels,mid_channels=None):
        if mid_channels==None:
            mid_channels=out_channels
        super(FEEMIn,self).__init__(
            DoubleConv(in_channels,out_channels,mid_channels)
        )

class FEEMDown(nn.Module):
    def __init__(self,in_channels,out_channels,pooling=True):
        super(FEEMDown,self).__init__()
        self.in_channels=in_channels
        self.out_channels=out_channels
        self.pooling=pooling
        self.Cov=DoubleConv(self.out_channels,self.out_channels)
        self.MaxPool=nn.MaxPool2d(2,2)

    def forward(self,x,p):
        x=torch.cat((x,p),dim=1)
        #print(x.shape)
        x=self.Cov(x)
        if self.pooling:
            x=self.MaxPool(x)
        return x

class PEMIn(nn.Sequential):
    def __init__(self,in_channels=5,out_channels=64,mid_channels=64):
        super(PEMIn,self).__init__(
            DoubleConv(in_channels,out_channels,mid_channels)
        )

class PEMDown(nn.Sequential):
    def __init__(self,in_channels,out_channels):
        super(PEMDown,self).__init__(
            nn.MaxPool2d(2,2),
            DoubleConv(in_channels,out_channels)
        )



class MDMUp(nn.Module):
    def __init__(self,in_channels,out_channels,mid_channels=None,uping=True):
        super(MDMUp,self).__init__()
        self.in_channels=in_channels
        self.out_channels=out_channels
        self.mid_channels=mid_channels
        if self.mid_channels==None:
            self.mid_channels=out_channels
        self.uping=uping

        self.Cov1=DoubleConv(self.in_channels,self.out_channels,self.mid_channels)
        self.Up=nn.Upsample(scale_factor=2,mode='bilinear',align_corners=True)
        self.Cov2 = DoubleConv(self.in_channels, self.out_channels, self.mid_channels)
    def forward(self,x,p):
        x=self.Cov1(x)

        x=torch.cat((x,p),dim=1)
        x=self.Cov2(x)
        if self.uping:
            x=self.Up(x)
            # diff_y = x2.size()[2] - x1.size()[2]
            # diff_x = x2.size()[3] - x1.size()[3]
            # # padding_left padding_right padding_top padding_bootom
            # x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
            #                 diff_y // 2, diff_y - diff_y // 2])
        return x

class OutConv(nn.Sequential):
    def __init__(self,in_channels,num_classes):
        super(OutConv,self).__init__(
            nn.Conv2d(in_channels,num_classes,kernel_size=1)
        )



class PPSN(nn.Module):
    def __init__(self,in_channels=3,p_in_channels=5,num_class=1,base_c=64,chooseProb=1):
        super().__init__()
        self.chooseProb=chooseProb


        self.PEMIn=PEMIn(p_in_channels,base_c,base_c)
        self.PEMDown1=PEMDown(base_c,base_c*2)
        self.PEMDown2 = PEMDown(base_c*2,base_c*4)

        self.FEEMIn = FEEMIn(in_channels, base_c)
        self.FEEMDown1 = FEEMDown(base_c,base_c*2)
        self.FEEMDown2 = FEEMDown(base_c*2, base_c*4)
        self.FEEMDown3 = FEEMDown(base_c*4,base_c*8 ,pooling=False)

        self.MDMUp1=MDMUp(base_c*8,base_c*4)
        self.MDMUp2 = MDMUp(base_c*4,base_c*2)
        self.MDMUp3 = MDMUp(base_c*2,base_c,uping=False)
        self.Out=OutConv(base_c,num_class)


    def forward(self, x,prompt):
        randChoose=torch.rand(prompt.size(1),device=prompt.device)<self.chooseProb
        #print("prompt.size(1)",prompt.size(1))
        randChoose=randChoose.view(1,prompt.size(1),1,1).expand(prompt.size(0),-1,prompt.size(2),prompt.size(3))
        prompt=prompt*randChoose
        p1=self.PEMIn(prompt)
        p2 = self.PEMDown1(p1)
        p3 = self.PEMDown2(p2)

        x1=self.FEEMIn(x)
        x2=self.FEEMDown1(x1,p1)
        x3 = self.FEEMDown2(x2, p2)
        x4 = self.FEEMDown3(x3, p3)

        m1=self.MDMUp1(x4,p3)
        m2 = self.MDMUp2(m1, p2)
        m3 = self.MDMUp3(m2, p1)
        m=self.Out(m3)

        return m



if __name__ == '__main__':
    #test
    x=torch.randn((1,3,352,352))
    p = torch.randn((1, 5, 352, 352))
    model=PPSN(in_channels=3,p_in_channels=5,num_class=1,base_c=64)
    predict=model(x,p)
    print(x.shape)
    print(p.shape)
    print(predict.shape)



