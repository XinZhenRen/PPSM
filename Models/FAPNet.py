import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

#from lib.Res2Net_v1b import res2net50_v1b_26w_4s
import math
import torch.utils.model_zoo as model_zoo


__all__ = ['Res2Net', 'res2net50_v1b', 'res2net101_v1b', 'res2net50_v1b_26w_4s']

model_urls = {
    'res2net50_v1b_26w_4s': 'https://shanghuagao.oss-cn-beijing.aliyuncs.com/res2net/res2net50_v1b_26w_4s-3cf99910.pth',
    'res2net101_v1b_26w_4s': 'https://shanghuagao.oss-cn-beijing.aliyuncs.com/res2net/res2net101_v1b_26w_4s-0812c246.pth',
}


class Bottle2neck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, baseWidth=26, scale=4, stype='normal'):
        """ Constructor
        Args:
            inplanes: input channel dimensionality
            planes: output channel dimensionality
            stride: conv stride. Replaces pooling layer.
            downsample: None when stride = 1
            baseWidth: basic width of conv3x3
            scale: number of scale.
            type: 'normal': normal set. 'stage': first block of a new stage.
        """
        super(Bottle2neck, self).__init__()

        width = int(math.floor(planes * (baseWidth / 64.0)))
        self.conv1 = nn.Conv2d(inplanes, width * scale, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width * scale)

        if scale == 1:
            self.nums = 1
        else:
            self.nums = scale - 1
        if stype == 'stage':
            self.pool = nn.AvgPool2d(kernel_size=3, stride=stride, padding=1)
        convs = []
        bns = []
        for i in range(self.nums):
            convs.append(nn.Conv2d(width, width, kernel_size=3, stride=stride, padding=1, bias=False))
            bns.append(nn.BatchNorm2d(width))
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)

        self.conv3 = nn.Conv2d(width * scale, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stype = stype
        self.scale = scale
        self.width = width

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        spx = torch.split(out, self.width, 1)
        for i in range(self.nums):
            if i == 0 or self.stype == 'stage':
                sp = spx[i]
            else:
                sp = sp + spx[i]
            sp = self.convs[i](sp)
            sp = self.relu(self.bns[i](sp))
            if i == 0:
                out = sp
            else:
                out = torch.cat((out, sp), 1)
        if self.scale != 1 and self.stype == 'normal':
            out = torch.cat((out, spx[self.nums]), 1)
        elif self.scale != 1 and self.stype == 'stage':
            out = torch.cat((out, self.pool(spx[self.nums])), 1)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Res2Net(nn.Module):

    def __init__(self, block, layers, baseWidth=26, scale=4, num_classes=1000):
        self.inplanes = 64
        super(Res2Net, self).__init__()
        self.baseWidth = baseWidth
        self.scale = scale
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, 1, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, 1, 1, bias=False)
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.AvgPool2d(kernel_size=stride, stride=stride,
                             ceil_mode=True, count_include_pad=False),
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample=downsample,
                            stype='stage', baseWidth=self.baseWidth, scale=self.scale))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, baseWidth=self.baseWidth, scale=self.scale))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def res2net50_v1b(pretrained=False, **kwargs):
    """Constructs a Res2Net-50_v1b lib.
    Res2Net-50 refers to the Res2Net-50_v1b_26w_4s.
    Args:
        pretrained (bool): If True, returns a lib pre-trained on ImageNet
    """
    model = Res2Net(Bottle2neck, [3, 4, 6, 3], baseWidth=26, scale=4, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['res2net50_v1b_26w_4s']))
    return model


def res2net101_v1b(pretrained=False, **kwargs):
    """Constructs a Res2Net-50_v1b_26w_4s lib.
    Args:
        pretrained (bool): If True, returns a lib pre-trained on ImageNet
    """
    model = Res2Net(Bottle2neck, [3, 4, 23, 3], baseWidth=26, scale=4, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['res2net101_v1b_26w_4s']))
    return model


def res2net50_v1b_26w_4s(pretrained=False, **kwargs):
    """Constructs a Res2Net-50_v1b_26w_4s lib.
    Args:
        pretrained (bool): If True, returns a lib pre-trained on ImageNet
    """
    model = Res2Net(Bottle2neck, [3, 4, 6, 3], baseWidth=26, scale=4, **kwargs)
    if pretrained:
        model_state = torch.load('/home/gpu/PPSN-main/Others/MSNet-M2SNet-main/weight/res2net50_v1b_26w_4s-3cf99910.pth')
        model.load_state_dict(model_state)
        # lib.load_state_dict(model_zoo.load_url(model_urls['res2net50_v1b_26w_4s']))
    return model


def res2net101_v1b_26w_4s(pretrained=False, **kwargs):
    """Constructs a Res2Net-50_v1b_26w_4s lib.
    Args:
        pretrained (bool): If True, returns a lib pre-trained on ImageNet
    """
    model = Res2Net(Bottle2neck, [3, 4, 23, 3], baseWidth=26, scale=4, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['res2net101_v1b_26w_4s']))
    return model


def res2net152_v1b_26w_4s(pretrained=False, **kwargs):
    """Constructs a Res2Net-50_v1b_26w_4s lib.
    Args:
        pretrained (bool): If True, returns a lib pre-trained on ImageNet
    """
    model = Res2Net(Bottle2neck, [3, 8, 36, 3], baseWidth=26, scale=4, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['res2net152_v1b_26w_4s']))
    return model



class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x
    

###################################################################    
class MFAM0(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(MFAM0, self).__init__()
        
        self.relu = nn.ReLU(inplace=True)

  
        self.conv_1_1 = BasicConv2d(in_channels, out_channels, 1)
        self.conv_1_2 = BasicConv2d(in_channels, out_channels, 1)
        self.conv_1_3 = BasicConv2d(in_channels, out_channels, 1)
        self.conv_1_4 = BasicConv2d(in_channels, out_channels, 1)
        self.conv_1_5 = BasicConv2d(out_channels, out_channels, 3, stride=1, padding=1)
        
        self.conv_3_1 = nn.Conv2d(out_channels,   out_channels , kernel_size=3, stride=1, padding=1)
        self.conv_3_2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        
        self.conv_5_1 = nn.Conv2d(out_channels,   out_channels , kernel_size=5, stride=1, padding=2)
        self.conv_5_2 = nn.Conv2d(out_channels, out_channels, kernel_size=5, stride=1, padding=2)
        

    def forward(self, x):
        
        ###+
        x1     = x # self.conv_1_1(x)
        x2     = x # self.conv_1_2(x)
        x3     = x # self.conv_1_3(x)
        
        x_3_1  = self.relu(self.conv_3_1(x2))  ## (BS, 32, ***, ***)
        x_5_1  = self.relu(self.conv_5_1(x3))  ## (BS, 32, ***, ***)
        
        x_3_2 = self.relu(self.conv_3_2(x_3_1 + x_5_1))  ## (BS, 64, ***, ***)
        x_5_2 = self.relu(self.conv_5_2(x_5_1 + x_3_1))  ## (BS, 64, ***, ***)
         
        x_mul = torch. mul(x_3_2, x_5_2)
        out   = self.relu(x1 + self.conv_1_5(x_mul + x_3_1 + x_5_1))

        return out
    
class MFAM(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(MFAM, self).__init__()
        
        self.relu = nn.ReLU(inplace=True)

        self.conv_1_1 = BasicConv2d(in_channels, out_channels, 1)
        self.conv_1_2 = BasicConv2d(in_channels, out_channels, 1)
        self.conv_1_3 = BasicConv2d(in_channels, out_channels, 1)
        self.conv_1_4 = BasicConv2d(in_channels, out_channels, 1)
        self.conv_1_5 = BasicConv2d(out_channels, out_channels, 3, stride=1, padding=1)
        
        self.conv_3_1 = nn.Conv2d(out_channels,   out_channels , kernel_size=3, stride=1, padding=1)
        self.conv_3_2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        
        self.conv_5_1 = nn.Conv2d(out_channels,   out_channels , kernel_size=5, stride=1, padding=2)
        self.conv_5_2 = nn.Conv2d(out_channels, out_channels, kernel_size=5, stride=1, padding=2)
        

    def forward(self, x):
        
        ###+
        x1     = self.conv_1_1(x)
        x2     = self.conv_1_2(x)
        x3     = self.conv_1_3(x)
        
        x_3_1  = self.relu(self.conv_3_1(x2))  ## (BS, 32, ***, ***)
        x_5_1  = self.relu(self.conv_5_1(x3))  ## (BS, 32, ***, ***)
        
        x_3_2  = self.relu(self.conv_3_2(x_3_1 + x_5_1))  ## (BS, 64, ***, ***)
        x_5_2  = self.relu(self.conv_5_2(x_5_1 + x_3_1))  ## (BS, 64, ***, ***)
         
        x_mul  = torch.mul(x_3_2, x_5_2)
        
        out    = self.relu(x1 + self.conv_1_5(x_mul + x_3_1 + x_5_1))
         
        return out    
    
    
###################################################################
class FeaFusion(nn.Module):
    def __init__(self, channels):
        self.init__ = super(FeaFusion, self).__init__()
        
        self.relu     = nn.ReLU()
        self.layer1   = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        
        self.layer2_1 = nn.Conv2d(channels, channels //4, kernel_size=3, stride=1, padding=1)
        self.layer2_2 = nn.Conv2d(channels, channels //4, kernel_size=3, stride=1, padding=1)
        
        self.layer_fu = nn.Conv2d(channels//4, channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x1, x2):
        
        ###
        wweight    = nn.Sigmoid()(self.layer1(x1+x2))
        
        ###
        xw_resid_1 = x1+ x1.mul(wweight)
        xw_resid_2 = x2+ x2.mul(wweight)
        
        ###
        x1_2       = self.layer2_1(xw_resid_1)
        x2_2       = self.layer2_2(xw_resid_2)
        
        out        = self.relu(self.layer_fu(x1_2 + x2_2))
        
        return out
    
###################################################################  
class FeaProp(nn.Module):
    def __init__(self, in_planes):
        self.init__ = super(FeaProp, self).__init__()
        

        act_fn = nn.ReLU(inplace=True)
        
        self.layer_1  = nn.Sequential(nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(in_planes),act_fn)
        self.layer_2  = nn.Sequential(nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(in_planes),act_fn)
        
        self.gate_1   = nn.Conv2d(in_planes*2, 1, kernel_size=1, bias=True)
        self.gate_2   = nn.Conv2d(in_planes*2, 1, kernel_size=1, bias=True)

        self.softmax  = nn.Softmax(dim=1)
        

    def forward(self, x10, x20):
        
        ###
        x1 = self.layer_1(x10)
        x2 = self.layer_2(x20)
        
        cat_fea = torch.cat([x1,x2], dim=1)
        
        ###
        att_vec_1  = self.gate_1(cat_fea)
        att_vec_2  = self.gate_2(cat_fea)

        att_vec_cat  = torch.cat([att_vec_1, att_vec_2], dim=1)
        att_vec_soft = self.softmax(att_vec_cat)
        
        att_soft_1, att_soft_2 = att_vec_soft[:, 0:1, :, :], att_vec_soft[:, 1:2, :, :]
        x_fusion = x1 * att_soft_1 + x2 * att_soft_2
        
        return x_fusion    
    
###################################################################      

class FAPNet(nn.Module):
    # resnet based encoder decoder
    def __init__(self, channel=32, opt=None):
        super(FAPNet, self).__init__()
        
        
        act_fn           = nn.ReLU(inplace=True)
        self.nf          = channel

        self.resnet      = res2net50_v1b_26w_4s(pretrained=True)
        self.downSample  = nn.MaxPool2d(2, stride=2)
        
        ##  
        self.rf1         = MFAM0(64,  self.nf)
        self.rf2         = MFAM(256,  self.nf)
        self.rf3         = MFAM(512,  self.nf)
        self.rf4         = MFAM(1024, self.nf)
        self.rf5         = MFAM(2048, self.nf)
        
        
        ##
        self.cfusion2    = FeaFusion(self.nf)
        self.cfusion3    = FeaFusion(self.nf)
        self.cfusion4    = FeaFusion(self.nf)
        self.cfusion5    = FeaFusion(self.nf)
        
        ##
        self.cgate5      = FeaProp(self.nf)
        self.cgate4      = FeaProp(self.nf)
        self.cgate3      = FeaProp(self.nf)
        self.cgate2      = FeaProp(self.nf)
        
        
        self.de_5        = nn.Sequential(nn.Conv2d(self.nf, self.nf, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(self.nf),act_fn)
        self.de_4        = nn.Sequential(nn.Conv2d(self.nf, self.nf, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(self.nf),act_fn)
        self.de_3        = nn.Sequential(nn.Conv2d(self.nf, self.nf, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(self.nf),act_fn)
        self.de_2        = nn.Sequential(nn.Conv2d(self.nf, self.nf, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(self.nf),act_fn)
        

        
        ##
        self.edge_conv0 = nn.Sequential(nn.Conv2d(64,       self.nf, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(self.nf),act_fn) 
        self.edge_conv1 = nn.Sequential(nn.Conv2d(256,      self.nf, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(self.nf),act_fn) 
        self.edge_conv2 = nn.Sequential(nn.Conv2d(self.nf,  self.nf, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(self.nf),act_fn) 
        self.edge_conv3 = BasicConv2d(self.nf,   1,  kernel_size=3, padding=1)
        
        
        self.fu_5        = nn.Sequential(nn.Conv2d(self.nf*2, self.nf, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(self.nf),act_fn)
        self.fu_4        = nn.Sequential(nn.Conv2d(self.nf*2, self.nf, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(self.nf),act_fn)
        self.fu_3        = nn.Sequential(nn.Conv2d(self.nf*2, self.nf, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(self.nf),act_fn)
        self.fu_2        = nn.Sequential(nn.Conv2d(self.nf*2, self.nf, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(self.nf),act_fn)
        
        
        ##
        self.layer_out5  = nn.Sequential(nn.Conv2d(self.nf, 1,  kernel_size=3, stride=1, padding=1))
        self.layer_out4  = nn.Sequential(nn.Conv2d(self.nf, 1,  kernel_size=3, stride=1, padding=1))
        self.layer_out3  = nn.Sequential(nn.Conv2d(self.nf, 1,  kernel_size=3, stride=1, padding=1))
        self.layer_out2  = nn.Sequential(nn.Conv2d(self.nf, 1,  kernel_size=3, stride=1, padding=1))
        
        
       
        ##
        self.up_2        = nn.Upsample(scale_factor=2,  mode='bilinear', align_corners=True)
        self.up_4        = nn.Upsample(scale_factor=4,  mode='bilinear', align_corners=True)
        self.up_8        = nn.Upsample(scale_factor=8,  mode='bilinear', align_corners=True)
        self.up_16       = nn.Upsample(scale_factor=16, mode='bilinear', align_corners=True)
        


    def forward(self, xx):
        
        # ---- feature abstraction -----
        x   = self.resnet.conv1(xx)
        x   = self.resnet.bn1(x)
        x   = self.resnet.relu(x)
        
        # - low-level features
        x1  = self.resnet.maxpool(x)       # (BS, 64, 88, 88)
        x2  = self.resnet.layer1(x1)       # (BS, 256, 88, 88)
        x3  = self.resnet.layer2(x2)       # (BS, 512, 44, 44)
        x4  = self.resnet.layer3(x3)     # (BS, 1024, 22, 22)
        x5  = self.resnet.layer4(x4)     # (BS, 2048, 11, 11)
        
        ## -------------------------------------- ##
        xf1 = self.rf1(x1)
        xf2 = self.rf2(x2)
        xf3 = self.rf3(x3)
        xf4 = self.rf4(x4)
        xf5 = self.rf5(x5)
        
        
        ## edge 
        x21           = self.edge_conv1(x2)
        edge_guidance = self.edge_conv2(self.edge_conv0(x1) + x21)
        edge_out      = self.up_4(self.edge_conv3(edge_guidance))
        

        ### layer 5
        en_fusion5   = self.cfusion5(self.up_2(xf5), xf4)              ## (BS, 64, 22, 22)
        out_gate_fu5 = self.fu_5(torch.cat((en_fusion5, F.interpolate(edge_guidance, scale_factor=1/4, mode='bilinear')),dim=1))
        out5         = self.up_16(self.layer_out5(out_gate_fu5))
        
        
        de_feature4  = self.de_4(self.up_2(en_fusion5))                       ## (BS, 64, 22, 22)
        en_fusion4   = self.cfusion4(self.up_2(xf4), xf3)              ## (BS, 64, 44, 44)
        out_gate4    = self.cgate4(en_fusion4, de_feature4) ## (BS, 64, 44, 44) 
        out_gate_fu4 = self.fu_4(torch.cat((out_gate4, F.interpolate(edge_guidance, scale_factor=1/2, mode='bilinear')),dim=1))
        out4         = self.up_8(self.layer_out4(out_gate_fu4))
        
        
        de_feature3  = self.de_3(self.up_2(out_gate4))                 ## (BS, 64, 88, 88)
        en_fusion3   = self.cfusion3(self.up_2(xf3), xf2)              ## (BS, 64, 88, 88)
        out_gate3    = self.cgate3(en_fusion3, de_feature3)            ## (BS, 64, 88, 88)  
        out_gate_fu3 = self.fu_3(torch.cat((out_gate3, edge_guidance),dim=1))
        out3         = self.up_4(self.layer_out3(out_gate_fu3))
        
        
        de_feature2  = self.de_2(self.up_2(out_gate3))                 ## (BS, 64, 176, 176)
        en_fusion2   = self.cfusion2(self.up_2(xf2), self.up_2(xf1))   ## (BS, 64, 176, 176)
        out_gate2    = self.cgate2(en_fusion2, de_feature2)            ## (BS, 64, 176, 176)  
        out_gate_fu2 = self.fu_2(torch.cat((out_gate2, self.up_2(edge_guidance)), dim=1))
        out2         = self.up_2(self.layer_out2(out_gate_fu2))

        
        # ---- output ----
        return out5
        return out5, out4, out3, out2, edge_out


if __name__ == '__main__':
    ras = FAPNet().cuda()
    input_tensor = torch.randn(4, 3, 352, 352).cuda()
    out = ras(input_tensor)
    print(out.shape)