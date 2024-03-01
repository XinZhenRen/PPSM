from typing import Any, Optional
import torch
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transform
from pytorch_lightning.utilities.types import STEP_OUTPUT, OptimizerLRScheduler
from torch import nn,optim
from torch.utils.data import  DataLoader
from tqdm import tqdm
from torch.utils.data import random_split
import pytorch_lightning as pl
from typing import Dict
class DoubleConv(nn.Sequential):
    def __init__(self,in_channels,out_channels,mid_channels=None):
        if mid_channels is None:
            mid_channels=out_channels
        super(DoubleConv,self).__init__(
            nn.Conv2d(in_channels,mid_channels,3,1,1,bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels,out_channels,3,1,1,bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

class Down(nn.Sequential):
    def __init__(self,in_channels,out_channels):
        super(Down,self).__init__(
            nn.MaxPool2d(2,2),
            DoubleConv(in_channels,out_channels)
        )
class Up(nn.Module):
    def __init__(self,in_channels,out_channels,bilinear=True):
        super(Up,self).__init__()
        if bilinear:
            self.up=nn.Upsample(scale_factor=2,mode='bilinear',align_corners=True)
            self.conv=DoubleConv(in_channels,out_channels,in_channels//2)
        else:
            self.up=nn.ConvTranspose2d(in_channels,in_channels//2,kernel_size=2,stride=2)
            self.conv=DoubleConv(in_channels,out_channels)
    def forward(self,x1:torch.Tensor,x2:torch.Tensor)->torch.Tensor:
        x1= self.up(x1)
        #[N,C,H,W]
        diff_y=x2.size()[2]-x1.size()[2]
        diff_x=x2.size()[3]-x1.size()[3]
        #padding_left padding_right padding_top padding_bootom
        x1=F.pad(x1,[diff_x//2,diff_x-diff_x//2,
                     diff_y//2,diff_y-diff_y//2])
        x=torch.cat([x2,x1],dim=1)
        x=self.conv(x)
        return x

class OutConv(nn.Sequential):
    def __init__(self,in_channels,num_classes):
        super(OutConv,self).__init__(
            nn.Conv2d(in_channels,num_classes,kernel_size=1)
        )
class UNet(pl.LightningModule):
    def __init__(self,
                 in_channels:int=3,
                 num_classes:int=2,
                 bilinear:bool=True,
                 base_c:int =64):
        super(UNet,self).__init__()
        self.in_channels=in_channels
        self.num_classes=num_classes
        self.bilinear=bilinear
        self.base_c = base_c

        self.in_conv=DoubleConv(in_channels=self.in_channels,out_channels=self.base_c)
        self.down1=Down(in_channels=self.base_c,out_channels=2*self.base_c)
        self.down2=Down(in_channels=2*self.base_c,out_channels=4*self.base_c)
        self.down3 = Down(in_channels=4 * self.base_c, out_channels=8 * self.base_c)
        factor=2 if self.bilinear else 1
        self.down4 = Down(in_channels=8 * self.base_c, out_channels= self.self.base_c*16//2)
        self.up1=Up(in_channels=self.base_c*16,out_channels=self.base_c*8//factor,bilinear=self.bilinear)
        self.up2=Up(in_channels=self.base_c*8,out_channels=self.base_c*4//factor,bilinear=self.bilinear)
        self.up3 = Up(in_channels=self.base_c * 4, out_channels=self.base_c * 2 // factor, bilinear=self.bilinear)
        self.up4 = Up(in_channels=self.base_c * 2, out_channels=self.base_c, bilinear=self.bilinear)
        self.out_conv=OutConv(self.base_c,self.num_classes)
    def forward(self,x:torch.Tenseor)->Dict[str,torch.Tensor]:
        x1=self.in_conv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x3)
        x=self.up1(x5,x4)
        x=self.up1(x,x3)
        x = self.up1(x, x2)
        x = self.up1(x, x1)
        logits=self.out_conv(x)
        return {"out",logits}
    def _common_step(self,batch,batch_idx):
        p,x,y=batch
        #x=x.reshape(x.size(0),-1)
        scores=self.forward(x,p)
        loss=self.loss_fn(scores,y)
        return loss,scores,y
    def training_step(self,batch,batch_idx) -> STEP_OUTPUT:
        loss, scores,y =self._common_step(batch,batch_idx)
        self.log("train_loss",loss)
        return loss
    def validation_step(self,batch,batch_idx ):
        loss, scores, y = self._common_step(batch, batch_idx)
        self.log("validation_loss", loss)
        return loss
    def test_step(self,batch,batch_idx ):
        loss, scores, y = self._common_step(batch, batch_idx)
        self.log("test_loss", loss)
        return loss
    def predict_step(self, batch,batch_idx) -> Any:
        x,y=batch
        x=x.reshape(x.size(0)-1)
        scores=self.forward(x)
        preds=torch.argmax(scores,dim=1)
        return preds
    def configure_optimizers(self) -> OptimizerLRScheduler:
        return optim.Adam(self.parameters(),lr=0.001)





