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

class PEM(nn.Module):
    def __init__(self):
        super(PEM,self).__init__()
        self.conv1=nn.Conv2d(5,1,3)
        self.conv2 = nn.Conv2d(1, 1, 3)
    def forward(self,x):
        x1=F.relu(conv1(x))
        x2=F.relu(conv2(x1))
        return x2
class FEEM(nn.Module):
    def __init__(self):
        super(FEEM,self).__init__()
        #256
        self.conv1=nn.Conv2d(3,64,3,1,1)
        self.conv2 = nn.Conv2d(64, 64, 3,1,1)
        self.MP1= nn.MaxPool2d(2, 2)
        #128
        self.conv3 = nn.Conv2d(64, 128, 3, 1, 1)
        self.conv4 = nn.Conv2d(128, 128, 3, 1, 1)
        self.MP2 = nn.MaxPool2d(2, 2)
        #64
    def forward(self,x,p1,p2):
        x1=self.conv1(x)
        x2=self.conv2(x1)
        x2 = self.MP1(x2)
        x3=torch.concat(x2,p1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x4 = self.MP2(x4,p2)
        return x4
class MDM(nn.Module):
    def __init__(self):
        super(MDM,self).__init__()
        self.conv1=nn.ConvTranspose2d(256,128,3)
        self.conv2 = nn.ConvTranspose2d(128, 64, 3)
        self.conv3 = nn.ConvTranspose2d(64, 3, 3)
    def forward(self,x,p1,p2):
        x1=F.relu(self.conv1(x))
        x1=torch.concat(x1,p1)
        x2=F.relu(self.conv2(x1))
        x2 = torch.concat(x2, p2)
        x3=F.relu(self.conv3(x2))
        return x3

class PPSN(pl.LightningModule):
    def __init__(self,PEM,FEEM,MDM):
        super().__init__()
        self.PEM=PEM
        self.FEEM=FEEM
        self.MDM=MDM
    def forward(self, x,p0) -> Any:
        p2=self.PEM(p0)
        p1=self.PEM.x1
        x1=self.FEEM(x,p1,p2)
        x3=self.MDM(x1)
        return x3
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

#hyperparameters
input_size=704
num_classer=1
learning_rate=0.001
batch_size=64
num_epochs=3
#load
trainer=pl.Trainer(accelerator="gpu",devices=[0],min_epochs=1,max_epochs=3,precision=16)
trainer.fit(model,train_loder,val_loder)





#
# class NN(pl.LightningModule):
#     def __init__(self,input_size,num_classes):
#         super().__init__()
#         self.fc1=nn.Linear(input_size,50)
#         self.fc2=nn.Linear(50,num_classes)
#         self.loss_fn=nn.CrossEntropyLoss()
#     def forward(self, x) -> Any:
#         x=F.relu(self.fc1(x))
#         x=self.fc2(x)
#         return x
#     def _common_step(self,batch,batch_idx):
#         x,y=batch
#         x=x.reshape(x.size(0),-1)
#         scores=self.forward(x)
#         loss=self.loss_fn(scores,y)
#         return loss,scores,y
#     def training_step(self,batch,batch_idx) -> STEP_OUTPUT:
#         loss, scores,y =self._common_step(batch,batch_idx)
#         self.log("train_loss",loss)
#         return loss
#     def validation_step(self,batch,batch_idx ):
#         loss, scores, y = self._common_step(batch, batch_idx)
#         self.log("validation_loss", loss)
#         return loss
#     def test_step(self,batch,batch_idx ):
#         loss, scores, y = self._common_step(batch, batch_idx)
#         self.log("test_loss", loss)
#         return loss
#     def predict_step(self, batch,batch_idx) -> Any:
#         x,y=batch
#         x=x.reshape(x.size(0)-1)
#         scores=self.forward(x)
#         preds=torch.argmax(scores,dim=1)
#         return preds
#     def configure_optimizers(self) -> OptimizerLRScheduler:
#         return optim.Adam(self.parameters(),lr=0.001)
#
# #hyperparameters
# input_size=704
# num_classer=1-
# learning_rate=0.001
# batch_size=64
# num_epochs=3
# #load
# trainer=pl.Trainer(accelerator="gpu",devices=[0],min_epochs=1,max_epochs=3,precision=16)
# trainer.fit(model,train_loder,val_loder)




