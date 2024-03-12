import os
from PIL import Image
import glob
import cv2
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import datetime


mapWidth=512
mapHeight=512
reach = 25;
valueRange = 100;
class HPoint:
    def __int__(self,x,y,value):
        self.x=x
        self.y=y
        self.value=value

class HRect:
    def __int__(self,left ,top ,right,bottom  ):
        self.left=left
        self.top=top
        self.right=right
        self.bottom = bottom


