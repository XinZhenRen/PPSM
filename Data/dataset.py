import random
import sys

import numpy as np

sys.path.append("..")
from skimage.io import imread
import os
import glob
import torch
from torch.utils import data
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
#from train import get_args as config

def get_image_files(root_dir):
    image_files=[]
    for ext in ['*.jpg','*.jpeg','*.png','*.tif','*.bmp','*.gif']:
        image_files.extend(glob.glob(os.path.join(root_dir,'**',ext),recursive=True))
    return image_files

class VideoDataset(Dataset):
    def __init__(self, video_dataset, transform=None, time_interval=1):
        super(VideoDataset, self).__init__()
        with open("Print.txt",'w') as file0:
            #self.time_clips = config.video_time_clips
            self.time_clips = 5
            self.video_train_list = []

            #video_root = os.path.join(config.dataset_root, video_dataset)
            video_root = os.path.join("/home/xinzhen/VPS-main/data/SUN-SEG", video_dataset)
            img_root = os.path.join(video_root, 'Frame')
            gt_root = os.path.join(video_root, 'GT')
            Polygon_root=os.path.join(video_root, 'Polygon')
            Scribble_root=os.path.join(video_root, 'Scribble')
            Edge_root=os.path.join(video_root, 'Edge')
            #print("img_root:",img_root,file=file0)
            #print("gt_root:", gt_root,file=file0)
            cls_list = os.listdir(img_root)#Frame/class1....class99   cls means "class"
            self.video_filelist = {}#dictionary
            for cls in cls_list:#cls=class1.....
                self.video_filelist[cls] = []
                cls_img_path = os.path.join(img_root, cls)#Frame/case1-1
                cls_label_path = os.path.join(gt_root, cls)#GT/case1-1
                cls_Edge_path = os.path.join(Edge_root, cls)
                cls_Polygon_path = os.path.join(Polygon_root, cls)
                cls_Scribble_path = os.path.join(Scribble_root, cls)
                tmp_list = os.listdir(cls_img_path)#Frame/case1-1/p1.jpg......p99.jpg

                #print("cls_img_path:", cls_img_path, file=file0)
                #print("cls_label_path:", cls_label_path, file=file0)
                #print("tmp_list_old:", tmp_list, file=file0)
                #case1-1/
                #case_M_20181001100941_0U62372100109341_1_005_001-1_a2_ayy_image0001.png
                #case_M_20181001100941_0U62372100109341_1_005_001-1_a2_ayy_image0002.png
                #....
                #case_M_20181001100941_0U62372100109341_1_005_001-1_a2_ayy_image0152.png
                #case1-3
                #case_M_20181001100941_0U62372100109341_1_005_001-1_a10_ayy_image0001.png
                #case_M_20181001100941_0U62372100109341_1_005_001-1_a10_ayy_image0002.png
                #....
                #case_M_20181001100941_0U62372100109341_1_005_001-1_a10_ayy_image0161.png
                tmp_list.sort(key=lambda name: (
                    int(name.split('-')[0].split('_')[-1]),#001
                    int(name.split('_a')[1].split('_')[0]),#2
                    int(name.split('_image')[1].split('.jpg')[0])))#0001
                # 001-2-0001 ->001-2-0002 ......001-3-0001.....002-1-0001
                #print("tmp_list_new:", tmp_list,file=file0)
                for filename in tmp_list:

                    self.video_filelist[cls].append((
                        os.path.join(cls_img_path, filename),
                        os.path.join(cls_label_path, filename.replace(".jpg", ".png")),
                        os.path.join(cls_Edge_path, filename.replace(".jpg", ".png")),
                        os.path.join(cls_Polygon_path, filename.replace(".jpg", ".png")),
                        os.path.join(cls_Scribble_path, filename.replace(".jpg", ".png")),
                    ))
            #print("\n\nself.video_filelist_shape:", self.video_filelist., file=file0)
            #print("self.video_filelist:", self.video_filelist, file=file0)
            # ensemble   ????    clips->pian duan  interval->jian ge
            for cls in cls_list:
                li = self.video_filelist[cls]
                for begin in range(1,len(li) - (self.time_clips - 1) * time_interval - 1):
                    batch_clips = []
                    batch_clips.append(li[0])
                    for t in range(self.time_clips):
                        batch_clips.append(li[begin + time_interval * t])
                    self.video_train_list.append(batch_clips)
            print("self.video_train_list:", self.video_train_list, file=file0)
            self.img_label_transform = transform

    def __getitem__(self, idx):
        img_label_li = self.video_train_list[idx]
        IMG = None
        LABEL = None
        img_li = []
        label_li = []
        for idx, (img_path, label_path) in enumerate(img_label_li):
            img = Image.open(img_path).convert('RGB')
            label = Image.open(label_path).convert('L')
            img_li.append(img)
            label_li.append(label)
        img_li, label_li = self.img_label_transform(img_li, label_li)
        for idx, (img, label) in enumerate(zip(img_li, label_li)):
            if idx == 0:
                IMG = torch.zeros(len(img_li), *(img.shape))
                LABEL = torch.zeros(len(img_li) - 1, *(label.shape))

                IMG[idx, :, :, :] = img
            else:
                IMG[idx, :, :, :] = img
                LABEL[idx - 1, :, :, :] = label

        return IMG, LABEL

    def __len__(self):
        return len(self.video_train_list)
class SegDataset(data.Dataset):
    def __init__(
        self,
        input_paths: list,
        target_paths: list,
        transform_input=None,
        transform_target=None,
        hflip=False,
        vflip=False,
        affine=False,
    ):
        self.input_paths = input_paths
        self.target_paths = target_paths
        self.transform_input = transform_input
        self.transform_target = transform_target
        self.hflip = hflip
        self.vflip = vflip
        self.affine = affine

    def __len__(self):
        return len(self.input_paths)


    def __getitem__(self, index: int):
        input_ID = self.input_paths[index]
        target_ID = self.target_paths[index]


        x, y = imread(input_ID), imread(target_ID)

        x = self.transform_input(x)
        y = self.transform_target(y)

        if self.hflip:
            if random.uniform(0.0, 1.0) > 0.5:
                x = TF.hflip(x)
                y = TF.hflip(y)

        if self.vflip:
            if random.uniform(0.0, 1.0) > 0.5:
                x = TF.vflip(x)
                y = TF.vflip(y)

        if self.affine:
            angle = random.uniform(-180.0, 180.0)
            h_trans = random.uniform(-352 / 8, 352 / 8)
            v_trans = random.uniform(-352 / 8, 352 / 8)
            scale = random.uniform(0.5, 1.5)
            shear = random.uniform(-22.5, 22.5)
            x = TF.affine(x, angle, (h_trans, v_trans), scale, shear, fill=-1.0)
            y = TF.affine(y, angle, (h_trans, v_trans), scale, shear, fill=0.0)
        return x.float(), y.float()


class PromptDataset(data.Dataset):
    def __init__(self,
                 input_paths: list,
                 target_paths: list,
                 prompt_path: str,
                 transform_input=None,
                 transform_target=None,
            transform_prompt=None,
        hflip=False,vflip=False,affine=False,
    ):

        self.input_paths = input_paths
        self.target_paths = target_paths
        self.prompt_path = prompt_path

        self.transform_input = transform_input
        self.transform_target = transform_target
        self.transform_prompt = transform_prompt
        self.hflip = hflip
        self.vflip = vflip
        self.affine = affine

        assert os.path.exists(self.prompt_path), f"path '{self.prompt_path}' does not exists"

        self.edge = self.prompt_path + "Circumcircle/*"
        self.point = self.prompt_path + "Point/*"
        self.polygon = self.prompt_path + "Polygon/*"
        self.scribble = self.prompt_path + "Scribble/*"
        self.sam = self.prompt_path + "Sam/*"


        #print(self.edge)

        self.edge_paths=sorted(glob.glob(self.edge))
        self.point_paths = sorted(glob.glob(self.point))
        self.polygon_paths = sorted(glob.glob(self.polygon))
        self.scribble_paths = sorted(glob.glob(self.scribble))
        self.sam_paths = sorted(glob.glob(self.sam))#[sam/case1/99232.png,]

        #print(self.edge_paths)


    def __len__(self):
        return len(self.input_paths)

    def __getitem__(self, index:int):
        input_ID = self.input_paths[index]
        target_ID = self.target_paths[index]
        edge_ID=self.edge_paths[index]
        point_ID = self.point_paths[index]
        polygon_ID = self.polygon_paths[index]
        scribble_ID = self.scribble_paths[index]
        sam_ID = self.sam_paths[index]

        x = imread(input_ID)
        y = imread(target_ID)
        prompt_edge=imread(edge_ID)
        prompt_point=imread(point_ID)
        prompt_polygon=imread(polygon_ID)
        prompt_scribble=imread(scribble_ID)
        prompt_sam=imread(sam_ID)

        x = self.transform_input(x)
        y = self.transform_target(y)
        prompt_edge = self.transform_prompt(prompt_edge)
        prompt_point = self.transform_prompt(prompt_point)
        prompt_polygon = self.transform_prompt(prompt_polygon)
        prompt_scribble = self.transform_prompt(prompt_scribble)
        prompt_sam = self.transform_prompt(prompt_sam)

        prompts=torch.cat((prompt_edge,prompt_point,prompt_polygon,prompt_scribble,prompt_sam),dim=0)
        #print(prompts.shape)

        if self.hflip:
            if random.uniform(0.0, 1.0) > 0.5:
                prompts = TF.hflip(prompts)
                x = TF.hflip(x)
                y = TF.hflip(y)
        if self.vflip:
            if random.uniform(0.0, 1.0) > 0.5:
                prompts = TF.vflip(prompts)
                x = TF.vflip(x)
                y = TF.vflip(y)
        if self.affine:
            angle = random.uniform(-180.0, 180.0)
            h_trans = random.uniform(-352 / 8, 352 / 8)
            v_trans = random.uniform(-352 / 8, 352 / 8)
            scale = random.uniform(0.5, 1.5)
            shear = random.uniform(-22.5, 22.5)
            prompts = TF.affine(prompts, angle, (h_trans, v_trans), scale, shear, fill=-1.0)
            x = TF.affine(x, angle, (h_trans, v_trans), scale, shear, fill=-1.0)
            y = TF.affine(y, angle, (h_trans, v_trans), scale, shear, fill=0.0)
            #y = TF.affine(y, angle, (h_trans, v_trans), scale, shear, fill=0.0) why fill is 0 and upper fill=-1.fill means pixel will filled with -1.0 or 0
        return x.float(), prompts.float(), y.float()

if __name__ == '__main__':
    prompt_path="../SUN-SEG/TrainDataset/"
    transform_prompt = transforms.Compose(
            [transforms.ToTensor(), transforms.Resize((352, 352)),transforms.Grayscale()]
            # [transforms.ToTensor(), transforms.Resize((352, 352))]
        )
    hflip = True
    vflip = True
    affine = True

    Prompt_dataloader=PromptDataset(prompt_path,transform_prompt,hflip,vflip,affine)
    p=Prompt_dataloader.__getitem__(0)
    print(p)



