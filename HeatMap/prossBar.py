import os
from PIL import Image
import glob
import cv2
import sys
import time
import progressbar


# p = progressbar.ProgressBar()
# N = 1000
# p.start(N)
# for i in range(N):
#     time.sleep(0.01)
#     p.update(i+1)
# p.finish()

def get_image_files(root_dir):
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.gif', '*.bmp', '*.tif']:
        image_files.extend(glob.glob(os.path.join(root_dir, '**', ext), recursive=True))
    return image_files


bar = progressbar.ProgressBar(widgets=[progressbar.Timer(),
    progressbar.Percentage(),
    ' (', progressbar.SimpleProgress(), ') ',
    ' (', progressbar.ETA(), ') ',
    ' (', progressbar.AbsoluteETA(), ') '])

print("finding")
# 指定要遍历的根目录
root_directory2 = 'D:/SHU/PyTorch/polyp segmentation dataset/sundatabase_positive_part1/'
root_directory = 'D:/SHU/PyTorch/polyp segmentation dataset/CVC-ClinicDB/CVC-ClinicDB/Ground Truth/'
# root_directory= 'D:/SHU/PhD/PPSN Prompt Polyp Segmentation Network/HeatMap/Ground Truth/'
root_directory = 'D:/SHU/PyTorch/polyp segmentation dataset/kvasir-sessile/sessile-main-Kvasir-SEG/masks/'
root_directory ='S:/PyTorch/PPSN-main/SUN/GT/'
root_directory ='G:/PyTorch/polyp segmentation dataset/SUN-SEG-Annotation/TestHardDataset/Seen/'


img_path = os.path.join(root_directory, "GT", "**", "*")
image_files = sorted(glob.glob(img_path))
#image_files = get_image_files(root_directory)



for image_file in bar(image_files):
    print(image_file)
    image = cv2.imread(image_file)
#     len(image_files)
#
# for i in bar(range(1000)):
#     time.sleep(0.01)