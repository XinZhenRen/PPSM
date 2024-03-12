import os
from PIL import Image
import glob
import cv2
import numpy as np
# import seaborn as sns
import matplotlib.pyplot as plt
import datetime
import progressbar

dataName="Kvasir-SEG"
# 定义一个函数，用于递归地获取目录下的所有图片文件
def get_image_files(root_dir):
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.gif', '*.bmp', '*.tif']:
        image_files.extend(glob.glob(os.path.join(root_dir, '**', ext), recursive=True))
    return image_files

def findCenterPoint2coordinate():
    print("finding")
    # 指定要遍历的根目录
    root_directory2 = 'D:/SHU/PyTorch/polyp segmentation dataset/sundatabase_positive_part1/'
    root_directory ='D:/SHU/PyTorch/polyp segmentation dataset/hyper-kvasir/segmented-images/masks/'
    root_directory = 'D:/SHU/PyTorch/polyp segmentation dataset/MICCAI-VPS-dataset/MICCAI-VPS-dataset/IVPS-TrainSet/GT/'
    root_directory = 'S:/PyTorch/PPSN-main/SUN/GT/'
    root_directory = 'D:/SHU/PyTorch/polyp segmentation dataset/kvasir-sessile/sessile-main-Kvasir-SEG/masks/'
    root_directory = 'D:/SHU/PyTorch/polyp segmentation dataset/CVC-ClinicDB/CVC-ClinicDB/Ground Truth/'
    root_directory = 'D:/SHU/PyTorch/polyp segmentation dataset/kvasir-seg/Kvasir-SEG/masks/'
    root_directory ='Z:/XinzhenRen/PPSN-main/polypGen2021_MultiCenterData_v3/imagesAll_mask/'
    #root_directory= 'D:/SHU/PhD/PPSN Prompt Polyp Segmentation Network/HeatMap/Ground Truth/'


    bar = progressbar.ProgressBar(widgets=[progressbar.Timer(),
                                           progressbar.Percentage(),
                                           ' (', progressbar.SimpleProgress(), ') ',
                                           ' (', progressbar.ETA(), ') ',
                                           ' (', progressbar.AbsoluteETA(), ') '])
    # 获取所有图片文件
    image_files = get_image_files(root_directory)
    print(image_files)
    textPicture=cv2.imread(image_files[0])
    # cv2.imshow('textPicture',textPicture)
    # cv2.waitKey(0)
    height, width, _ = textPicture.shape #height=row

    centerData = []
    #print(len(image_files))
    i=0
    maxX=0
    maxY=0
    # 遍历图片文件并进行处理
    # for image_file in bar(image_files):
    #     # 打开图片文件
    #     image = cv2.imread(image_file)
    #     height, width, _ = image.shape
    #     if maxX<width:
    #         maxX=width
    #     if maxY<height:
    #         maxY=height
    # print("maxY",maxY)
    # print("maxX", maxX)
    #pred = np.zeros((maxY, maxX))
    size=1000
    pred = np.zeros((size, size))
    print("pred",pred)
    #i=0
    for image_file in bar(image_files):
        # print(i,"//",len(image_files))
        # i+=1
        try:
            # 转换为灰度图像
            image = cv2.imread(image_file)
            #print(image.shape)
            #image=cv2.resize(image,(maxX,maxY))#转成最大的
            image = cv2.resize(image, (size, size))#转成400成400
            #print(image.shape)
            # cv2.imshow('image',image)
            # cv2.waitKey(0)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # 二值化处理
            _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)
            for row  in range(image.shape[0]):
                for col in range(image.shape[1]):
                    if binary[row,col]==255:
                        pred[row,col]=pred[row,col]+1.0

            # 查找轮廓
            # contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            #
            # # 初始化最亮点坐标
            # brightest_point = None
            #
            #
            # # 遍历每个轮廓
            # for contour in contours:
            #     # 计算轮廓的中心坐标
            #     M = cv2.moments(contour)
            #     if M["m00"] != 0:
            #         cX = int(M["m10"] / M["m00"])
            #         cY = int(M["m01"] / M["m00"])
            #
            #         # 更新最亮点坐标
            #         if brightest_point is None:
            #             brightest_point = (cX, cY)
            #         else:
            #             # 比较亮度，更新为最亮的点
            #             if gray[cY, cX] > gray[brightest_point[1], brightest_point[0]]:
            #                 brightest_point = (cX, cY)
            # # 画中心/记录中心点
            # if brightest_point is not None:
            #     centerData.append(brightest_point)
        except Exception as e:
            # 处理图片时发生异常的情况
            print(f"Error processing image {image_file}: {str(e)}")

    # numPycenterData = np.array(centerData, dtype=np.int32)
    # print("maxX:",maxX)
    # print("maxY:", maxY)
    #return numPycenterData,maxY,maxX
    print(pred)
    current_time = datetime.datetime.now()
    TxtName = dataName + current_time.strftime("%Y%m%d_%H%M%S") + ".txt"
    np.savetxt(TxtName, pred)  # 如果文件路径末尾没有扩展名.npy，该扩展名会被自动加上。
    return pred


def drawHeart(data,maxY,maxX):
    print("drawHeart")

    # heatmapshow = None
    # heatmapshow = cv2.normalize(data, heatmapshow, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    # heatmapshow = cv2.applyColorMap(heatmapshow, cv2.COLORMAP_JET)
    # cv2.imshow("Heatmap", hetmapshow)
    # cv2.waitKey(0)
    width, height = maxX,maxY

    # 创建一个全白色的图像（白色在BGR颜色空间中是(255, 255, 255)）
    white_image = np.full((height, width, 3), (255, 255, 255), dtype=np.uint8)
    for i in data:
        x=i[0]
        y=i[1]
        print(i)
        for xx in range(-10,10):
            for yy in range(-10,10):

                blue, green, red = white_image[y+yy, x+xx]
                red -= 1
                blue -= 1
                green -= 1
                white_image[y, x] = [blue, green, red]

    cv2.imshow('White Image', white_image)
    cv2.waitKey(0)
    current_time = datetime.datetime.now()
    file_name = current_time.strftime("%Y%m%d_%H%M%S") + '.png'
    cv2.imwrite(file_name, white_image)
    cv2.destroyAllWindows()

def draw3Heart(data,maxX,maxY):
    X, Y = np.meshgrid(np.arange(maxX), np.arange(maxY))
    Z=np.zeros((maxX, maxY))

    for i in data:
        print("data",i)
        print("data[0]", i[0])
        Z[i[0],i[1]]=Z[i[0],i[1]]+1;

    print("Z:",Z)
    # 创建一个三维图
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # 绘制热力图
    ax.plot_surface(X, Y, Z, cmap='viridis')

    # 设置轴标签
    ax.set_xlabel('X轴')
    ax.set_ylabel('Y轴')
    ax.set_zlabel('像素强度')

    # 显示图形
    plt.show()


def textNparray(pred):
    # np.random.seed(0)
    # pred = np.random.random_sample((1024, 1024))
    plt.imshow(pred, cmap=plt.cm.jet,interpolation='bicubic')#'hot'   plt.cm.jet 'RdBu''viridis'
    current_time = datetime.datetime.now()
    file_name = "Kvasir-SEG"+current_time.strftime("%Y%m%d_%H%M%S") + ".svg"
    file_name = "CVC-ClinicDB" + current_time.strftime("%Y%m%d_%H%M%S") + ".svg"
    file_name = "kvasir-sessile" + current_time.strftime("%Y%m%d_%H%M%S") + ".svg"
    file_name = "SUN" + current_time.strftime("%Y%m%d_%H%M%S") + ".svg"
    file_name = "MICCAI-VPS-dataset" + current_time.strftime("%Y%m%d_%H%M%S") + ".svg"
    file_name = "hyper-kvasir" + current_time.strftime("%Y%m%d_%H%M%S") + ".svg"
    file_name = dataName + current_time.strftime("%Y%m%d_%H%M%S") + ".svg"
    plt.axis('off')
    plt.margins(0, 0)
    plt.savefig(file_name,bbox_inches='tight',dpi=300, pad_inches=0.0)
    plt.show()








if __name__ == '__main__':

    ppre  = findCenterPoint2coordinate()
    textNparray(ppre)
    # # draw3Heart(data,maxX, maxY)
    # drawHeart(data, maxY,maxX)

