import numpy as np
import matplotlib.pyplot as plt
import progressbar
import datetime
import cv2

fillName=[
"uniform"
    ,
    "kvasir-sessile20231126_160357.txt",
          "SUN20231126_154955.txt",
          "Kvasir-SEG20231126_172709.txt",
          "CVC-ClinicDB20231126_161519.txt",
          "IVPS-TrainSet20231126_133434.txt",
          "hyper-kvasir20231126_115149.txt",
          "polypGen20240311_204043.txt"
]
dName=["uniform"
    ,
    "kvasir-sessile",
"SUN",
"Kvasir-SEG",
"CVC-ClinicDB",
"MICCVI",
"Hyper-Kvasir",
       "polypGen"
]
for data_array,dataName in zip(fillName,dName):
    bar = progressbar.ProgressBar(widgets=[progressbar.Timer(),
                                           progressbar.Percentage(),
                                           ' (', progressbar.SimpleProgress(), ') ',
                                           ' (', progressbar.ETA(), ') ',
                                           ' (', progressbar.AbsoluteETA(), ') '])

    # 假设normalized_data是你的归一化数据
    # normalized_data = np.random.rand(10, 10)
    # data_array=np.loadtxt("kvasir-sessile20231126_160357.txt")
    # data_array=np.loadtxt("SUN20231126_154955.txt")
    # data_array=np.loadtxt("Kvasir-SEG20231126_172709.txt")
    # data_array=np.loadtxt("CVC-ClinicDB20231126_161519.txt")
    # data_array=np.loadtxt("IVPS-TrainSet20231126_133434.txt")
    # data_array=np.loadtxt("hyper-kvasir20231126_115149.txt")
    if data_array=="uniform":
        data_array=np.full((400, 400), 0.5)
        plt.imshow(data_array, cmap=plt.cm.jet, interpolation='bicubic')  # 'hot'   plt.cm.jet 'RdBu'
        # plt.show()
        plt.xticks([])  # 去 x 轴刻度
        plt.yticks([])  # 去 y 轴刻度
        plt.savefig("Uniform.png", bbox_inches='tight', pad_inches=0)  # 保存为 PNG 格式
        normalized_data = data_array
    else:
        data_array = np.loadtxt(data_array)
        min_values = data_array.min()
        max_values = data_array.max()
        if max_values == min_values:
            max_values = min_values + 1
        normalized_data = (data_array - min_values) / (max_values - min_values)
    # dataName="kvasir-sessile"
    # dataName="SUN"
    # dataName="Kvasir-SEG"
    # dataName="CVC-ClinicDB"
    # dataName="IVPS"
    # dataName="Hyper-Kvasir"
    # 设置采样 5代表没各个像素采样一次
    # plt.imshow(data_array, cmap='viridis', interpolation='nearest')
    # plt.colorbar()  # 添加颜色条
    # plt.xlabel('X Label')
    # plt.ylabel('Y Label')
    # plt.title('Title')
    # plt.show()

    fs = 5
    zoom = 20
    allFlage=True
    larg=10



    # mean_val=normalized_data.mean()
    # std_val = normalized_data.std()
    # normalized_data=(normalized_data-mean_val)/std_val

    # plt.imshow(normalized_data, cmap='viridis', interpolation='nearest')
    # plt.colorbar()  # 添加颜色条
    # plt.xlabel('X Label')
    # plt.ylabel('Y Label')
    # plt.title('Title')
    # plt.show()

    print(normalized_data)

    # 提取数组形状
    rows, cols = normalized_data.shape
    print("rows:", rows)
    print("cols:", cols)

    # 设置图像大小
    img = np.zeros((rows*larg, cols*larg, 3), np.uint8)
    # 遍历数组，在每个位置生成随机数，决定是否在该位置绘制点
    for i in bar(range(int(rows / fs))):
        for j in range(int(cols / fs)):
            if i==0 or j==0:
                continue
            x = j * fs*larg
            y = i * fs*larg
            #cv2.circle(img, (int(x), int(y)), int(zoom), (255, 255, 255), -1)
            # 生成0到1的随机数，如果小于概率密度函数的值，就在该位置绘制点
            if not allFlage:
                if np.random.rand() < normalized_data[i * fs, j * fs] or normalized_data[i * fs, j * fs] >= 0.8:
                # if   normalized_data[i * fs, j * fs]>=0.9:
                    #size = normalized_data[i * fs, j * fs] * zoom  # 随机生成点的大小
                    size = ((1 / (1 + np.exp((-normalized_data[i * fs, j * fs] + 0.5)*5)))) * zoom
                    print("ij:", normalized_data[i * fs, j * fs])
                    print("size:", size)
                    cv2.circle(img, (int(x), int(y)), int(size), (255, 255, 255), -1)
                    # circle = plt.Circle((x, y), size, color='r', fill=True)  # 圆心坐标为(0.5, 0.5)，半径为0.2
                    # plt.gca().add_patch(circle)
            else:
                #size = normalized_data[i * fs, j * fs] * zoom  # 随机生成点的大小
                size = ((1 / (1 + np.exp((-normalized_data[i * fs, j * fs] + 0.5)*5)))) * zoom
                print("ij:", normalized_data[i * fs, j * fs])
                print("size:", size)
                cv2.circle(img, (int(x), int(y)), int(size), (255, 255, 255), -1)
    # cv2.imshow(dataName, img)
    # cv2.waitKey(0)

    current_time = datetime.datetime.now()
    file_name = dataName + "_fs" + str(fs) + "_" + current_time.strftime("%Y%m%d_%H%M%S") + ".jpg"
    file_name = dataName +"_cv_all"+str(allFlage)+ "_fs" + str(fs) + "_zoom" + str(zoom) + ".jpg"
    #img=cv2.resize(img,(rows,cols))
    cv2.imwrite(file_name,img)



