import numpy as np
import matplotlib.pyplot as plt
import progressbar
import datetime

fillName=[
    # "kvasir-sessile20231126_160357.txt",
    #       "SUN20231126_154955.txt",
    #       "Kvasir-SEG20231126_172709.txt",
    #       "CVC-ClinicDB20231126_161519.txt",
    #       "IVPS-TrainSet20231126_133434.txt",
    #       "hyper-kvasir20231126_115149.txt",
          "polypGen20240311_204043.txt"]
dName=[
#     "kvasir-sessile",
# "SUN",
# "Kvasir-SEG",
# "CVC-ClinicDB",
# "MICCVI",
# "Hyper-Kvasir",
       "polypGen"]
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
    data_array=np.loadtxt(data_array)
    # dataName="kvasir-sessile"
    # dataName="SUN"
    # dataName="Kvasir-SEG"
    # dataName="CVC-ClinicDB"
    # dataName="IVPS"
    # dataName="Hyper-Kvasir"
    # 设置采样 5代表没各个像素采样一次
    fs = 40
    zoom = 40
    # data_array=np.random.rand(400, 400)
    print(data_array)
    min_values = data_array.min(axis=0)
    max_values = data_array.max(axis=0)

    # 避免分母为零，将最小值和最大值相等的情况处理为1
    max_values[max_values == min_values] = 1
    normalized_data = (data_array - min_values) / (max_values - min_values)

    print(normalized_data)

    # 提取数组形状
    rows, cols = normalized_data.shape
    print("rows:", rows)
    print("cols:", cols)

    # 设置图像大小
    plt.figure(figsize=(cols / 100, rows / 100), dpi=1000)
    plt.axes().set_facecolor("black")
    size_inches = plt.gcf().get_size_inches()
    print("图形大小（英寸）：", size_inches)
    # 遍历数组，在每个位置生成随机数，决定是否在该位置绘制点
    for i in bar(range(int(rows / fs))):
        for j in range(int(cols / fs)):
            x = j / cols
            y = i / rows
            x = j * fs
            y = i * fs

            # 生成0到1的随机数，如果小于概率密度函数的值，就在该位置绘制点
            if np.random.rand() < normalized_data[i * fs, j * fs]:
                size = normalized_data[i * fs, j * fs] * zoom  # 随机生成点的大小
                plt.scatter(x, y, s=size, color='white', alpha=1)
                # circle = plt.Circle((x, y), size, color='r', fill=True)  # 圆心坐标为(0.5, 0.5)，半径为0.2
                # plt.gca().add_patch(circle)

            # size = normalized_data[i * fs, j * fs] * zoom  # 随机生成点的大小
            # plt.scatter(x, y, s=size, color='white', alpha=1)


    # plt.title('Circle Dots based on Probability Density Function')
    # plt.xlabel('X-axis')
    # plt.ylabel('Y-axis')

    current_time = datetime.datetime.now()
    file_name = dataName + "_fs" + str(fs) + "_" + current_time.strftime("%Y%m%d_%H%M%S") + ".jpg"
    file_name = dataName + "_fs" + str(fs) + "_zoom" + str(zoom) + ".jpg"
    # plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    # plt.margins(0, 0)
    plt.xticks([])  # 去 x 轴刻度
    plt.yticks([])  # 去 y 轴刻度
    plt.savefig(file_name, bbox_inches='tight', pad_inches=0)  # 保存为 PNG 格式
    # plt.show()


