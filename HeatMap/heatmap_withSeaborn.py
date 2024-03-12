import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

def heatmap_with_custom_palette(image_path, pixel_size, palette_name):
    # 打开处理后的图片
    #img = Image.open("pixelated_" + str(pixel_size) + "_" + image_path)
    img = Image.open(image_path)

    # 将图像转为NumPy数组
    img_array = np.array(img)

    # 创建热力图
    plt.figure(figsize=(10, 8))
    sns.heatmap(img_array[:, :, 0], cmap=palette_name, cbar=False)

    # 设置图像标题和坐标轴标签
    plt.title(f"Heatmap with {palette_name} Palette")
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")

    # 显示热力图
    plt.show()

def heatmap_from_image(image_path, pixel_size):
    # 打开处理后的图片
    #img = Image.open("pixelated_" + str(pixel_size) + "_" + image_path)
    img = Image.open(image_path)

    # 将图像转为NumPy数组
    img_array = np.array(img)

    # 创建热力图
    plt.figure(figsize=(10, 8))
    sns.heatmap(img_array[:, :, 0], cmap="viridis", cbar=False)

    # 设置图像标题和坐标轴标签
    plt.title("Heatmap from Pixelated Image")
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")

    # 显示热力图
    plt.show()

# 使用例子
heatmap_from_image("cju0qkwl35piu0993l0dewei2.jpg", 10)

# 使用例子
heatmap_with_custom_palette("cju0qkwl35piu0993l0dewei2.jpg", 10, "coolwarm")
