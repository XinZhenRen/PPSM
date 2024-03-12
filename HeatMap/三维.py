import cv2
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
# 读取图像
image = cv2.imread('cju0qkwl35piu0993l0dewei2.jpg')

# 将图像转换为灰度图像
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# 获取灰度图像的高度和宽度
height, width = gray_image.shape

# 创建X和Y坐标网格
X, Y = np.meshgrid(np.arange(width), np.arange(height))

# 使用灰度图像的像素强度作为Z轴的值
Z = gray_image
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
