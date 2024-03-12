% cdata = [1 2 3 4 5; 5 4 3 2 1; 1 2 3 4 5; 5 4 3 2 1; 1 2 3 4 5];
% xvalues = {'1x', '2x', '3x', '4x', '5x'};
% yvalues = {'1y', '2y', '3y', '4y', '5y'};
% h = heatmap(xvalues, yvalues, cdata);
% clc;clear;close all;




% 定义点(x,y,z)
x			= randn(50,1);
xmax 	= max(x);
xmin 	= min(x);
y			= randn(50,1);
ymax 	= max(y);
ymin 	= min(y);
z = exp(sin(x.^2)) + exp(cos(y.^2));
N = 500; % 每个维度的数据点数
% 网格化x,y二维空间
[X,Y] = meshgrid(linspace(xmin,xmax,N),linspace(ymin,ymax,N));
% 采用插值法扩展数据，可用方法有'linear'(default)|'nearest'|'natural'|'cubic'|'v4'|
Z = griddata(x,y,z,X,Y,'v4');

%% 等高线法
figure('NumberTitle','off','Name','等高线法','Color','w','MenuBar','none','ToolBar','none');
contourf(X,Y,Z,N, 'LineColor','none');
colormap('jet');
colorbar;
axis off;

%% 投影图法
figure('NumberTitle','off','Name','投影图法','Color','w','MenuBar','none','ToolBar','none');
surf(X,Y,Z,'LineStyle','none');
xlim([min(X(:)) max(X(:))]);
ylim([min(Y(:)) max(Y(:))]);
axis off;
colormap('jet');
colorbar;
shading interp;
view(0,90);

%% imagesc法
figure('NumberTitle','off','Name','imagesc法','Color','w','MenuBar','none','ToolBar','none');
% 因为图像坐标和笛卡尔坐标起始位置不一样，需要上下翻转
imagesc(flipud(Z));
colormap('jet');
colorbar;
axis off;

%% pcolor法
figure('NumberTitle','off','Name','pcolor法','Color','w','MenuBar','none','ToolBar','none');
pcolor(X,Y,Z);
colormap('jet');
colorbar;
shading interp;
axis off;














% % 指定图像文件夹的路径
% imageFolder = 'D:\SHU\PyTorch\polyp segmentation dataset\kvasir-seg\Kvasir-SEG\masks\';
% 
% % 获取文件夹中的所有图像文件
% imageFiles = dir(fullfile(imageFolder, '*.jpg')); % 可以根据实际情况更改文件类型
% 
% % 初始化一个空的中心点坐标矩阵
% centerPoints = [];
% 
% % 循环处理每个图像文件
% for i = 1:length(imageFiles)
%     % 读取图像
%     img = imread(fullfile(imageFolder, imageFiles(i).name));
%     img
%     % 将图像转为灰度图像
%     grayImg = rgb2gray(img);
% 
%     % 自适应阈值化二值化
%     threshold = graythresh(grayImg);
%     binaryImg = im2bw(grayImg, threshold);
% 
%     % 使用regionprops函数查找二值化图像中的中心点
%     stats = regionprops(binaryImg, 'Centroid');
%     center = stats.Centroid;
% 
%     % 将中心点的坐标添加到centerPoints矩阵中
%     centerPoints = [centerPoints; center];
% end
% centerPoints
% % 生成热力图
% heatmap = zeros(size(binaryImg));
% 
% % xmax 	= max(x);
% % xmin 	= min(x);
% % % 网格化x,y二维空间
% % [X,Y] = meshgrid(linspace(xmin,xmax,N),linspace(ymin,ymax,N));
% % % 采用插值法扩展数据，可用方法有'linear'(default)|'nearest'|'natural'|'cubic'|'v4'|
% % Z = griddata(x,y,z,X,Y,'v4');
% 
% % 将中心点的位置添加到热力图中
% for i = 1:size(centerPoints, 1)
%     x = round(centerPoints(i, 1));
%     y = round(centerPoints(i, 2));
%     if x > 0 && x <= size(heatmap, 2) && y > 0 && y <= size(heatmap, 1)
%         heatmap(y, x) = heatmap(y, x) + 1;
%     end
% end
% 
% % 显示热力图
% imagesc(heatmap);
% colormap('hot');
% colorbar;
% title('热力图');