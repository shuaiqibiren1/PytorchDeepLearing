# 图像分割模型
> 有一些3D图像分割和2D图像分割的模型网络

## 图像分割网络
> 有2D和3D版本的Vnet

# 图像分割损失函数
> 有一些3D图像分割和2D图像分割的损失函数

## 如何使用
我们使用了pytorch1.10.0重新实现了图像分割损失函数

有二元交叉熵、dice损失、focal损失等，都有2D和3D版本。

有分类交叉熵损失、dice损失、focal损失等，都有2D和3D版本。

用于计算图像相似性的MS-SSIM损失和SSIM损失。

用于血管分割的中心线dice损失。

有9种分割度量，包括dice、表面距离、jaccard、VOE、RVD、FNR、FPR、ASSD、RMSD、MSD等。

flask_app.py是Flask深度学习分割模型服务部署的demo实例
