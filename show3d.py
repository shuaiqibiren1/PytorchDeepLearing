import nibabel as nib
import myvi
import numpy as np
from dataprocess.utils import file_name_path
import scipy.ndimage as ndimg

# 加载 NIfTI 文件
# nii = nib.load("dataset\Mask\mr_train_1002.nii")
# maskpathname = "dataset\Mask\mr_train_1002.nii"
# 需要添加一部分来导入uploads中的数据
datapath = "data/uploads/Image"
maskpath = "data/uploads/Mask"
image_path_list = file_name_path(datapath, False, True)
maskpathname = maskpath + "/" + "result.nii"
nii = nib.load(maskpathname)

# nii = nib.load('myvi/data/patient2_LGE_manual.nii')
imgs = nii.get_fdata()

# 获取唯一标签值
unique_labels = np.unique(imgs)

# 打印唯一标签值
print("Unique labels in imgs:", unique_labels)

# 映射标签值到连续整数范围
mapped_labels = np.linspace(0, len(unique_labels) - 1, len(unique_labels)).astype(int)

# 获取图像空间信息
zoom = nii.header.get_zooms()

# 设置平滑参数
sigma = 1

# 创建 Manager 对象
manager = myvi.Manager()

# 预定义颜色数组
colors = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (1, 0, 1), (0, 1, 1), (0.5, 0.5, 0.5)]

# 遍历唯一标签值，为每个标签值创建相应的器官表面
for label, mapped_label in zip(unique_labels, mapped_labels):
    # 使用 Gaussian 滤波平滑数据
    organ = ndimg.gaussian_filter(np.float32(imgs == label), sigma)
    
    # 构建器官表面
    vts, fs, ns, vs = myvi.util.build_surf3d(organ, 1, 0.5, zoom)
    
    # 获取对应的颜色
    color = colors[mapped_label % len(colors)]  # 使用模运算确保颜色索引不超出范围
    
    # 将器官添加到 Manager 对象中
    manager.add_surf(str(int(label)), vts, fs, ns, color)  # 将标签转换为整数作为名称，并使用独热编码的颜色
    
# 显示器官 3D 模型
manager.show('Organ 3D Demo')

