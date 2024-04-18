from flask import Flask, request, send_file, render_template
from model import *
import os
import torch
import SimpleITK as sitk
import subprocess

# Use CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
use_cuda = torch.cuda.is_available()

app = Flask(__name__)

# 载入模型
newSize = (112, 112, 128)
Unet3d = MutilUNet3dModel(image_depth=128, image_height=112, image_width=112, image_channel=1, numclass=8,
                          batch_size=1, loss_name='MutilFocalLoss', inference=True,
                          model_path='log/MutilUNet3d/dice/MutilUNet3d.pth')

root_dir = r"data/uploads/Image"
if not os.path.exists(root_dir):
    os.makedirs(root_dir)

root_Mask_dir = r"data/uploads/Mask"
if not os.path.exists(root_Mask_dir):
    os.makedirs(root_Mask_dir)


# 定义根路由，渲染 HTML 页面
@app.route('/')
def index():
    return render_template('index.html')

# 定义服务接口
@app.route('/process', methods=['POST'])
def process():
    file = request.files.get('file')  # 获取上传的文件
    if file:
        # 保存上传的文件
        file.save(root_dir + '/' + file.filename)
        
        # 读取上传的文件
        sitk_image = sitk.ReadImage(root_dir + '/' + file.filename)
        
        # 执行推理
        sitk_mask = Unet3d.inference(sitk_image, newSize)
        
        # 保存推理结果
        sitk.WriteImage(sitk_mask, root_Mask_dir + '/' + "result.nii")
        
        return 'Segmentation Success!'
    else:
        return 'No file uploaded'

# 定义下载结果文件接口
@app.route('/download', methods=['GET'])
def download():
    filename = request.args.get('file')  # 获取请求参数中的文件名
    # print("The filename is:", filename)  # 打印文件名
    if not filename:
        print("Missing parameter: file")
        return "Missing parameter: file"  # 没有提供文件名
    filepath = os.path.join(root_Mask_dir + '/' + filename)  # 生成完整的文件路径
    filepath = "C:\\Users\\admin\\Desktop\\challenge\\PytorchDeepLearing-main\\data\\uploads\\Mask" + '\\' + filename
    print("filepath : ", filepath)
    
    try:
        return send_file(filepath) #进行了修改
    except FileNotFoundError:
        return "The file does not exist"  # 文件不存在

# 定义可视化接口
@app.route('/visualization', methods=['GET'])
def visualization():
    # 调用 show3d.py 进行可视化
    subprocess.run(['python', 'show3d.py'])
    return 'Visualization Success!'

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
