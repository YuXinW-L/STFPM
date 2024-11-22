import argparse      #获取命令行参数
import os            #文件操作
import cv2           #图像操作
import numpy as np   #数组操作

import torch
import torch.optim
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms

from evaluate import evaluate
from sklearn.model_selection import train_test_split   #用于将数据集随机划分为训练集和测试集
from glob import glob
from PIL import Image


import matplotlib.pyplot as plt
from matplotlib import cm
class MVTecDataset(object):
    '''
    功能：加载和预处理数据
    参数：self.image_list：保存传入的图像路径列表。
          self.transform：保存传入的或默认的图像预处理方法。
          self.dataset：保存加载后的图像数据。
    '''
    def __init__(self, image_list, transform=None):
        self.image_list = image_list
        self.transform = transform
        if self.transform is None:
            self.transform = transforms.ToTensor() 
            #ToTensor() 用于将图像转换为张量格式，并自动将像素值归一化到 [0, 1]
        self.dataset = self.load_dataset()


    '''
    功能：加载 image_list 中的图像路径，将每张图像读入内存并转换为 RGB 格式。
          Image.open(p)：使用 PIL（Pillow）库打开图像文件路径 p。
          .convert('RGB')：将图像统一转换为 RGB 格式，即使图像本身是灰度图或其他模式。
          [Image.open(p).convert('RGB') for p in self.image_list]：对 image_list 中的每个路径执行上述操作
    返回：包含图像对象的列表。   
    '''
    def load_dataset(self):
        return [Image.open(p).convert('RGB') for p in self.image_list]

       
    '''
    功能：实现数据集的 len() 方法，返回数据集中图像的数量。
    逻辑：len(self.dataset)：返回 self.dataset（即加载后的图像列表）的长度。  
    '''
    def __len__(self):
        return len(self.dataset)
    
       
    '''
    功能：  实现数据集的索引操作，通过索引 idx 获取图像数据。
    参数：  idx：索引值，指定需要访问的图像。
    返回值：self.image_list[idx]：返回图像的路径，便于追溯来源。
            image：返回应用了 transform 转换后的图像张量。    
    '''
    def __getitem__(self, idx):
        image = self.transform(self.dataset[idx])
        return self.image_list[idx], image

#定义了一个名为 ResNet18_MS3 的自定义神经网络模块，继承自 PyTorch 的 nn.Module
#它基于 ResNet-18 进行修改，用于提取多尺度特征（MS3 表示 "multi-scale features from 3 layers"）
class ResNet18_MS3(nn.Module): #ResNet18_MS3 是一个自定义的神经网络模块，必须继承 nn.Module，以便利用 PyTorch 的功能

    '''
    功能：初始化模型结构，截取 ResNet-18 的部分网络层。
    步骤：
          1.继承自 nn.Module：ResNet18_MS3 是一个自定义的神经网络模块，必须继承 nn.Module，以便利用 PyTorch 的功能。
          2.加载 ResNet-18：
             models.resnet18(pretrained=pretrained)：
               从 torchvision.models 中加载 ResNet-18。
               如果 pretrained=True，会加载在 ImageNet 上预训练的权重。
               如果 pretrained=False，则使用随机初始化的权重。
          3.删除最后两层：
             list(net.children())[:-2]：
               将 ResNet-18 的网络层转化为一个列表，去掉最后的两层（也就是全连接层 fc 和最后一个池化层 avgpool）。
               ResNet-18 的主要结构是多个卷积块，删除最后两层后保留了特征提取的主要部分。
             torch.nn.Sequential：
               将剩余的网络层重新组合成一个序列模型 self.model。
    '''
    def __init__(self, pretrained=False):
        super(ResNet18_MS3, self).__init__()     #调用了父类（基类）的 __init__ 方法来初始化父类的属性
        net = models.resnet18(pretrained=pretrained)
        # ignore the last block and fc
        self.model = torch.nn.Sequential(*(list(net.children())[:-2]))  #删除最后两层


    '''
    功能：执行前向传播，提取多尺度的特征图。
    步骤：
          1.初始化输出列表：
            res = []：用于存储从特定层提取的特征图。
          2.逐层执行前向传播：
            self.model._modules.items()：
              遍历 self.model（即截取后的 ResNet-18）的所有模块。
              name 是层的名称（字符串），例如 '4'、'5'。
              module 是对应的网络层。
            x = module(x)：
              将输入 x 依次通过每一层。
          3.特定层的特征提取：     
            if name in ['4', '5', '6']：
              当层名称为 '4'（layer1）、'5'（layer2）或 '6'（layer3）时，提取该层的输出特征图。
            res.append(x)：将提取到的特征图添加到列表 res。
          4.返回多尺度特征：         
            return res：
              返回一个包含 3 个特征图的列表，对应 layer1、layer2 和 layer3。
    '''
    def forward(self, x):
        res = []
        for name, module in self.model._modules.items(): #实际上是遍历每一个block
            x = module(x)  #将输入 x 依次通过每一层block
            if name in ['4', '5', '6']:
                res.append(x)
                #print(f"Layer {name} output shape: {x.shape}")
        #res是一个包含三个元素的列表，每一个元素是一个四维tensor (含义)：[batch_size, channels, height, width]
        return res


'''
功能:该函数 load_gt 用于加载某个类别（cls）的ground truth（GT，真实标签）图像数据，并将它们处理成固定大小、可用格式的 numpy 数组
参数：root：数据的根目录路径
      cls：类别的名称，表示需要加载的某个具体类别的 GT 数据      
'''
def load_gt(root, cls):
    gt = []  #初始化 GT 列表，用于存储所有处理后的 GT 图像
    gt_dir = os.path.join(root, cls, 'ground_truth')  #构造 Ground Truth 的文件路径
    sub_dirs = sorted(os.listdir(gt_dir))  #sub_dirs：获取 ground_truth 文件夹下的所有子目录，并按名称排序
    for sb in sub_dirs:  #遍历子目录和子文件；外层循环遍历每个子目录 sb。
        for fname in sorted(os.listdir(os.path.join(gt_dir, sb))):  #内层循环遍历子目录 sb 中的所有文件 fname（按文件名排序）。
            temp = cv2.imread(os.path.join(gt_dir, sb, fname), cv2.IMREAD_GRAYSCALE)
            #使用 OpenCV 以灰度模式读取图像文件 fname，输出的 temp 是一个二维 numpy 数组，形状为 [height, width]。
            temp = cv2.resize(temp, (256, 256)).astype(np.bool)[None, ...]
            #将图像重新调整为 256×256 的大小，便于统一处理。
            #将像素值转为布尔型（True 表示前景，False 表示背景）。
            #使用 [None, ...] 为该数组增加一个新维度，形状变为 [1, 256, 256]，适应后续处理。
            gt.append(temp)  #每处理一个 GT 图像，将其添加到 gt 列表中。
    gt = np.concatenate(gt, 0)  
    #合并所有 GT 图像；将所有 [1, 256, 256] 的 GT 图像按第一维（通道维度）拼接成一个大数组，
    #最终形状为[num_images, 256, 256]，num_images 是所有 GT 图像的总数。

    return gt


def main():
    #定义了一个命令行参数解析器，并设置了一些默认值和参数。
    parser = argparse.ArgumentParser(description="Anomaly Detection")
    #创建一个命令行解析器，description 参数用于为该脚本提供简短的说明：“Anomaly Detection” 表示该脚本用于异常检测任务。
    parser.add_argument("split", nargs="?", choices=["train", "test"])
    #split 参数：可选参数，指定数据集的分割方式，是用于训练（train）还是测试（test）。choices 限定了可选值是“train”或“test”。
    
    # required training super-parameters  必需的训练超参数
    parser.add_argument("--checkpoint", type=str, default=None, help="student checkpoint")
    #checkpoint 参数：指定用于恢复训练的模型检查点路径（文件）。该参数是可选的，默认为 None。
    parser.add_argument("--category", type=str , default='leather', help="category name for MvTec AD dataset")
    #category 参数：指定 MvTec AD 数据集中使用的类别（例如“leather”、“carpet”等）。默认为“leather”。
    parser.add_argument("--epochs", type=int, default=100, help='number of epochs')
    #epochs 参数：指定训练的总周期数，默认为 100。
    
    parser.add_argument("--checkpoint-epoch", type=int, default=100, help="checkpoint resumed for testing (1-based)")
    #checkpoint-epoch 参数：指定测试时需要恢复的检查点（即从第几轮训练中恢复），默认为 100。
    parser.add_argument("--batch-size", type=int, default=32, help='batch size')
    #batch-size 参数：指定训练或测试时的批量大小，默认为 32。
    
    # trivial parameters  次要参数
    parser.add_argument("--result-path", type=str, default='results', help="save results")
    #result-path 参数：指定保存结果（如图像、日志等）的路径，默认为“results”。
    parser.add_argument("--save-fig", action='store_true', help="save images with anomaly score")
    #save-fig 参数：这是一个布尔型的标志，若提供该参数，则会保存带有异常分数的图像。
    parser.add_argument("--mvtec-ad", type=str, default='mvtec_anomaly_detection', help="MvTec-AD dataset path")
    #mvtec-ad 参数：指定 MvTec AD 数据集的路径，默认为“mvtec_anomaly_detection”。
    parser.add_argument('--model-save-path', type=str, default='snapshots', help='path where student models are saved')
    #model-save-path 参数：指定保存训练模型的路径，默认为“snapshots”。

    args = parser.parse_args()  #parse_args：从命令行获取参数，并将其转换为 args 对象，args 是包含所有命令行参数的命名空间对象。通过 args   可以访问传入的各个参数，例如 args.split、args.epochs 等。

    np.random.seed(0)  
    torch.manual_seed(0)
    #设置 NumPy 和 PyTorch 的随机种子为 0，以确保结果的可重复性。即每次运行该脚本时，生成的随机数序列是相同的。
    
    transform = transforms.Compose([     #组合多个数据预处理操作
        transforms.Resize([256, 256]),   #将图像缩放到 256x256 的大小。
        transforms.ToTensor(),           #将图像数据从 PIL.Image 或 numpy.ndarray 转换为 PyTorch 的张量（Tensor）。
                                         #数据会从一个形状为 (H, W, C)的 NumPy 数组转换为一个形状为 (C, H, W) 的 PyTorch Tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  #对RGB图像进行归一化处理，使图像的均值为前者，标准差为后者
    ])

    if args.split == 'train':
        image_list = sorted(glob(os.path.join(args.mvtec_ad, args.category, 'train', 'good', '*.png')))
        #用来加载训练数据集中的所有图像路径。
        train_image_list, val_image_list = train_test_split(image_list, test_size=0.2, random_state=0)
        #这行代码将 image_list 划分为训练集和验证集。
        train_dataset = MVTecDataset(train_image_list, transform=transform)
        #创建一个 train_dataset，它是一个 MVTecDataset 类的实例。
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)
        #这行代码将训练数据集封装成一个 DataLoader，用于批量加载数据。DataLoader 是 PyTorch 中用于加载数据的类，它是 torch.utils.data 模块的一部分，主要用于将数据集批量化、打乱数据顺序、并行加载数据等操作。
        val_dataset = MVTecDataset(val_image_list, transform=transform)
        #这行代码创建一个 val_dataset，它是一个 MVTecDataset 类的实例，负责处理验证集数据
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)
        #这行代码将验证数据集封装成一个 DataLoader，用于批量加载验证数据。
    
    elif args.split == 'test':
        test_neg_image_list = sorted(glob(os.path.join(args.mvtec_ad, args.category, 'test', 'good', '*.png')))
        #使用 glob 查找所有位于 args.mvtec_ad 路径下的、属于指定类别（args.category）的测试集中的“good”文件夹下的 .png 图像。
        test_pos_image_list = set(glob(os.path.join(args.mvtec_ad, args.category, 'test', '*', '*.png'))) - set(test_neg_image_list)
        #这一行的目的是从测试集中的所有图像（包括正常和异常的图像）中去除正常图像（test_neg_image_list）后，得到异常图像列表。
        test_pos_image_list = sorted(list(test_pos_image_list))
        #将 test_pos_image_list 转换为列表（list），然后用 sorted() 对列表进行排序。排序是为了保证图像加载的顺序一致，并可以方便地进行后续处理
        test_neg_dataset = MVTecDataset(test_neg_image_list, transform=transform)
        #这一行创建了一个 MVTecDataset 的实例，用于加载正常图像数据。
        test_pos_dataset = MVTecDataset(test_pos_image_list, transform=transform)
        #同样地，这一行创建了一个 MVTecDataset 的实例，用于加载异常图像数据。
        test_neg_loader = DataLoader(test_neg_dataset, batch_size=1, shuffle=False, drop_last=False)
        #这一行创建了一个 DataLoader 实例，用于加载正常图像数据（test_neg_dataset）。
        test_pos_loader = DataLoader(test_pos_dataset, batch_size=1, shuffle=False, drop_last=False)
        #同样地，这一行创建了一个 DataLoader 实例，用于加载异常图像数据（test_pos_dataset）。

    teacher = ResNet18_MS3(pretrained=True)
    #这一行创建了 ResNet18_MS3 类的一个实例，作为“教师”模型（teacher）。
    #通过 pretrained=True，它会加载一个预训练的 ResNet-18 模型权重，即在 ImageNet 数据集上训练过的权重。预训练权重可以帮助模型更快地收敛，因为它已经学习了通用的特征。
    student = ResNet18_MS3(pretrained=False)
    #这一行创建了另一个 ResNet18_MS3 类的实例，作为“学生”模型（student）。
    #pretrained=False 表示学生模型从随机初始化的权重开始训练，并没有加载预训练的权重。学生模型通常用来通过知识蒸馏（Knowledge Distillation）等方法从教师模型学习。
    
    teacher.cuda()
    student.cuda()
    #这两行代码将 teacher 和 student 模型迁移到 GPU 上，以便在 GPU 上进行计算。cuda() 是 PyTorch 中用于将模型或张量从 CPU 转移到 GPU 的方法。

    if args.split == 'train':
        train_val(teacher, student, train_loader, val_loader, args)
    
    elif args.split == 'test':
        saved_dict = torch.load(args.checkpoint)  #加载模型的保存状态（checkpoint），将其存储在 saved_dict 中。
        category = args.category                  #表示当前要进行测试的分类
        gt = load_gt(args.mvtec_ad, category)     #函数加载 category 对应的 ground truth 数据 gt，通常是用于评估的标签数据

        print('load ' + args.checkpoint)
        student.load_state_dict(saved_dict['state_dict'])  #将保存的模型状态字典加载到 student 模型中
        
        pos = test(teacher, student, test_pos_loader)  #负样本的 loss_map
        neg = test(teacher, student, test_neg_loader)  #正样本的 loss_map

        scores = []
        
        for i in range(len(pos)):
            temp = cv2.resize(pos[i], (256, 256))  #将 pos 和 neg 中的每一张图像通过 cv2.resize 调整到 256x256 的大小。
            scores.append(temp)  #scores 会包含所有样本的分数图（正负样本的分数图合并在一起）
        for i in range(len(neg)):
            temp = cv2.resize(neg[i], (256, 256))
            scores.append(temp)

        scores = np.stack(scores)  #将列表转换为一个 NumPy 数组，形状为 (num_samples, 256, 256)，其中 num_samples 是正负样本的总数。
        #print('------------------Scores shape:', scores.shape)
        
        neg_gt = np.zeros((len(neg), 256, 256), dtype=np.bool)  #负样本的 ground truth，值为 0，并且它的形状为 (len(neg), 256, 256)
        gt_pixel = np.concatenate((gt, neg_gt), 0)  #通过将正样本的 gt 和负样本的 neg_gt 合并在一起，构成像素级别的 ground truth。
        gt_image = np.concatenate((np.ones(pos.shape[0], dtype=np.bool), np.zeros(neg.shape[0], dtype=np.bool)), 0)
        #gt_image 是图像级别的 ground truth，正样本pos(有缺陷的)标记为 1，负样本neg(无缺陷的)标记为 0        
        #print('------------------gt_pixel shape:', gt_pixel.shape)
        
        pro = evaluate(gt_pixel, scores, metric='pro')               #调用evaluate函数计算pro分数
        auc_pixel = evaluate(gt_pixel.flatten(), scores.flatten(), metric='roc')  #调用evaluate函数计算像素级auc分数
        auc_image_max = evaluate(gt_image, scores.max(-1).max(-1), metric='roc')  #调用evaluate函数计算图像集auc分数
        print('Catergory: {:s}\tPixel-AUC: {:.6f}\tImage-AUC: {:.6f}\tPRO: {:.6f}'.format(category, auc_pixel, auc_image_max, pro))  
        
        ######################生成热力图并保存       
        for i, score_map in enumerate(scores):
            # 读取对应的原图（假设原图是 test_pos_loader 或 test_neg_loader 中的图片）
            image_path = test_pos_image_list[i] if i < len(test_pos_image_list) else test_neg_image_list[i - len(test_pos_image_list)]
            image = cv2.imread(image_path)
            # 生成热力图叠加图
            overlay_image = generate_heatmap_with_overlay(image, score_map)
        
            # 确定保存文件名
            if i >= len(test_pos_image_list):
                filename = f"good_heatmap_{i}.png"
            else:
                filename = f"anomaly_heatmap_{i}.png"
            
                # 确定保存路径（包含类别子目录）
            category_dir = os.path.join(args.result_path, args.category)
            save_path = os.path.join(category_dir, filename)
            # 保存或显示叠加后的图像
            if args.save_fig:
                # 确保目录存在
                os.makedirs(category_dir, exist_ok=True)
                cv2.imwrite(save_path, overlay_image)


'''
功能：生成一个loss_map，通过对比教师模型和学生模型的特征输出来衡量它们之间的差异，loss_map可以帮助检测输入图像中的异常区域。
参数：
     teacher：教师模型，已预训练，用于提供给学生模型学习的指导。
     student：训练过后的学生模型。
     loader： 测试样本的DataLoader 实例
返回值：损失映射loss_map
'''  
def test(teacher, student, loader):
    teacher.eval()  #模型切换到评估模式
    student.eval()  #模型切换到评估模式
    loss_map = np.zeros((len(loader.dataset), 64, 64))  #初始化损失映射；每张图像的损失映射大小固定为 (64, 64)，表示异常检测的空间分布
    i = 0
    for batch_data in loader:        #遍历数据加载器
        _, batch_img = batch_data    #数据加载器 loader 按批次返回数据（图像及其标签），这里忽略了标签部分，只保留图像 batch_img
        batch_img = batch_img.cuda() #将 batch_img 转移到 GPU 上进行加速计算。
        with torch.no_grad():        #计算教师模型和学生模型的特征；停止计算梯度，torch.no_grad()：以减少内存占用，提高推理速度
            t_feat = teacher(batch_img)  #t_feat 和 s_feat 是包含多层特征图的列表，
            s_feat = student(batch_img)  #列表中每个元素的形状为 (batch_size, channels, height, width)。
        score_map = 1.
        for j in range(len(t_feat)):
            t_feat[j] = F.normalize(t_feat[j], dim=1)  #F.normalize(t_feat[j], dim=1)：对特征图的每个通道进行
            s_feat[j] = F.normalize(s_feat[j], dim=1)  #L2 归一化，使得特征在比较时具有统一的尺度。
            sm = torch.sum((t_feat[j] - s_feat[j]) ** 2, 1, keepdim=True)  
            #计算教师和学生模型在第 j 层的特征图差异。逐通道计算平方误差，并在通道维度求和，结果的形状是 (batch_size, 1, height, width)。
            sm = F.interpolate(sm, size=(64, 64), mode='bilinear', align_corners=False)  
            #将差异图（score map）通过双线性插值调整到 (64, 64) 的固定大小。
            score_map = score_map * sm  #通过逐层相乘的方式累积每层的损失，最终得到整体的异常评分。
        loss_map[i: i + batch_img.size(0)] = score_map.squeeze().cpu().data.numpy()
        #将当前批次的损失映射 score_map 存入全局的 loss_map，使用 squeeze 去掉多余的维度，转换为 NumPy 格式。
        i += batch_img.size(0)  #通过 i 索引确保每个批次的结果存入 loss_map 的正确位置。
    return loss_map
    

"""
功能：生成热力图并叠加到原图上。
参数：
      image: 原始图像，形状为 [H, W, 3]。
      score_map: 分数图，形状为 [H, W]。
      alpha: 热力图的透明度（0.0 到 1.0），0.0 为完全透明，1.0 为完全不透明。

返回值：叠加后的图像。
"""
def generate_heatmap_with_overlay(image, score_map, alpha=0.5):

    # 归一化 score_map 到 [0, 255] 并转换为 uint8
    score_map_normalized = cv2.normalize(score_map, None, 0, 255, cv2.NORM_MINMAX)
    score_map_uint8 = score_map_normalized.astype('uint8')
    
    #print("~~~~~~~~~~~~~~~~~~~~~~score_map shape:", score_map.shape)
    # 将 score_map 转换为伪彩色热力图
    heatmap = cv2.applyColorMap(score_map_uint8, cv2.COLORMAP_JET)
    image = cv2.resize(image,(256,256))
    #print("~~~~~~~~~~~~~~~~~~~~~~Image shape:", image.shape)       # 显示原图的形状
    #print("~~~~~~~~~~~~~~~~~~~~~~Heatmap shape:", heatmap.shape)   # 显示热力图的形状
    
    # 检查 image 是否为 RGB/BGR 图像
    if len(image.shape) != 3 or image.shape[2] != 3:
        raise ValueError("Input image must be a 3-channel (RGB/BGR) image")

    # 检查 image 和 heatmap 尺寸是否一致，如果不一致则调整大小
    if image.shape[:2] != heatmap.shape[:2]:
        image = cv2.resize(image, (heatmap.shape[1], heatmap.shape[0]))
        print("Resized heatmap to match image shape.")

    # 将热力图叠加到原图上，使用 alpha 透明度
    overlay = cv2.addWeighted(image, 1 - alpha, heatmap, alpha, 0)

    return overlay


'''
功能：训练学生模型并进行验证。

参数：
     teacher：教师模型，通常已预训练，用于提供给学生模型学习的指导。
     student：学生模型，通过学习教师模型的知识来提升其性能。
     train_loader：训练集的 DataLoader，提供训练数据。
     val_loader：验证集的 DataLoader，用于在训练过程中评估学生模型的性能。
     args：包含训练过程的超参数配置（如训练的轮数、学习率等）。

'''
def train_val(teacher, student, train_loader, val_loader, args):
    
    min_err = 10000  #初始化一个变量 min_err，用于记录在验证集上的最小误差。
    teacher.eval()   #将教师模型设置为评估模式，禁用诸如 dropout 和 batch normalization 这类只在训练时使用的层。
    student.train()  #将学生模型设置为训练模式，启用训练时特有的操作（如 dropout）

    optimizer = torch.optim.SGD(student.parameters(), 0.4, momentum=0.9, weight_decay=1e-4)
    #使用SGD优化器来更新学生模型的权重，设置了学习率为 0.4，动量为 0.9，权重衰减为 1e-4，这有助于防止过拟合。
    for epoch in range(args.epochs):  #开始训练过程，循环进行 args.epochs 次迭代（即训练的轮数）
        student.train()
        for batch_data in train_loader:  
        #train_loader 会逐批次地提供训练数据，每次迭代中，train_loader的shape：(batch_size, channels, height, width)
        #batch_data 是从 train_loader 中获取的每一个批次的数据，batch_data 通常是一个元组（labels，图像数据）
            _, batch_img = batch_data    #忽略第一个元素，从 batch_data 中提取batch_img（即当前批次的图像），shape：(32, 3, 256, 256)
            batch_img = batch_img.cuda() #并将其移动到 GPU 上进行计算（batch_img.cuda()）

            with torch.no_grad():  #使用 torch.no_grad() 禁用梯度计算，因为在这里我们不需要对教师模型进行反向传播，仅仅是获取教师模型的特征。
                t_feat = teacher(batch_img)  #t_feat = teacher(batch_img)：将当前批次的图像输入教师模型，得到教师模型的特征输出。
            s_feat = student(batch_img)  #将当前批次的图像输入学生模型，得到学生模型的特征输出

            loss =  0  #初始化 loss 为 0，用于累积每个特征层之间的损失。
            for i in range(len(t_feat)):
                t_feat[i] = F.normalize(t_feat[i], dim=1)  #对于教师和学生模型的每一层特征t_feat[i] 和 s_feat[i]，先进行 L2 归一化F.normalize()
                s_feat[i] = F.normalize(s_feat[i], dim=1)  #这样可以将特征值缩放到相同的尺度，减少尺度不一致的影响。
                loss += torch.sum((t_feat[i] - s_feat[i]) ** 2, 1).mean()  #计算教师和学生特征之间的均方误差（MSE）损失：torch.sum((t_feat[i] - s_feat[i]) ** 2, 1).mean()，表示在每一维特征上，计算教师和学生特征之间的差异，并求平均值。并将所有层的损失累加到 loss 变量中

            print('[%d/%d] loss: %f' % (epoch, args.epochs, loss.item()))  #打印当前 epoch 的损失值，监控训练过程。
            optimizer.zero_grad()  #清空之前计算的梯度。
            loss.backward()        #计算损失的梯度。
            optimizer.step()       #根据计算得到的梯度更新学生模型的参数。
        
        err = test(teacher, student, val_loader).mean()  #调用 test 函数，计算学生模型在验证集上的误差（err）。
        print('Valid Loss: {:.7f}'.format(err.item()))   #打印当前验证集上的损失。
        if err < min_err:       #如果当前验证集的误差比之前的最小误差 min_err 小，说明学生模型在验证集上的表现有所改进。
            min_err = err       #更新 min_err，并为模型保存路径 save_name 生成文件夹（如果文件夹不存在）。
            save_name = os.path.join(args.model_save_path, args.category, 'best.pth.tar')
            dir_name = os.path.dirname(save_name)
            if dir_name and not os.path.exists(dir_name):
                os.makedirs(dir_name)
            state_dict = {    #将学生模型的状态字典（即模型的所有参数）保存到文件 save_name 中。state_dict 也包含了当前的类别名称 args.category。
                'category': args.category,
                'state_dict': student.state_dict()
            }
            torch.save(state_dict, save_name)

if __name__ == "__main__":
    main()
