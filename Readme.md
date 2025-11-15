Detecting Linear Deformation Features in Arctic Landfast Ice Using Radar Interferometry简介 (Introduction)本项目旨在利用雷达干涉测量 (InSAR) 技术，结合先进的深度学习语义分割模型，对北极固定冰 (Arctic Landfast Ice) 中的线性形变特征（如裂缝、剪切带等）进行自动、精确的提取和识别。本仓库包含了九种用于极地冰情分析的深度学习模型实现，并详细记录了它们的配置和实验性能。模型架构 (Model Architectures)本研究系统地比较了三种主流语义分割网络架构（U-Net, DeepLabV3, PSPNet）及其引入了通道注意力（Channel Attention, CA）和空间注意力（Spatial Attention, SA）机制的变体。所有模型均在 PyTorch 或兼容框架下实现，对应文件如下：文件名 (Filename)对应模型 (Model Name)核心架构 (Core Architecture)注意力机制 (Attention Mechanism)Unet.pyU-Net编码器-解码器结构无 (None)Unet-ChannalAttention.pyU-Net-CAU-Net通道注意力 (Channel Attention)Unet-SpatialAttention.pyU-Net-SAU-Net空间注意力 (Spatial Attention)DeepLabV3.pyDeepLabV3带有空洞空间金字塔池化 (ASPP)无 (None)DeepLabV3ChannelAttention.pyDeepLabV3-CADeepLabV3通道注意力 (Channel Attention)DeepLabV3SpatialAttention.pyDeepLabV3-SADeepLabV3空间注意力 (Spatial Attention)PSPNet.pyPSPNet带有金字塔池化模块 (PPM)无 (None)PSPNet-ChannelAttention.pyPSPNet-CAPSPNet通道注意力 (Channel Attention)PSPNet-SpatialAttention.pyPSPNet-SAPSPNet空间注意力 (Spatial Attention)注意力机制说明 (Attention Mechanism)通道注意力 (CA): 专注于特征图的不同通道之间的关系，允许模型对重要通道进行加权，从而增强有意义的特征表示。空间注意力 (SA): 专注于特征图的不同空间位置之间的关系，帮助模型识别图像中对分割任务最重要的区域。实验设置 (Experimental Configurations)以下表格总结了所有模型的训练配置参数：架构 (Architecture)注意力机制 (Attention Mechanism)主干网络 (Backbone)输入尺寸 (Input Size)批量大小 (Batch Size)初始学习率 (Initial Learning Rate)U-NetNoneNone64128-U-Net-CAChannel AttentionNone64128-U-Net-SASpatial AttentionNone64128-DeepLabV3NoneResNet-5064128-DeepLabV3-CAChannel AttentionResNet-5064128-DeepLabV3-SASpatial AttentionResNet-5064128-PSPNetNoneResNet-5064128-PSPNet-CAChannel AttentionResNet-5064128-PSPNet-SASpatial AttentionResNet-5064128-注: 初始学习率 (Initial Learning Rate) 参数在原始数据中缺失，请根据你的实验设置补充。实验结果 (Experimental Results)模型在测试集上的性能指标如下所示。性能指标越高，表示模型对线性形变特征的分割能力越强。模型 (Model)mIoU (%)mPA (%)F1-Score (%)OA (%)U-Net74.0379.1882.7896.44U-Net-CA76.0980.9383.6096.88U-Net-SA75.1579.2683.7296.73DeepLabV367.4072.0075.1895.72DeepLabV3-CA66.9070.9275.8795.52DeepLabV3-SA67.4872.4276.4995.42PSPNet66.5170.4275.4695.49PSPNet-CA66.5470.9275.5095.39PSPNet-SA65.7269.6574.6195.35指标说明:mIoU (Mean Intersection over Union): 平均交并比，是衡量语义分割精度的核心指标。mPA (Mean Pixel Accuracy): 平均像素精度。F1-Score: 分割的平衡指标。OA (Overall Accuracy): 整体像素精度。从结果可以看出，在本项目任务中，引入通道注意力机制的 U-Net-CA 取得了最高的 mIoU 和 mPA 表现。如何使用 (Usage)请按照以下步骤运行和测试模型：环境配置:# 示例：安装必要的依赖
pip install torch torchvision numpy opencv-python
数据准备:下载InSAR形变图和对应的标签数据。将数据组织成以下结构：data/
├── images/
│   ├── 001.png
│   ├── 002.png
│   └── ...
└── labels/
    ├── 001.png
    ├── 002.png
    └── ...
运行训练:# 以 U-Net-CA 为例
python Unet-ChannalAttention.py --config_path config.yaml
引用 (Citation)如果您的研究使用了本仓库的代码或方法，请引用我们的文章：@article{YourName2023Detecting,
  title={Detecting Linear Deformation Features in Arctic Landfast Ice Using Radar Interferometry},
  author={Your Name and Co-Author Name},
  journal={Journal Name},
  year={Year},
  volume={Volume},
  number={Number},
  pages={Pages}
}
