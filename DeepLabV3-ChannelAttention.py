import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.transforms.functional as F_T 
import torchvision.models.segmentation as segmentation 
import numpy as np
from PIL import Image
import os
import time
import json 
from typing import Tuple, List, Dict, Any
# 导入数学工具，用于指标计算
import numpy as np

# --- 配置参数 (Configuration Parameters) ---

# TODO: 【重要】请将此路径修改为您运行数据预处理脚本后生成的 'VOCdevkit/VOC2012' 目录的实际位置。
VOC_ROOT = "../testsize647500" 

# 输入图像的尺寸
IMAGE_SIZE = 64
# 分割类别数 (背景 0 + 海冰 1)
NUM_CLASSES = 2 
# 训练参数
BATCH_SIZE = 128
NUM_EPOCHS = 25
LEARNING_RATE = 1e-4

# TODO: 【新增】指定使用的 GPU 索引。如果您有多张 GPU (如 0, 1, 2...)，请指定其中一个。
# 如果设置为 None 或您只有 CPU，代码会回退到自动检测。
GPU_ID = 0 

# TODO: 【重要】模型保存路径。
MODEL_SAVE_PATH = "trained_models/deeplabv3_sea_ice_segmentation_64x64_cam_best.pth" 
# 【新增】指标保存路径
METRICS_SAVE_PATH = "training_metrics/training_history_deeplabv3_64x64_cam.json" 

# 用于检查 GPU
if GPU_ID is not None and torch.cuda.is_available():
    DEVICE = torch.device(f"cuda:{GPU_ID}")
else:
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# --- 1. 通道注意力模块定义 (Channel Attention Module, CAM) ---

class ChannelAttentionModule(nn.Module):
    """
    通道注意力模块 (CAM)。
    通过沿着空间轴的平均池化和最大池化，然后通过一个共享 MLP 生成通道注意力向量。
    """
    def __init__(self, in_channels: int, reduction_ratio: int = 16):
        super(ChannelAttentionModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        # 共享 MLP，使用 1x1 卷积实现
        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction_ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. 沿着空间轴进行平均池化和最大池化
        avg_out = self.avg_pool(x)
        max_out = self.max_pool(x)
        
        # 2. 将池化结果分别送入共享 MLP 并相加
        out = self.mlp(avg_out) + self.mlp(max_out)
        
        # 3. Sigmoid 激活得到注意力向量
        attention_vector = self.sigmoid(out)
        
        # 4. 将注意力向量乘回原特征图 (通道加权)
        return x * attention_vector
    
# --- 2. 带注意力机制的 DeepLabV3 模型定义 (DeepLabV3 Model Definition with Attention) ---

class DeepLabV3_SingleChannel_Attn(nn.Module):
    """
    封装 DeepLabV3 模型，支持单通道输入，并在 ASPP 输出后加入通道注意力模块 (CAM)。
    """
    def __init__(self, num_classes: int = 2, backbone: str = 'resnet50', pretrained: bool = True):
        super(DeepLabV3_SingleChannel_Attn, self).__init__()
        
        # 1. 加载 DeepLabV3 模型
        if backbone == 'resnet50':
            self.model = segmentation.deeplabv3_resnet50(pretrained=pretrained, progress=True)
        elif backbone == 'resnet101':
            self.model = segmentation.deeplabv3_resnet101(pretrained=pretrained, progress=True)
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        # DeepLabV3 ASPP 后的特征通道数
        final_conv_in_channels = 256
        
        # 2. 修改输出层以匹配类别数
        self.model.classifier[4] = nn.Conv2d(final_conv_in_channels, num_classes, kernel_size=(1, 1), stride=(1, 1))

        # 3. 适配单通道输入：修改 ResNet 的第一层卷积
        original_conv1 = self.model.backbone.conv1 
        new_conv1 = nn.Conv2d(
            1, original_conv1.out_channels, kernel_size=original_conv1.kernel_size,
            stride=original_conv1.stride, padding=original_conv1.padding,
            bias=original_conv1.bias is not None
        )
        if pretrained:
            with torch.no_grad():
                # 使用 RGB 三个通道的平均权重作为新的单通道权重
                mean_weights = original_conv1.weight.mean(dim=1, keepdim=True)
                new_conv1.weight.data = mean_weights
                if original_conv1.bias is not None:
                    new_conv1.bias.data = original_conv1.bias.data
        self.model.backbone.conv1 = new_conv1
        
        # 4. 实例化通道注意力模块 (CAM)
        self.attention_module = ChannelAttentionModule(in_channels=final_conv_in_channels, reduction_ratio=16)

        # 5. 【修正】拆分 DeepLabV3 的 forward 流程：使用 nn.Sequential 显式重组子模块，避免切片错误
        self.backbone = self.model.backbone
        # ASPP (classifier[0])
        self.aspp = self.model.classifier[0] 
        
        # ASPP 后处理部分：1x1 Conv + BN + ReLU (classifier[1] 到 classifier[3])
        # 使用 nn.Sequential 手动组合，避免 DeepLabHead 的 __init__ 错误
        self.post_aspp_conv = nn.Sequential(
            self.model.classifier[1], # 1x1 Conv
            self.model.classifier[2], # BatchNorm
            self.model.classifier[3]  # ReLU
        ) 
        
        # 最终的 1x1 卷积 (classifier[4])
        self.upsample_conv = self.model.classifier[4] 


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_shape = x.shape[-2:]
        
        # 1. Backbone (ResNet) 特征提取
        features = self.backbone(x)
        x = features['out']

        # 2. ASPP 模块
        x = self.aspp(x)
        
        # 3. ASPP 后处理卷积 (通道数仍是 256)
        x = self.post_aspp_conv(x)
        
        # 4. 应用通道注意力
        x = self.attention_module(x) 

        # 5. 最终分类卷积
        x = self.upsample_conv(x)

        # 6. 上采样到原始输入尺寸
        x = F_T.resize(x, input_shape, interpolation=T.InterpolationMode.BILINEAR)

        return x

# --- 3. 数据集加载 (Dataset Loader) ---

class VOCSegDataset(Dataset):
    """
    加载由预处理脚本生成的 VOC 格式海冰分割数据集。
    """
    def __init__(self, voc_root: str, image_size: int, image_set: str = 'train', transforms=None):
        self.voc_root = voc_root
        self.image_size = image_size
        self.transforms = transforms
        
        self.image_dir = os.path.join(voc_root, 'JPEGImages')
        self.mask_dir = os.path.join(voc_root, 'SegmentationClass')
        self.image_set_path = os.path.join(voc_root, 'ImageSets', 'Segmentation', f'{image_set}.txt')

        if not os.path.exists(self.image_set_path):
            raise FileNotFoundError(f"ImageSets 文件未找到。请确认文件 {image_set}.txt 存在。Path: {self.image_set_path}")

        with open(self.image_set_path, 'r') as f:
            self.ids = [line.strip() for line in f.readlines()]
            
        print(f"成功加载 {image_set} 集合的 {len(self.ids)} 个切片数据。")

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_id = self.ids[idx]
        
        img_path = os.path.join(self.image_dir, f"{img_id}.jpg")
        # 读取为灰度图 (L)
        img = Image.open(img_path).convert('L') 
        
        mask_path = os.path.join(self.mask_dir, f"{img_id}.png")
        # 读取掩膜为灰度图 (L)
        mask = Image.open(mask_path).convert('L') 
        
        if self.transforms is not None:
            img = self.transforms(img)

        # 掩膜 Resize 使用最近邻插值
        mask = F_T.resize(mask, 
                              (self.image_size, self.image_size), 
                              interpolation=T.InterpolationMode.NEAREST)
        
        mask = T.ToTensor()(mask)
        # 转换为长整型，并移除通道维度 (C=1)
        mask = mask.squeeze(0).long() 

        return img, mask

# --- 4. 性能指标计算辅助函数 (Metrics Computation) ---

def compute_metrics(outputs: torch.Tensor, masks: torch.Tensor, num_classes: int) -> Dict[str, float]:
    """计算 PA/OA, mIoU, mPA 和 mF1-Score。"""
    _, preds = torch.max(outputs, 1)
    
    # 忽略索引 255 的像素
    valid_mask = (masks != 255)
    
    preds_flat = preds[valid_mask]
    masks_flat = masks[valid_mask]
    
    # 将 Tensor 转移到 CPU 并转为 NumPy 
    preds_flat = preds_flat.cpu().numpy()
    masks_flat = masks_flat.cpu().numpy()
    
    # 初始化统计量
    TP = np.zeros(num_classes) # True Positives
    FP = np.zeros(num_classes) # False Positives
    FN = np.zeros(num_classes) # False Negatives

    for cls in range(num_classes):
        # True Positives: 真实为 cls 且预测为 cls
        TP[cls] = ((masks_flat == cls) & (preds_flat == cls)).sum()
        # False Positives: 真实非 cls 且预测为 cls
        FP[cls] = ((masks_flat != cls) & (preds_flat == cls)).sum()
        # False Negatives: 真实为 cls 且预测非 cls
        FN[cls] = ((masks_flat == cls) & (preds_flat != cls)).sum()

    # --- 1. Overall Accuracy (OA) / Pixel Accuracy (PA) ---
    total_correct_pixels = TP.sum() 
    total_pixels = TP.sum() + FP.sum() + FN.sum() 
    
    acc = total_correct_pixels / total_pixels if total_pixels > 0 else 0.0

    # --- 2. Mean IoU (mIoU) ---
    union = TP + FP + FN
    # 计算 IoU 时，避免除以零。对于 union=0 的类别，IoU 记为 0
    iou = np.divide(TP, union, out=np.zeros_like(TP, dtype=float), where=union!=0)
    # 只对在数据中出现过的类别 (union > 0) 求平均
    miou = np.mean(iou[union > 0]) if np.any(union > 0) else 0.0

    # --- 3. Mean Pixel Accuracy (mPA) ---
    # 类别准确率 (Recall) = TP / (TP + FN)
    recall = np.divide(TP, (TP + FN), out=np.zeros_like(TP, dtype=float), where=(TP + FN)!=0)
    # 只对在真实标签中出现过的类别 (TP + FN > 0) 求平均
    mpa = np.mean(recall[(TP + FN) > 0]) if np.any((TP + FN) > 0) else 0.0

    # --- 4. Mean F1-Score (mF1) ---
    # Precision = TP / (TP + FP)
    precision = np.divide(TP, (TP + FP), out=np.zeros_like(TP, dtype=float), where=(TP + FP)!=0)
    
    # F1 = 2 * (Precision * Recall) / (Precision + Recall)
    f1_scores = np.divide(2 * precision * recall, (precision + recall), 
                          out=np.zeros_like(TP, dtype=float), 
                          where=(precision + recall)!=0)
    
    # Mean F1-Score (mF1)
    mf1 = np.mean(f1_scores[(TP + FN) > 0]) if np.any((TP + FN) > 0) else 0.0 # 同样只对出现过的类别求平均
    
    return {
        'acc': acc, # PA/OA
        'miou': miou,
        'mpa': mpa,
        'mf1': mf1
    }

# --- 5. 验证辅助函数 (Validation Function) ---

@torch.no_grad()
def validate_model(model: DeepLabV3_SingleChannel_Attn, dataloader: DataLoader, criterion: nn.Module) -> Dict[str, float]:
    """在验证集上评估模型的性能，返回指标字典"""
    model.eval()
    
    running_loss = 0.0
    
    total_metrics = {'acc': 0.0, 'miou': 0.0, 'mpa': 0.0, 'mf1': 0.0}
    num_samples = 0

    for images, masks in dataloader:
        images = images.to(DEVICE)
        masks = masks.to(DEVICE)
        
        outputs = model(images)
        loss = criterion(outputs, masks)
        
        running_loss += loss.item() * images.size(0)
        
        metrics = compute_metrics(outputs.detach(), masks.detach(), NUM_CLASSES)
        
        for key in total_metrics:
            total_metrics[key] += metrics[key] * images.size(0)
            
        num_samples += images.size(0)

    if num_samples > 0:
        val_metrics = {
            'val_loss': running_loss / num_samples,
            'val_acc': total_metrics['acc'] / num_samples,
            'val_miou': total_metrics['miou'] / num_samples,
            'val_mpa': total_metrics['mpa'] / num_samples,
            'val_mf1': total_metrics['mf1'] / num_samples,
        }
    else:
        val_metrics = {'val_loss': 0.0, 'val_acc': 0.0, 'val_miou': 0.0, 'val_mpa': 0.0, 'val_mf1': 0.0}

    print(f"\n--- 验证集评估结果 ---")
    print(f"验证集平均 Loss: {val_metrics['val_loss']:.4f}")
    print(f"验证集平均 mIoU: {val_metrics['val_miou']:.4f}")
    
    model.train()
    return val_metrics

# --- 6. 训练主函数 (Main Training Function) ---

def train_model(model: DeepLabV3_SingleChannel_Attn, train_dataloader: DataLoader, val_dataloader: DataLoader, 
                 criterion: nn.Module, optimizer: optim.Optimizer, num_epochs: int = NUM_EPOCHS) -> List[Dict[str, Any]]:
    """执行训练循环，并在训练结束后保存训练历史记录"""
    
    start_time = time.time()
    best_m_iou = -1.0 
    
    history: List[Dict[str, Any]] = [] 
    
    for epoch in range(num_epochs):
        # --- 训练阶段 ---
        model.train()
        running_loss = 0.0
        
        total_metrics = {'acc': 0.0, 'miou': 0.0, 'mpa': 0.0, 'mf1': 0.0}
        
        for i, (images, masks) in enumerate(train_dataloader):
            # 将数据移至指定的 DEVICE
            images = images.to(DEVICE)
            masks = masks.to(DEVICE) 

            optimizer.zero_grad()
            outputs = model(images) 
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            
            metrics = compute_metrics(outputs.detach(), masks.detach(), NUM_CLASSES)
            
            for key in total_metrics:
                total_metrics[key] += metrics[key] * images.size(0)

            if (i + 1) % 100 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_dataloader)}], Train Loss: {loss.item():.4f}, Train mIoU: {metrics['miou']:.4f}")

        # --- Epoch 结束时总结 (训练集) ---
        num_train_samples = len(train_dataloader.dataset)
        train_metrics = {
            'epoch': epoch + 1,
            'train_loss': running_loss / num_train_samples,
            'train_acc': total_metrics['acc'] / num_train_samples,
            'train_miou': total_metrics['miou'] / num_train_samples,
            'train_mpa': total_metrics['mpa'] / num_train_samples,
            'train_mf1': total_metrics['mf1'] / num_train_samples,
        }
        
        print(f"\n--- Epoch {epoch+1}/{num_epochs} 完成 (训练集) ---")
        print(f"训练集总平均 Loss: {train_metrics['train_loss']:.4f}")
        print(f"训练集总平均 mIoU: {train_metrics['train_miou']:.4f}")
        
        # --- 验证阶段 ---
        if val_dataloader:
            val_metrics = validate_model(model, val_dataloader, criterion)
            current_m_iou = val_metrics['val_miou']
            
            epoch_history = {**train_metrics, **val_metrics} 
            
            # 保存最佳模型 (基于验证集 mIoU)
            if current_m_iou > best_m_iou:
                best_m_iou = current_m_iou
                os.makedirs(os.path.dirname(MODEL_SAVE_PATH) or '.', exist_ok=True)
                torch.save(model.state_dict(), MODEL_SAVE_PATH)
                print(f"*** Val mIoU 提升至 {best_m_iou:.4f}，模型已保存至: {MODEL_SAVE_PATH} ***")
            else:
                print(f"Val mIoU ({current_m_iou:.4f}) 未超过最佳 mIoU ({best_m_iou:.4f}).")
        else:
            epoch_history = train_metrics

        history.append(epoch_history)


    end_time = time.time()
    print(f"\n训练流程完成! 总耗时: {(end_time - start_time) / 60:.2f} 分钟")
    
    return history 

# --- 7. 主函数 (Main Execution) ---

def main():
    # 检查数据目录是否存在
    if not os.path.isdir(os.path.join(VOC_ROOT, 'JPEGImages')):
        print("错误: 找不到数据集目录。请确保您已运行数据预处理脚本，并将 VOC_ROOT 设置正确。")
        print(f"期望的目录: {VOC_ROOT}")
        return

    # --- 数据增强与归一化 ---
    data_transforms = T.Compose([
        T.Resize((IMAGE_SIZE, IMAGE_SIZE)), 
        T.ToTensor(),
        T.Normalize(mean=[0.5], std=[0.5]) 
    ])
    
    # --- 实例化数据集和 DataLoader ---
    num_workers = os.cpu_count() // 2 if os.cpu_count() else 2
    
    try:
        train_dataset = VOCSegDataset(
            voc_root=VOC_ROOT, image_size=IMAGE_SIZE, image_set='train', transforms=data_transforms
        )
        train_dataloader = DataLoader(
            train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers
        )
        
        val_dataset = VOCSegDataset(
            voc_root=VOC_ROOT, image_size=IMAGE_SIZE, image_set='val', transforms=data_transforms
        )
        val_dataloader = DataLoader(
            val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers
        )

    except FileNotFoundError as e:
        print(e)
        return
        
    if len(train_dataset) == 0:
        print("训练数据集为空。")
        return

    # --- 实例化模型、损失函数和优化器 ---
    print(f"正在实例化 DeepLabV3 (带通道注意力) 模型 (Backbone: ResNet50)。")
    model = DeepLabV3_SingleChannel_Attn(num_classes=NUM_CLASSES, backbone='resnet50', pretrained=False) 
    
    # 关键：将模型移动到指定的 DEVICE
    model.to(DEVICE)
    
    criterion = nn.CrossEntropyLoss(ignore_index=255)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # --- 开始训练并获取历史记录 ---
    training_history = train_model(model, train_dataloader, val_dataloader, criterion, optimizer)
    
    # --- 保存训练历史记录到 JSON 文件 ---
    if training_history:
        os.makedirs(os.path.dirname(METRICS_SAVE_PATH) or '.', exist_ok=True)
        try:
            with open(METRICS_SAVE_PATH, 'w') as f:
                # 确保 JSON 可序列化 (特别是对于 PyTorch Tensor 和 NumPy 类型)
                json_history = [{k: (v.item() if isinstance(v, (torch.Tensor, np.generic)) else v) for k, v in epoch_data.items()} 
                                 for epoch_data in training_history]
                json.dump(json_history, f, indent=4)
            print(f"\n✅ 训练历史指标已保存至: {METRICS_SAVE_PATH}")
            
        except Exception as e:
            print(f"❌ 警告：保存指标文件时出错: {e}")

if __name__ == "__main__":
    main()