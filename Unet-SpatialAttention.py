import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
# 导入 functional 模块用于掩膜的 Resize
import torchvision.transforms.functional as F_T 
import numpy as np 
from PIL import Image
import os
import time
import json 
from typing import Tuple, List, Dict, Any

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

# 【重要】指定使用的 GPU 编号。例如：0, 1, 2 等。设置为 None 则使用默认 GPU。
GPU_INDEX = 1 

# TODO: 【重要】模型保存路径。
MODEL_SAVE_PATH = "trained_models/unet_sea_ice_segmentation_model_64x64_attention_best.pth" 
# 【新增】指标保存路径
METRICS_SAVE_PATH = "training_metrics/training_history_unet_64x64_attention.json"

# **********************************************
# DEVICE 变量将在 main() 中根据 GPU_INDEX 确定
# **********************************************

# --- 1. U-Net 模型基本组件定义 ---
class DoubleConv(nn.Module):
    """(Conv -> BN -> ReLU) * 2 模块"""
    def __init__(self, in_channels: int, out_channels: int):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)

# --- 2. 空间注意力模块定义 (Spatial Attention Module) ---
class SpatialAttention(nn.Module):
    """
    简化的空间注意力模块 (基于 CBAM 空间子模块的思想)。
    通过对通道维度的 AvgPool 和 MaxPool 压缩，学习一个空间注意力图。
    """
    def __init__(self):
        super(SpatialAttention, self).__init__()
        # 使用 DoubleConv 来增强 2 个压缩通道的特征提取
        self.compress_and_map = DoubleConv(2, 1) 
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. 沿通道维度压缩：AvgPool 和 MaxPool
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        # 在通道维度上拼接 (Batch x 2 x H x W)
        x_compress = torch.cat([avg_out, max_out], dim=1) 
        
        # 2. 生成空间注意力图
        attention_map = self.compress_and_map(x_compress)
        
        # 3. 归一化并施加注意力
        scale = self.sigmoid(attention_map)
        return x * scale # 施加注意力

# --- 3. 包含空间注意力 (SA) 的 U-Net 模型定义 ---
class UNet(nn.Module):
    def __init__(self, in_channels: int = 1, num_classes: int = 2):
        super(UNet, self).__init__()
        
        # Encoder (下采样路径)
        self.inc = DoubleConv(in_channels, 64)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(256, 512))
        self.down4 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(512, 1024))
        
        # Decoder (上采样路径)
        self.up1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.conv1 = DoubleConv(1024, 512) 
        
        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv2 = DoubleConv(512, 256)
        
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv3 = DoubleConv(256, 128)
        
        self.up4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv4 = DoubleConv(128, 64)
        
        # 空间注意力模块的实例化 (用于跳跃连接融合后)
        self.sa1 = SpatialAttention() 
        self.sa2 = SpatialAttention() 
        self.sa3 = SpatialAttention() 
        self.sa4 = SpatialAttention() 
        
        # 输出层
        self.outc = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # Decoder and Skip Connections (应用 SA 模块)
        
        # UP 1
        x = self.up1(x5)
        x = torch.cat([x4, x], dim=1) 
        x = self.sa1(x) # 应用空间注意力
        x = self.conv1(x)
        
        # UP 2
        x = self.up2(x)
        x = torch.cat([x3, x], dim=1)
        x = self.sa2(x) # 应用空间注意力
        x = self.conv2(x)

        # UP 3
        x = self.up3(x)
        x = torch.cat([x2, x], dim=1)
        x = self.sa3(x) # 应用空间注意力
        x = self.conv3(x)

        # UP 4
        x = self.up4(x)
        x = torch.cat([x1, x], dim=1)
        x = self.sa4(x) # 应用空间注意力
        x = self.conv4(x)

        # Output
        logits = self.outc(x)
        return logits


# --- 4. 数据集加载 (Dataset Loader) ---
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
        img = Image.open(img_path).convert('L') 
        
        mask_path = os.path.join(self.mask_dir, f"{img_id}.png")
        mask = Image.open(mask_path).convert('L') 
        
        if self.transforms is not None:
            img = self.transforms(img)

        # 掩膜 Resize 使用最近邻插值
        mask = F_T.resize(mask, 
                              (self.image_size, self.image_size), 
                              interpolation=T.InterpolationMode.NEAREST)
        
        mask = T.ToTensor()(mask)
        mask = mask.squeeze(0).long() 

        return img, mask

# --- 5. 性能指标计算辅助函数 (Metrics Computation) ---
def compute_metrics(outputs: torch.Tensor, masks: torch.Tensor, num_classes: int) -> Dict[str, float]:
    """
    计算 PA/OA, mIoU, mPA 和 mF1-Score。
    """
    _, preds = torch.max(outputs, 1)
    
    # 忽略 index=255 的像素
    valid_mask = (masks != 255)
    
    preds_flat = preds[valid_mask]
    masks_flat = masks[valid_mask]
    
    # 将 Tensor 转移到 CPU 并转为 NumPy 
    preds_flat = preds_flat.cpu().numpy()
    masks_flat = masks_flat.cpu().numpy()
    
    TP = np.zeros(num_classes)
    FP = np.zeros(num_classes)
    FN = np.zeros(num_classes)

    for cls in range(num_classes):
        TP[cls] = ((masks_flat == cls) & (preds_flat == cls)).sum()
        FP[cls] = ((masks_flat != cls) & (preds_flat == cls)).sum()
        FN[cls] = ((masks_flat == cls) & (preds_flat != cls)).sum()

    # --- 1. Overall Accuracy (OA) / Pixel Accuracy (PA) ---
    total_correct_pixels = TP.sum() 
    total_pixels = TP.sum() + FP.sum() + FN.sum() 
    acc = total_correct_pixels / total_pixels if total_pixels > 0 else 0.0

    # --- 2. Mean IoU (mIoU) ---
    union = TP + FP + FN
    iou = np.divide(TP, union, out=np.zeros_like(TP, dtype=float), where=union!=0)
    # 只计算存在样本的类别的平均值
    miou = np.mean(iou[union > 0]) if np.any(union > 0) else 0.0

    # --- 3. Mean Pixel Accuracy (mPA) ---
    recall = np.divide(TP, (TP + FN), out=np.zeros_like(TP, dtype=float), where=(TP + FN)!=0)
    mpa = np.mean(recall[(TP + FN) > 0]) if np.any((TP + FN) > 0) else 0.0

    # --- 4. Mean F1-Score (mF1) ---
    precision = np.divide(TP, (TP + FP), out=np.zeros_like(TP, dtype=float), where=(TP + FP)!=0)
    f1_scores = np.divide(2 * precision * recall, (precision + recall), 
                          out=np.zeros_like(TP, dtype=float), 
                          where=(precision + recall)!=0)
    mf1 = np.mean(f1_scores[(TP + FN) > 0]) if np.any((TP + FN) > 0) else 0.0 
    
    return {
        'acc': acc, 
        'miou': miou,
        'mpa': mpa,
        'mf1': mf1
    }

# --- 6. 验证辅助函数 (Validation Function) ---

@torch.no_grad()
def validate_model(model: UNet, dataloader: DataLoader, criterion: nn.Module, device: torch.device) -> Dict[str, float]:
    """在验证集上评估模型的性能，返回指标字典"""
    model.eval()
    
    running_loss = 0.0
    total_metrics = {'acc': 0.0, 'miou': 0.0, 'mpa': 0.0, 'mf1': 0.0}
    num_samples = 0

    for images, masks in dataloader:
        images = images.to(device)
        masks = masks.to(device)
        
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
    print(f"验证集平均 Acc (OA): {val_metrics['val_acc']:.4f}")
    print(f"验证集平均 mIoU: {val_metrics['val_miou']:.4f}")
    
    model.train()
    return val_metrics

# --- 7. 训练主函数 (Main Training Function) ---

def train_model(model: UNet, train_dataloader: DataLoader, val_dataloader: DataLoader, 
                 criterion: nn.Module, optimizer: optim.Optimizer, device: torch.device, num_epochs: int = NUM_EPOCHS) -> List[Dict[str, Any]]:
    """执行训练循环，并在训练结束后保存训练历史记录"""
    model.to(device) 
    
    start_time = time.time()
    best_m_iou = -1.0 
    
    history: List[Dict[str, Any]] = [] 
    
    for epoch in range(num_epochs):
        # --- 训练阶段 ---
        model.train()
        running_loss = 0.0
        
        total_metrics = {'acc': 0.0, 'miou': 0.0, 'mpa': 0.0, 'mf1': 0.0}
        
        for i, (images, masks) in enumerate(train_dataloader):
            images = images.to(device)
            masks = masks.to(device) 

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
                print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_dataloader)}], Train Loss: {loss.item():.4f}, Train Acc: {metrics['acc']:.4f}, Train mIoU: {metrics['miou']:.4f}")

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
            val_metrics = validate_model(model, val_dataloader, criterion, device)
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

# --- 8. 主函数 (Main Execution) ---

def main():
    
    # 根据配置确定 DEVICE
    global DEVICE
    if torch.cuda.is_available():
        if GPU_INDEX is not None and GPU_INDEX < torch.cuda.device_count():
            DEVICE = torch.device(f"cuda:{GPU_INDEX}")
            # 设置默认 device，确保所有新张量都在此 GPU 上创建
            torch.cuda.set_device(DEVICE) 
            print(f"✅ 成功指定使用 GPU: {GPU_INDEX} ({torch.cuda.get_device_name(GPU_INDEX)})")
        else:
            # 默认为 cuda:0
            DEVICE = torch.device("cuda")
            print(f"⚠️ GPU_INDEX 设置无效或超出范围，使用默认 GPU: {DEVICE}")
    else:
        DEVICE = torch.device("cpu")

    print(f"Using device: {DEVICE}")

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
            train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers, pin_memory=True
        )
        
        val_dataset = VOCSegDataset(
            voc_root=VOC_ROOT, image_size=IMAGE_SIZE, image_set='val', transforms=data_transforms
        )
        val_dataloader = DataLoader(
            val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers, pin_memory=True
        )

    except FileNotFoundError as e:
        print(e)
        return
        
    if len(train_dataset) == 0:
        print("训练数据集为空。")
        return

    # --- 实例化模型、损失函数和优化器 ---
    print("正在实例化 U-Net 模型 (含空间注意力)。")
    model = UNet(in_channels=1, num_classes=NUM_CLASSES) 
    criterion = nn.CrossEntropyLoss(ignore_index=255)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # --- 开始训练并获取历史记录 ---
    training_history = train_model(model, train_dataloader, val_dataloader, criterion, optimizer, DEVICE) 
    
    # --- 保存训练历史记录到 JSON 文件 ---
    if training_history:
        os.makedirs(os.path.dirname(METRICS_SAVE_PATH) or '.', exist_ok=True)
        try:
            with open(METRICS_SAVE_PATH, 'w') as f:
                # 转换 PyTorch Tensor 为 Python 原生类型以便 JSON 序列化
                json_history = [{k: (v.item() if isinstance(v, torch.Tensor) else v) for k, v in epoch_data.items()} 
                                for epoch_data in training_history]
                json.dump(json_history, f, indent=4)
            print(f"\n✅ 训练历史指标已保存至: {METRICS_SAVE_PATH}")
            
        except Exception as e:
            print(f"❌ 警告：保存指标文件时出错: {e}")

if __name__ == "__main__":
    main()