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
VOC_ROOT = "../testsize64" 

# 输入图像的尺寸
IMAGE_SIZE = 128
# 分割类别数 (背景 0 + 海冰 1)
NUM_CLASSES = 2 
# 训练参数
BATCH_SIZE = 8
NUM_EPOCHS = 25
LEARNING_RATE = 1e-4

# 【新增】指定使用的 GPU 编号。例如：0, 1, 2 等。
# 如果设置为 None，则使用 torch.device("cuda") 默认选择的 GPU。
# 如果只有一块 GPU，请设置为 0。
GPU_INDEX = 1 

# TODO: 【重要】模型保存路径。
MODEL_SAVE_PATH = "trained_models/unet_sea_ice_segmentation_model_64x64_se_best.pth" # 修改文件名以区分
# 【新增】指标保存路径
METRICS_SAVE_PATH = "training_metrics/training_history_unet_64x64_se.json" # 修改文件名以区分

# **********************************************
# DEVICE 变量将在 main() 中根据 GPU_INDEX 确定
# **********************************************

# --- 1. U-Net 模型定义 (U-Net Model Definition) ---

class SELayer(nn.Module):
    """
    Squeeze-and-Excitation 模块 (通道注意力)
    """
    def __init__(self, channel: int, reduction: int = 16):
        super(SELayer, self).__init__()
        # Squeeze 操作：全局平均池化
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # Excitation 操作：两个全连接层（MLP）
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid() # 产生 0 到 1 之间的通道权重
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        
        # 1. Squeeze
        y = self.avg_pool(x).view(b, c)
        
        # 2. Excitation
        y = self.fc(y).view(b, c, 1, 1)
        
        # 3. Scale and return
        # 将通道权重 y 乘到输入特征 x 上
        return x * y.expand_as(x)


class DoubleConv(nn.Module):
    """(Conv -> BN -> ReLU) * 2 模块，新增可选的 SE 通道注意力"""
    def __init__(self, in_channels: int, out_channels: int, use_se: bool = False): # 默认关闭，在 UNet 中控制
        super(DoubleConv, self).__init__()
        
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        
        # 可选地添加 SE 模块
        if use_se:
            # SE 模块接收的通道数是 DoubleConv 的输出通道数
            layers.append(SELayer(out_channels)) 

        self.double_conv = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)


class UNet(nn.Module):
    def __init__(self, in_channels: int = 1, num_classes: int = 2, use_se: bool = False): # 新增 use_se 参数
        super(UNet, self).__init__()
        
        # Encoder (下采样路径)
        self.inc = DoubleConv(in_channels, 64, use_se=use_se)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128, use_se=use_se))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256, use_se=use_se))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(256, 512, use_se=use_se))
        self.down4 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(512, 1024, use_se=use_se))
        
        # Decoder (上采样路径)
        self.up1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.conv1 = DoubleConv(1024, 512, use_se=use_se) 
        
        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv2 = DoubleConv(512, 256, use_se=use_se)
        
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv3 = DoubleConv(256, 128, use_se=use_se)
        
        self.up4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv4 = DoubleConv(128, 64, use_se=use_se)
        
        # 输出层
        self.outc = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # Decoder and Skip Connections
        x = self.up1(x5)
        # 确保通道数匹配
        if x.shape[2] != x4.shape[2] or x.shape[3] != x4.shape[3]:
             x = F_T.resize(x, size=(x4.shape[2], x4.shape[3]), interpolation=T.InterpolationMode.BILINEAR)
        x = torch.cat([x4, x], dim=1) 
        x = self.conv1(x)
        
        x = self.up2(x)
        if x.shape[2] != x3.shape[2] or x.shape[3] != x3.shape[3]:
             x = F_T.resize(x, size=(x3.shape[2], x3.shape[3]), interpolation=T.InterpolationMode.BILINEAR)
        x = torch.cat([x3, x], dim=1)
        x = self.conv2(x)

        x = self.up3(x)
        if x.shape[2] != x2.shape[2] or x.shape[3] != x2.shape[3]:
             x = F_T.resize(x, size=(x2.shape[2], x2.shape[3]), interpolation=T.InterpolationMode.BILINEAR)
        x = torch.cat([x2, x], dim=1)
        x = self.conv3(x)

        x = self.up4(x)
        if x.shape[2] != x1.shape[2] or x.shape[3] != x1.shape[3]:
             x = F_T.resize(x, size=(x1.shape[2], x1.shape[3]), interpolation=T.InterpolationMode.BILINEAR)
        x = torch.cat([x1, x], dim=1)
        x = self.conv4(x)

        # Output
        logits = self.outc(x)
        return logits


# --- 2. 数据集加载 (Dataset Loader) ---
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
        # 将 [1, H, W] 转换为 [H, W]，并转为 Long 类型作为 CrossEntropyLoss 的输入
        mask = mask.squeeze(0).long() 

        return img, mask

# --- 3. 性能指标计算辅助函数 (Metrics Computation) ---
def compute_metrics(outputs: torch.Tensor, masks: torch.Tensor, num_classes: int) -> Dict[str, float]:
    """
    计算 PA/OA, mIoU, mPA 和 mF1-Score。
    返回一个包含所有指标的字典。
    """
    _, preds = torch.max(outputs, 1)
    
    # 忽略索引 255
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
        # 统计 True Positives, False Positives, False Negatives
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
    # 只计算有数据的类别的平均值
    miou = np.mean(iou[union > 0]) if np.any(union > 0) else 0.0

    # --- 3. Mean Pixel Accuracy (mPA) (等于 Mean Recall) ---
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

# --- 4. 验证辅助函数 (Validation Function) ---

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
    print(f"验证集平均 mPA: {val_metrics['val_mpa']:.4f}")
    print(f"验证集平均 mF1: {val_metrics['val_mf1']:.4f}")
    
    model.train()
    return val_metrics

# --- 5. 训练主函数 (Main Training Function) ---

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
        print(f"训练集总平均 Acc (OA): {train_metrics['train_acc']:.4f}")
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
                # 保存模型时，最好只保存 state_dict
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

# --- 6. 主函数 (Main Execution) ---

def main():
    
    # 【新增】根据配置确定 DEVICE
    global DEVICE
    if torch.cuda.is_available():
        if GPU_INDEX is not None and GPU_INDEX < torch.cuda.device_count():
            DEVICE = torch.device(f"cuda:{GPU_INDEX}")
            # 【关键】设置默认 device，确保所有新张量都在此 GPU 上创建
            torch.cuda.set_device(DEVICE) 
            print(f"✅ 成功指定使用 GPU: {GPU_INDEX} ({torch.cuda.get_device_name(GPU_INDEX)})")
        else:
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
    # 根据 CPU 核心数设置 num_workers
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
    print("正在实例化 U-Net 模型 (已启用 SE 通道注意力)。")
    # 【启用通道注意力】设置 use_se=True
    model = UNet(in_channels=1, num_classes=NUM_CLASSES, use_se=True) 
    criterion = nn.CrossEntropyLoss(ignore_index=255)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # --- 开始训练并获取历史记录 ---
    training_history = train_model(model, train_dataloader, val_dataloader, criterion, optimizer, DEVICE) 
    
    # --- 保存训练历史记录到 JSON 文件 ---
    if training_history:
        os.makedirs(os.path.dirname(METRICS_SAVE_PATH) or '.', exist_ok=True)
        try:
            with open(METRICS_SAVE_PATH, 'w') as f:
                # 转换 torch.Tensor 到 Python 基本类型 (float)
                json_history = [{k: (v.item() if isinstance(v, torch.Tensor) else v) for k, v in epoch_data.items()} 
                                for epoch_data in training_history]
                json.dump(json_history, f, indent=4)
            print(f"\n✅ 训练历史指标已保存至: {METRICS_SAVE_PATH}")
            
            print("\n--- 绘图数据预览 (第一轮) ---")
            if len(json_history) > 0:
                print(json.dumps(json_history[0], indent=4))
                
        except Exception as e:
            print(f"❌ 警告：保存指标文件时出错: {e}")

if __name__ == "__main__":
    main()