import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F  # <--- NEW: Import torch.nn.functional
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
NUM_EPOCHS = 100
LEARNING_RATE = 1e-4

# TODO: 【重要】模型保存路径。
MODEL_SAVE_PATH = "trained_models/fcn_sea_ice_segmentation_64x64_spatial_attn_best.pth" 
# 【新增】指标保存路径
METRICS_SAVE_PATH = "training_metrics/training_history_fcn_64x64_spatial_attn.json"

# 【修改】指定使用的 GPU ID (例如 GPU 0)
GPU_ID = 4
if torch.cuda.is_available():
    try:
        torch.cuda.set_device(GPU_ID)
        DEVICE = torch.device(f"cuda:{GPU_ID}")
    except RuntimeError:
        print(f"警告: GPU ID {GPU_ID} 不可用，将使用第一个 GPU (cuda:0) 或 CPU。")
        DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
else:
    DEVICE = torch.device("cpu")
print(f"Using device: {DEVICE}")

# --- 1. 空间注意力模块 (Spatial Attention Module) ---

class SpatialAttention(nn.Module):
    """
    简化的空间注意力模块 (Inspired by CBAM spatial attention)
    输入特征图 X -> 卷积得到注意力图 M -> X * M
    """
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        
        x_concat = torch.cat([avg_out, max_out], dim=1)
        
        attention_map = self.conv(x_concat)
        attention_weights = self.sigmoid(attention_map)
        
        return x * attention_weights

# --- 2. FCN 模型定义 (FCN Model Definition) ---

class FCN_SingleChannel_SpatialAttn(nn.Module):
    """
    封装 torchvision FCN 模型，支持单通道输入，适配 NUM_CLASSES，并添加空间注意力。
    """
    def __init__(self, num_classes: int = 2, backbone: str = 'resnet50', pretrained: bool = False):
        super(FCN_SingleChannel_SpatialAttn, self).__init__()
        
        if backbone == 'resnet50':
            self.model = segmentation.fcn_resnet50(pretrained=pretrained, progress=True)
            LOW_LEVEL_IN_CHANNELS = 2048
        elif backbone == 'resnet101':
            self.model = segmentation.fcn_resnet101(pretrained=pretrained, progress=True)
            LOW_LEVEL_IN_CHANNELS = 2048
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        # --- 核心修改 1: 适配单通道输入 ---
        original_conv1 = self.model.backbone.conv1 
        new_conv1 = nn.Conv2d(
            1, # 输入通道改为 1
            original_conv1.out_channels,
            kernel_size=original_conv1.kernel_size,
            stride=original_conv1.stride,
            padding=original_conv1.padding,
            bias=original_conv1.bias is not None
        )
        # 权重迁移逻辑
        if pretrained:
            with torch.no_grad():
                mean_weights = original_conv1.weight.mean(dim=1, keepdim=True)
                new_conv1.weight.data = mean_weights
                if original_conv1.bias is not None:
                    new_conv1.bias.data = original_conv1.bias.data
        self.model.backbone.conv1 = new_conv1
        
        # --- 核心修改 2: 替换整个分类器模块以适配类别数 (NUM_CLASSES) ---
        new_classifier = segmentation.fcn.FCNHead(LOW_LEVEL_IN_CHANNELS, num_classes)
        self.model.classifier = new_classifier
        
        # 🚨 【新增】空间注意力模块 
        self.spatial_attention = SpatialAttention() 
        
        conv_output_layer = self.model.classifier[3]
        if isinstance(conv_output_layer, nn.Conv2d):
            print(f"Model configured: Conv1 in_channels={self.model.backbone.conv1.in_channels}, Classifier out_channels={conv_output_layer.out_channels}")
        else:
            print("Model configured. Output layer type check pending (Expected Conv2d).")


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 🚨 【输入检查】
        if x.dim() != 4 or x.size(1) != 1:
            raise ValueError(f"FCN_SingleChannel_SpatialAttn expected input shape (B, 1, H, W), but got {x.shape}. Check DataLoader/Dataset.")
        
        input_shape = x.shape[-2:]
        
        # 1. 获取 ResNet Backbone 的输出
        features = self.model.backbone(x) 
        
        # 2. 从特征字典中获取主要特征
        high_level_feature = features['out'] 
        
        # 3. 🚨 【应用空间注意力】
        attended_high_level_feature = self.spatial_attention(high_level_feature)
        
        # 4. 将注意力后的特征放回特征字典
        features['out'] = attended_high_level_feature
        
        # 5. 将特征字典传入 FCN 的分类器
        x = self.model.classifier(features['out'])
        
        # 6. 最终上采样到输入尺寸
        # 修正: 使用 F.interpolate，它支持 mode (bilinear) 和 align_corners。
        x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False) 
        
        return x

# --- 3. 数据集加载 (Dataset Loader) ---
class VOCSegDataset(Dataset):
    """加载由预处理脚本生成的 VOC 格式海冰分割数据集。"""
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
        img = Image.open(img_path).convert('L') # 读取为单通道灰度图 'L'
        
        mask_path = os.path.join(self.mask_dir, f"{img_id}.png")
        mask = Image.open(mask_path).convert('L') # 读取掩膜，转为单通道
        
        if self.transforms is not None:
            img = self.transforms(img)

        # 掩膜 Resize 使用最近邻插值
        mask = F_T.resize(mask, 
                             (self.image_size, self.image_size), 
                             interpolation=T.InterpolationMode.NEAREST)
        
        mask = T.ToTensor()(mask)
        # 确保掩膜是 LongTensor 且形状为 (H, W)
        mask = mask.squeeze(0).long() 

        return img, mask

# --- 4. 性能指标计算辅助函数 (Metrics Computation) ---
def compute_metrics(outputs: torch.Tensor, masks: torch.Tensor, num_classes: int) -> Dict[str, float]:
    """计算 PA/OA, mIoU, mPA 和 mF1-Score。"""
    _, preds = torch.max(outputs, 1)
    
    valid_mask = (masks != 255)
    
    preds_flat = preds[valid_mask].cpu().numpy()
    masks_flat = masks[valid_mask].cpu().numpy()
    
    TP = np.zeros(num_classes)
    FP = np.zeros(num_classes)
    FN = np.zeros(num_classes)

    for cls in range(num_classes):
        TP[cls] = ((masks_flat == cls) & (preds_flat == cls)).sum()
        FP[cls] = ((masks_flat != cls) & (preds_flat == cls)).sum()
        FN[cls] = ((masks_flat == cls) & (preds_flat != cls)).sum()

    # 1. Overall Accuracy (OA)
    total_correct_pixels = TP.sum() 
    total_pixels = TP.sum() + FP.sum() + FN.sum() 
    acc = total_correct_pixels / total_pixels if total_pixels > 0 else 0.0

    # 2. Mean IoU (mIoU)
    union = TP + FP + FN
    iou = np.divide(TP, union, out=np.zeros_like(TP, dtype=float), where=union!=0)
    miou = np.mean(iou[union > 0]) if np.any(union > 0) else 0.0

    # 3. Mean Pixel Accuracy (mPA)
    recall = np.divide(TP, (TP + FN), out=np.zeros_like(TP, dtype=float), where=(TP + FN)!=0)
    mpa = np.mean(recall[(TP + FN) > 0]) if np.any((TP + FN) > 0) else 0.0

    # 4. Mean F1-Score (mF1)
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

# --- 5. 验证辅助函数 (Validation Function) ---
@torch.no_grad()
def validate_model(model: nn.Module, dataloader: DataLoader, criterion: nn.Module) -> Dict[str, float]:
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
def train_model(model: nn.Module, train_dataloader: DataLoader, val_dataloader: DataLoader, 
                 criterion: nn.Module, optimizer: optim.Optimizer, num_epochs: int = NUM_EPOCHS) -> List[Dict[str, Any]]:
    """执行训练循环，并在训练结束后保存训练历史记录"""
    model.to(DEVICE)
    
    start_time = time.time()
    best_m_iou = -1.0 
    history: List[Dict[str, Any]] = [] 
    
    for epoch in range(num_epochs):
        # --- 训练阶段 ---
        model.train()
        running_loss = 0.0
        total_metrics = {'acc': 0.0, 'miou': 0.0, 'mpa': 0.0, 'mf1': 0.0}
        
        for i, (images, masks) in enumerate(train_dataloader):
            images = images.to(DEVICE)
            masks = masks.to(DEVICE) 
            
            if images.dim() != 4 or images.size(1) != 1:
                raise RuntimeError(f"FATAL ERROR: Input image tensor shape mismatch BEFORE model call. Expected (B, 1, H, W), but got {images.shape}. Check DataLoader/Dataset.")

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
            train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers, pin_memory=(DEVICE.type == 'cuda')
        )
        
        val_dataset = VOCSegDataset(
            voc_root=VOC_ROOT, image_size=IMAGE_SIZE, image_set='val', transforms=data_transforms
        )
        val_dataloader = DataLoader(
            val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers, pin_memory=(DEVICE.type == 'cuda')
        )

    except FileNotFoundError as e:
        print(e)
        return
        
    if len(train_dataset) == 0:
        print("训练数据集为空。")
        return

    # --- 实例化模型、损失函数和优化器 ---
    print("正在实例化 FCN 模型 (Backbone: ResNet50)，并集成空间注意力。")
    model = FCN_SingleChannel_SpatialAttn(num_classes=NUM_CLASSES, backbone='resnet50', pretrained=False) 
    
    criterion = nn.CrossEntropyLoss(ignore_index=255)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # --- 开始训练并获取历史记录 ---
    training_history = train_model(model, train_dataloader, val_dataloader, criterion, optimizer)
    
    # --- 保存训练历史记录到 JSON 文件 ---
    if training_history:
        os.makedirs(os.path.dirname(METRICS_SAVE_PATH) or '.', exist_ok=True)
        try:
            with open(METRICS_SAVE_PATH, 'w') as f:
                json_history = [{k: (v if not isinstance(v, (torch.Tensor, np.floating)) else float(v)) for k, v in epoch_data.items()} 
                                for epoch_data in training_history]
                json.dump(json_history, f, indent=4)
            print(f"\n✅ 训练历史指标已保存至: {METRICS_SAVE_PATH}")
            
        except Exception as e:
            print(f"❌ 警告：保存指标文件时出错: {e}")

if __name__ == "__main__":
    main()