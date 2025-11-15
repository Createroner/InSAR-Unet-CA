import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.transforms.functional as F_T 
import torchvision.models.segmentation as segmentation 
# 导入 ResNet 相关的模块用于检查 Bottleneck 类型
import torchvision.models.resnet as resnet_models 
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
MODEL_SAVE_PATH = "trained_models/fcn_sea_ice_segmentation_64x64_se_best.pth" 
# 【新增】指标保存路径
METRICS_SAVE_PATH = "training_metrics/training_history_fcn_64x64_se.json"

# --- 关键修改：指定 GPU 配置 ---
GPU_ID = 2  # <--- 在这里设置您想要使用的 GPU ID
# -----------------------------------

if torch.cuda.is_available():
    try:
        # 尝试将当前进程的默认 CUDA 设备设置为 GPU_ID
        torch.cuda.set_device(GPU_ID) 
        DEVICE = torch.device(f"cuda:{GPU_ID}")
    except RuntimeError as e:
        # 如果指定的 GPU ID 不存在或设置失败，使用默认的 'cuda:0'
        DEVICE = torch.device("cuda")
        print(f"警告：设置指定的 GPU ID {GPU_ID} 失败 ({e})。系统有 CUDA，将使用默认 GPU: {DEVICE}。")
else:
    DEVICE = torch.device("cpu")

print(f"Using device: {DEVICE}")

# --- 1. Squeeze-and-Excitation (SE) 模块定义 ---

class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation Block (通道注意力机制)
    """
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        # Squeeze: 全局平均池化
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # Excitation: 两个 1x1 卷积实现 FC 层，用于计算通道注意力权重
        self.fc = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # 1. Squeeze
        y = self.avg_pool(x)
        # 2. Excitation
        y = self.fc(y)
        # 3. Scale: 加权
        return x * y.expand_as(x)

# --- 2. 封装 ResNet Bottleneck Block 以集成 SE Block ---

class BottleneckWithSE(nn.Module):
    """
    继承原 ResNet 的 Bottleneck Block，并嵌入 SE Block
    """
    def __init__(self, bottleneck):
        super(BottleneckWithSE, self).__init__()
        # 复制原有的 ResNet Bottleneck 内部模块
        self.conv1 = bottleneck.conv1
        self.bn1 = bottleneck.bn1
        self.conv2 = bottleneck.conv2
        self.bn2 = bottleneck.bn2
        self.conv3 = bottleneck.conv3
        self.bn3 = bottleneck.bn3
        self.relu = bottleneck.relu
        self.downsample = bottleneck.downsample
        
        # 核心：添加 SE Block，通道数是 Bottleneck Block 的最终输出通道数
        in_channels = bottleneck.conv3.out_channels
        self.se_block = SEBlock(in_channels)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        
        # 插入 SE Block
        out = self.se_block(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


# --- 3. FCN 模型定义 (FCN Model Definition) ---

class FCN_SingleChannel_SE(nn.Module):
    """
    封装 torchvision FCN 模型，使其支持单通道 (灰度图) 输入，适配 NUM_CLASSES，并集成 SE Block。
    """
    def __init__(self, num_classes: int = 2, backbone: str = 'resnet50', pretrained: bool = False):
        super(FCN_SingleChannel_SE, self).__init__()
        
        # 1. 加载 FCN 模型
        if backbone == 'resnet50':
            self.model = segmentation.fcn_resnet50(pretrained=pretrained, progress=True)
            LOW_LEVEL_IN_CHANNELS = 2048
            RESNET_BLOCK_TYPE = resnet_models.Bottleneck 
        elif backbone == 'resnet101':
            self.model = segmentation.fcn_resnet101(pretrained=pretrained, progress=True)
            LOW_LEVEL_IN_CHANNELS = 2048
            RESNET_BLOCK_TYPE = resnet_models.Bottleneck 
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
        
        # 权重迁移逻辑（仅在 pretrained=True 时有意义）
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
        
        # --- 核心修改 3: 集成 SE Block 到 ResNet Backbone ---
        print("正在将 SE Block 集成到 ResNet Backbone...")
        for name, module in self.model.backbone.named_children():
            if name.startswith('layer'):
                for i, block in enumerate(module):
                    if isinstance(block, RESNET_BLOCK_TYPE):
                        module[i] = BottleneckWithSE(block)
        print("SE Block 集成完成。")

        # 最后的安全检查: 修复了 'convs' 和 'Dropout' 属性错误
        # FCNHead 结构：Conv(0) -> ReLU(1) -> Dropout(2) -> Conv(3)
        if isinstance(self.model.classifier, nn.Sequential) and len(self.model.classifier) >= 4:
            # 最终的输出层是索引为 3 的 Conv2d
            final_conv = self.model.classifier[3] 
            
            if isinstance(final_conv, nn.Conv2d):
                print(f"Model configured: Conv1 in_channels={self.model.backbone.conv1.in_channels}, Classifier out_channels={final_conv.out_channels}")
            else:
                print(f"Model configured, but classifier index 3 is {type(final_conv)}, not Conv2d. Check FCNHead structure.")
        else:
             print("Model configured. Classifier structure check failed.")


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4 or x.size(1) != 1:
             raise ValueError(f"FCN_SingleChannel_SE expected input shape (B, 1, H, W), but got {x.shape}. Check DataLoader/Dataset.")
        
        # model(x) 返回一个字典，键 'out' 对应最终输出
        return self.model(x)['out']

# --- 4. 数据集加载 (Dataset Loader) ---
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

# --- 5. 性能指标计算辅助函数 (Metrics Computation) ---
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

# --- 6. 验证辅助函数 (Validation Function) ---
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

# --- 7. 训练主函数 (Main Training Function) ---
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
            
            # 安全检查
            if images.dim() != 4 or images.size(1) != 1:
                 raise RuntimeError(f"FATAL ERROR: Input image tensor shape mismatch. Expected (B, 1, H, W), but got {images.shape}.")

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

# --- 8. 主函数 (Main Execution) ---

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
    # 优化 num_workers 的设置
    num_workers = min(8, os.cpu_count() or 0)
    
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
    print("正在实例化 FCN 模型 (Backbone: ResNet50) 并集成 SE Block。不加载预训练权重。")
    model = FCN_SingleChannel_SE(num_classes=NUM_CLASSES, backbone='resnet50', pretrained=False) 
    
    criterion = nn.CrossEntropyLoss(ignore_index=255) 
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # --- 开始训练并获取历史记录 ---
    training_history = train_model(model, train_dataloader, val_dataloader, criterion, optimizer)
    
    # --- 保存训练历史记录到 JSON 文件 ---
    if training_history:
        os.makedirs(os.path.dirname(METRICS_SAVE_PATH) or '.', exist_ok=True)
        try:
            with open(METRICS_SAVE_PATH, 'w') as f:
                # 转换所有 numpy/torch 类型为 Python 原生 float
                json_history = [{k: (v if not isinstance(v, (torch.Tensor, np.floating)) else float(v)) for k, v in epoch_data.items()} 
                                for epoch_data in training_history]
                json.dump(json_history, f, indent=4)
            print(f"\n✅ 训练历史指标已保存至: {METRICS_SAVE_PATH}")
            
        except Exception as e:
            print(f"❌ 警告：保存指标文件时出错: {e}")

if __name__ == "__main__":
    main()