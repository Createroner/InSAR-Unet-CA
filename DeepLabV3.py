import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
# 导入 functional 模块用于掩膜的 Resize
import torchvision.transforms.functional as F_T 
# 导入 DeepLabV3 模块
import torchvision.models.segmentation as segmentation 
import numpy as np
from PIL import Image
import os
import time
import json 
from typing import Tuple, List, Dict, Any

# --- 配置参数 (Configuration Parameters) ---

# TODO: 【重要】请将此路径修改为您运行数据预处理脚本后生成的 'VOCdevkit/VOC2012' 目录的实际位置。
VOC_ROOT = "../../testsize64" 

# 输入图像的尺寸
IMAGE_SIZE = 64
# 分割类别数 (背景 0 + 海冰 1)
NUM_CLASSES = 2 
# 训练参数
BATCH_SIZE = 8
NUM_EPOCHS = 25
LEARNING_RATE = 1e-4

# TODO: 【重要】模型保存路径。
# 注意：已更改模型名称以反映 DeeplabV3
MODEL_SAVE_PATH = "trained_models/deeplabv3_sea_ice_segmentation_64x64_best.pth" 
# 【新增】指标保存路径
METRICS_SAVE_PATH = "training_metrics/training_history_deeplabv3_64x64.json"

# 用于检查 GPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# --- 1. DeepLabV3 模型定义 (DeepLabV3 Model Definition) ---

class DeepLabV3_SingleChannel(nn.Module):
    """
    封装 torchvision DeepLabV3 模型，使其支持单通道 (灰度图) 输入。
    
    注意：当 pretrained=False 时，模型完全从随机初始化开始训练。
    """
    def __init__(self, num_classes: int = 2, backbone: str = 'resnet50', pretrained: bool = True):
        super(DeepLabV3_SingleChannel, self).__init__()
        
        # 1. 加载 DeepLabV3 模型
        if backbone == 'resnet50':
            # 加载时不使用预训练权重
            self.model = segmentation.deeplabv3_resnet50(pretrained=pretrained, progress=True)
        elif backbone == 'resnet101':
            # 加载时不使用预训练权重
            self.model = segmentation.deeplabv3_resnet101(pretrained=pretrained, progress=True)
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        # 2. 修改输出层以匹配类别数
        # DeepLabV3 的分类器是 self.model.classifier.4
        self.model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=(1, 1), stride=(1, 1))

        # 3. 适配单通道输入：修改 backbone (ResNet) 的第一层卷积
        original_conv1 = self.model.backbone.conv1 
        
        # 创建新的单通道卷积层
        # 使用原始参数，但 in_channels=1
        new_conv1 = nn.Conv2d(
            1, 
            original_conv1.out_channels,
            kernel_size=original_conv1.kernel_size,
            stride=original_conv1.stride,
            padding=original_conv1.padding,
            bias=original_conv1.bias is not None
        )
        
        # 如果使用预训练权重，则复制原 Conv2d (3, 64) 的平均权重到新 Conv2d (1, 64)
        if pretrained:
            # 此块在传入 pretrained=False 时将被跳过
            # 权重依然是随机初始化的（除了 ResNet 默认的层初始化）
            with torch.no_grad():
                # 计算三个输入通道的平均权重
                mean_weights = original_conv1.weight.mean(dim=1, keepdim=True)
                new_conv1.weight.data = mean_weights
                if original_conv1.bias is not None:
                    new_conv1.bias.data = original_conv1.bias.data
        else:
            # 如果不使用预训练权重，我们需要确保新创建的 new_conv1 权重被正确初始化
            # 默认情况下，PyTorch 会对新创建的层进行 Kaiming/Uniform 初始化，这里保持默认即可
            pass

        # 替换 backbone 的第一层
        self.model.backbone.conv1 = new_conv1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # DeepLabV3 模型返回一个字典，其中分割结果在 'out' 键下
        return self.model(x)['out']

# --- 2. 数据集加载 (Dataset Loader) ---
# （保持不变）

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

# --- 3. 性能指标计算辅助函数 (Metrics Computation) ---
# （保持不变）

def compute_metrics(outputs: torch.Tensor, masks: torch.Tensor, num_classes: int) -> Tuple[float, float]:
    """
    计算像素准确率 (Pixel Accuracy) 和平均交并比 (Mean IoU, mIoU)。
    """
    _, preds = torch.max(outputs, 1)
    
    # 忽略索引 255 的像素
    valid_mask = (masks != 255)
    
    # 1. 像素准确率
    correct_pixels = ((preds == masks) & valid_mask).sum().item() 
    total_pixels = valid_mask.sum().item()
    
    pixel_acc = correct_pixels / total_pixels if total_pixels > 0 else 0.0
    
    # 2. 平均交并比
    iou_list: List[float] = []
    preds_flat = preds[valid_mask]
    masks_flat = masks[valid_mask]
    
    for cls in range(num_classes):
        pred_mask = (preds_flat == cls)
        true_mask = (masks_flat == cls)
        
        intersection = (pred_mask & true_mask).sum().item()
        union = (pred_mask | true_mask).sum().item()
        
        if union > 0:
            iou = intersection / union
            iou_list.append(iou)
        elif intersection == 0:
            # 两个掩膜都没有该类别，不计入平均值
            continue 
        
    mean_iou = sum(iou_list) / len(iou_list) if iou_list else 0.0
    
    return pixel_acc, mean_iou

# --- 4. 验证辅助函数 (Validation Function) ---
# （保持不变）

@torch.no_grad()
def validate_model(model: DeepLabV3_SingleChannel, dataloader: DataLoader, criterion: nn.Module) -> Dict[str, float]:
    """在验证集上评估模型的性能，返回指标字典"""
    model.eval()
    
    running_loss = 0.0
    total_pixel_acc = 0.0
    total_mean_iou = 0.0
    num_samples = 0

    for images, masks in dataloader:
        images = images.to(DEVICE)
        masks = masks.to(DEVICE)
        
        outputs = model(images)
        loss = criterion(outputs, masks)
        
        running_loss += loss.item() * images.size(0)
        
        pixel_acc, mean_iou = compute_metrics(outputs.detach().cpu(), masks.detach().cpu(), NUM_CLASSES)
        total_pixel_acc += pixel_acc * images.size(0)
        total_mean_iou += mean_iou * images.size(0)
        num_samples += images.size(0)

    if num_samples > 0:
        val_metrics = {
            'val_loss': running_loss / num_samples,
            'val_acc': total_pixel_acc / num_samples,
            'val_miou': total_mean_iou / num_samples 
        }
    else:
        val_metrics = {'val_loss': 0.0, 'val_acc': 0.0, 'val_miou': 0.0}

    print(f"\n--- 验证集评估结果 ---")
    print(f"验证集平均 Loss: {val_metrics['val_loss']:.4f}")
    print(f"验证集平均 Acc: {val_metrics['val_acc']:.4f}")
    print(f"验证集平均 mIoU: {val_metrics['val_miou']:.4f}")
    
    model.train()
    return val_metrics

# --- 5. 训练主函数 (Main Training Function) ---
# （保持不变）

def train_model(model: DeepLabV3_SingleChannel, train_dataloader: DataLoader, val_dataloader: DataLoader, 
                criterion: nn.Module, optimizer: optim.Optimizer, num_epochs: int = NUM_EPOCHS) -> List[Dict[str, Any]]:
    """执行训练循环，并在训练结束后保存训练历史记录"""
    model.to(DEVICE)
    
    start_time = time.time()
    best_m_iou = -1.0 
    
    # 用于保存所有 Epoch 指标的列表
    history: List[Dict[str, Any]] = [] 
    
    for epoch in range(num_epochs):
        # --- 训练阶段 ---
        model.train()
        running_loss = 0.0
        total_pixel_acc = 0.0
        total_mean_iou = 0.0
        
        for i, (images, masks) in enumerate(train_dataloader):
            images = images.to(DEVICE)
            masks = masks.to(DEVICE) 

            optimizer.zero_grad()
            outputs = model(images) 
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            
            # 指标计算 (Step-wise metrics for logging)
            pixel_acc, mean_iou = compute_metrics(outputs.detach().cpu(), masks.detach().cpu(), NUM_CLASSES)
            total_pixel_acc += pixel_acc * images.size(0)
            total_mean_iou += mean_iou * images.size(0)

            if (i + 1) % 100 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_dataloader)}], Train Loss: {loss.item():.4f}, Train Acc: {pixel_acc:.4f}, Train mIoU: {mean_iou:.4f}")

        # --- Epoch 结束时总结 (训练集) ---
        num_train_samples = len(train_dataloader.dataset)
        train_metrics = {
            'epoch': epoch + 1,
            'train_loss': running_loss / num_train_samples,
            'train_acc': total_pixel_acc / num_train_samples,
            'train_miou': total_mean_iou / num_train_samples 
        }
        
        print(f"\n--- Epoch {epoch+1}/{num_epochs} 完成 (训练集) ---")
        print(f"训练集总平均 Loss: {train_metrics['train_loss']:.4f}")
        print(f"训练集总平均 Acc: {train_metrics['train_acc']:.4f}")
        print(f"训练集总平均 mIoU: {train_metrics['train_miou']:.4f}")
        
        # --- 验证阶段 ---
        if val_dataloader:
            val_metrics = validate_model(model, val_dataloader, criterion)
            current_m_iou = val_metrics['val_miou']
            
            # 合并训练和验证指标
            epoch_history = {**train_metrics, **val_metrics} 
            
            # 保存最佳模型 (基于验证集 mIoU)
            if current_m_iou > best_m_iou:
                best_m_iou = current_m_iou
                # 保存模型
                os.makedirs(os.path.dirname(MODEL_SAVE_PATH) or '.', exist_ok=True)
                torch.save(model.state_dict(), MODEL_SAVE_PATH)
                print(f"*** Val mIoU 提升至 {best_m_iou:.4f}，模型已保存至: {MODEL_SAVE_PATH} ***")
            else:
                print(f"Val mIoU ({current_m_iou:.4f}) 未超过最佳 mIoU ({best_m_iou:.4f}).")
        else:
            epoch_history = train_metrics

        # 保存当前 Epoch 的历史记录
        history.append(epoch_history)


    end_time = time.time()
    print(f"\n训练流程完成! 总耗时: {(end_time - start_time) / 60:.2f} 分钟")
    
    return history # 返回历史记录列表

# --- 6. 主函数 (Main Execution) ---
# （保持不变）

def main():
    # 检查数据目录是否存在
    if not os.path.isdir(os.path.join(VOC_ROOT, 'JPEGImages')):
        print("错误: 找不到数据集目录。请确保您已运行数据预处理脚本，并将 VOC_ROOT 设置正确。")
        print(f"期望的目录: {VOC_ROOT}")
        return

    # --- 数据增强与归一化 ---
    # 单通道灰度图，使用 [0.5] 和 [0.5] 进行归一化
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
    # *** 保持 DeepLabV3 模型，且 pretrained=False ***
    print("正在实例化 DeepLabV3 模型 (Backbone: ResNet50)。不加载预训练权重。")
    model = DeepLabV3_SingleChannel(num_classes=NUM_CLASSES, backbone='resnet50', pretrained=False) 
    
    # CrossEntropyLoss 是分割任务的标准损失函数，ignore_index=255 用于忽略掩膜中的边界或无效值
    criterion = nn.CrossEntropyLoss(ignore_index=255)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # --- 开始训练并获取历史记录 ---
    training_history = train_model(model, train_dataloader, val_dataloader, criterion, optimizer)
    
    # --- 保存训练历史记录到 JSON 文件 ---
    if training_history:
        os.makedirs(os.path.dirname(METRICS_SAVE_PATH) or '.', exist_ok=True)
        try:
            with open(METRICS_SAVE_PATH, 'w') as f:
                # 确保 JSON 可序列化
                json_history = [{k: (v.item() if isinstance(v, torch.Tensor) else v) for k, v in epoch_data.items()} 
                                for epoch_data in training_history]
                json.dump(json_history, f, indent=4)
            print(f"\n✅ 训练历史指标已保存至: {METRICS_SAVE_PATH}")
            
            # 打印用于绘图的关键步骤
            print("\n--- 绘图数据预览 ---")
            print("请使用 'epoch', 'train_loss', 'val_loss', 'train_miou', 'val_miou' 等键值进行绘图。")
            if len(json_history) > 0:
                print(json.dumps(json_history[0], indent=4))
                
        except Exception as e:
            print(f"❌ 警告：保存指标文件时出错: {e}")

if __name__ == "__main__":
    main()