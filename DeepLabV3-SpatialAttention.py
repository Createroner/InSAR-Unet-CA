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
# 新增：导入数学工具，用于指标计算
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

# TODO: 【重要】模型保存路径。
MODEL_SAVE_PATH = "trained_models/deeplabv3_sea_ice_segmentation_64x64_attn_best.pth" # 修改名称以区分
# 【新增】指标保存路径
METRICS_SAVE_PATH = "training_metrics/training_history_deeplabv3_64x64_attn.json" # 修改名称以区分

# 用于检查 GPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# --- 1. 空间注意力模块定义 (Spatial Attention Module) ---

class SpatialAttentionModule(nn.Module):
    """
    一个简单的空间注意力模块 (SAM)。
    它沿着通道轴应用平均池化和最大池化，然后将结果拼接，通过一个卷积层生成空间注意力图。
    """
    def __init__(self, kernel_size: int = 7):
        super(SpatialAttentionModule, self).__init__()
        # 输入通道为 2 (AvgPool + MaxPool)
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. 沿着通道轴进行平均池化和最大池化 (keepdim=True 保持维度)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        
        # 2. 拼接结果
        out = torch.cat([avg_out, max_out], dim=1) 
        
        # 3. 卷积并 Sigmoid 得到注意力图 (Attention Map)
        attention_map = self.sigmoid(self.conv(out))
        
        # 4. 将注意力图乘回原特征图 (Element-wise multiplication)
        return x * attention_map
    
# --- 2. 带注意力机制的 DeepLabV3 模型定义 (DeepLabV3 Model Definition with Attention) ---

class DeepLabV3_SingleChannel_Attn(nn.Module):
    """
    封装 DeepLabV3 模型，支持单通道输入，并在 ASPP 输出后加入空间注意力模块。
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

        # 2. 修改输出层以匹配类别数
        # 原始模型中，ASPP 输出后会有一个 1x1 卷积将通道数降到 256
        # 然后是上采样和最终分类卷积
        final_conv_in_channels = 256
        self.model.classifier[4] = nn.Conv2d(final_conv_in_channels, num_classes, kernel_size=(1, 1), stride=(1, 1))

        # 3. 适配单通道输入：修改 backbone (ResNet) 的第一层卷积 (与原代码一致)
        original_conv1 = self.model.backbone.conv1 
        new_conv1 = nn.Conv2d(
            1, original_conv1.out_channels, kernel_size=original_conv1.kernel_size,
            stride=original_conv1.stride, padding=original_conv1.padding,
            bias=original_conv1.bias is not None
        )
        if pretrained:
            with torch.no_grad():
                mean_weights = original_conv1.weight.mean(dim=1, keepdim=True)
                new_conv1.weight.data = mean_weights
                if original_conv1.bias is not None:
                    new_conv1.bias.data = original_conv1.bias.data
        self.model.backbone.conv1 = new_conv1
        
        # 4. 【新增】实例化空间注意力模块
        # 我们将其插入到 DeepLabV3 的 ASPP 模块输出 (即 classifier[0] 的输出) 之后。
        # 此时特征图的通道数通常为 256。但为了简化，我们先直接应用到最终输出（通道数=NUM_CLASSES=2）
        # 注意：应用在分类前的特征图上效果可能更好，但需要更深入地修改 torchvision 源码。
        # 此处我们将其视为一个后处理步骤，应用在最终上采样之前的特征上 (即 classifier[4] 的输入/输出之间)
        
        # 为了避免修改 torchvision 的内部 forward，我们将其作为一个**前置模块**，
        # 在模型的 backbone 和 ASPP/Decoder 之间（或者最终分类之前）插入。
        
        # 重新定义模型，将分类头拆开，以便插入注意力。
        self.backbone = self.model.backbone
        self.aspp = self.model.classifier[0] 
        self.post_aspp_conv = self.model.classifier[1]
        self.upsample_conv = self.model.classifier[4] # 最终的 1x1 卷积

        # 5. 【新增】在 ASPP 之后，最终分类之前，插入注意力模块
        # 我们将注意力放在 aspp_output (256通道) 和 post_aspp_conv (256通道) 之间
        self.attention_module = SpatialAttentionModule(kernel_size=7)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_shape = x.shape[-2:]
        
        # 1. Backbone (ResNet) 特征提取
        features = self.backbone(x)
        x = features['out']

        # 2. ASPP 模块
        x = self.aspp(x)
        x = self.post_aspp_conv(x)
        
        # 3. 【新增】应用空间注意力
        # 注意力在这里对 256 通道的特征进行加权
        x = self.attention_module(x) 

        # 4. 最终分类卷积 (降维到 NUM_CLASSES)
        x = self.upsample_conv(x)

        # 5. 上采样到原始输入尺寸
        x = F_T.resize(x, input_shape, interpolation=T.InterpolationMode.BILINEAR)

        return x

# --- 3. 数据集加载 (Dataset Loader) ---
# 代码保持不变，为节省空间省略

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
    
    # 累加所有指标的总和
    total_metrics = {'acc': 0.0, 'miou': 0.0, 'mpa': 0.0, 'mf1': 0.0}
    num_samples = 0

    for images, masks in dataloader:
        images = images.to(DEVICE)
        masks = masks.to(DEVICE)
        
        outputs = model(images)
        loss = criterion(outputs, masks)
        
        running_loss += loss.item() * images.size(0)
        
        # 计算所有指标
        metrics = compute_metrics(outputs.detach(), masks.detach(), NUM_CLASSES)
        
        # 累加指标
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

# --- 6. 训练主函数 (Main Training Function) ---

def train_model(model: DeepLabV3_SingleChannel_Attn, train_dataloader: DataLoader, val_dataloader: DataLoader, 
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
        
        # 累加所有指标的总和
        total_metrics = {'acc': 0.0, 'miou': 0.0, 'mpa': 0.0, 'mf1': 0.0}
        
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
            metrics = compute_metrics(outputs.detach(), masks.detach(), NUM_CLASSES)
            
            # 累加指标
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
        print(f"训练集总平均 Acc (OA): {train_metrics['train_acc']:.4f}")
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
    
    return history 

# --- 7. 主函数 (Main Execution) ---

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
    print("正在实例化 DeepLabV3 (带空间注意力) 模型 (Backbone: ResNet50)。不加载预训练权重。")
    # 【重要】使用新的模型类 DeepLabV3_SingleChannel_Attn
    model = DeepLabV3_SingleChannel_Attn(num_classes=NUM_CLASSES, backbone='resnet50', pretrained=False) 
    
    # CrossEntropyLoss 是分割任务的标准损失函数
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