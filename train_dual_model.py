import sys
import os
import cv2
import logging
from datetime import datetime
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
import torchvision.models as models
from torch import optim
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import numpy as np
from constants import OFFICIAL_CLASS_NAMES, NUM_CLASSES
import random

def setup_logging():
    log_filename = f"train_dual_model_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    log_format = '%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
    
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=[
            logging.FileHandler(log_filename, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    logger.info(f"日志文件已创建：{log_filename}")
    return logger, log_filename

logger, log_file = setup_logging()

RAW_ID_TO_NAME = {
    0: "SUV",
    1: "box_truck",
    2: "bus",
    3: "flatbed_truck",
    4: "motorcycle",
    5: "pickup_truck",
    6: "pickup_truck_w_trailer",
    7: "sedan",
    8: "semi_w_trailer",
    9: "van"
}
RAW_NAME_TO_ID = {v: k for k, v in RAW_ID_TO_NAME.items()}

MAJOR_CLASS_INDICES = [0, 5, 7, 9]
MINOR_CLASS_INDICES = [1, 2, 3, 4, 6, 8]

LABEL2MAJOR = {
    0: 0,    # SUV → 0
    5: 1,    # pickup_truck → 1
    7: 2,    # sedan → 2
    9: 3,    # van → 3
    1: 4,    # box_truck → 4
    2: 4,    # bus → 4
    3: 4,    # flatbed_truck → 4
    4: 4,    # motorcycle → 4
    6: 4,    # pickup_truck_w_trailer → 4
    8: 4     # semi_w_trailer → 4
}
MINOR_UNIFIED_LABEL = 4  

MINOR_LABEL_MAP = {
    1: 0,    # box_truck → 0
    2: 1,    # bus → 1
    3: 2,    # flatbed_truck → 2
    4: 3,    # motorcycle → 3
    6: 4,    # pickup_truck_w_trailer → 4
    8: 5     # semi_w_trailer → 5
}
MINOR_OUTPUT_TO_RAW = {v: k for k, v in MINOR_LABEL_MAP.items()}

logger.info("="*80)
logger.info("📌 硬编码类别映射初始化（基于原始数据）：")
logger.info(f"   原始ID→类别名：{RAW_ID_TO_NAME}")
logger.info(f"   大类索引（样本数多）：{MAJOR_CLASS_INDICES} → {[RAW_ID_TO_NAME[idx] for idx in MAJOR_CLASS_INDICES]}")
logger.info(f"   小类索引（样本数少）：{MINOR_CLASS_INDICES} → {[RAW_ID_TO_NAME[idx] for idx in MINOR_CLASS_INDICES]}")
logger.info(f"   大类映射：{LABEL2MAJOR}")
logger.info(f"   小类映射：{MINOR_LABEL_MAP}")
logger.info("="*80)

BASE_DIR = Path(__file__).resolve().parent
sys.path.append(BASE_DIR)

from utils.utils_reg import FocalLoss, da_loss
from feature_extractor import FeatureExtractor

device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
logger.info(f"使用设备：{device}")

# ========================
# 1. 配置参数
# ========================
MAJOR_OOD_THRESHOLD = 0.7 
MINOR_OOD_THRESHOLD = 0.6
UNFREEZE_LAYERS = ['layer4']  
MINOR_TRAIN_EPOCHS = 30
MINOR_THRESHOLD = 0.03        
SAMPLE_TOTAL = 80000          

LOAD_PRETRAINED_MAJOR = True  
MAJOR_SAR_MODEL_PATH = BASE_DIR / "resnet_Major_SAR" / "best_ema.pth"  
MAJOR_NUM_CLASSES = 5

logger.info("="*80)
logger.info("训练配置参数：")
logger.info(f"MAJOR_OOD_THRESHOLD: {MAJOR_OOD_THRESHOLD}")
logger.info(f"MINOR_OOD_THRESHOLD: {MINOR_OOD_THRESHOLD}")
logger.info(f"UNFREEZE_LAYERS: {UNFREEZE_LAYERS}")
logger.info(f"MINOR_TRAIN_EPOCHS: {MINOR_TRAIN_EPOCHS}")
logger.info(f"MINOR_THRESHOLD: {MINOR_THRESHOLD}")
logger.info(f"SAMPLE_TOTAL: {SAMPLE_TOTAL}")
logger.info(f"LOAD_PRETRAINED_MAJOR: {LOAD_PRETRAINED_MAJOR}")
if LOAD_PRETRAINED_MAJOR:
    logger.info(f"MAJOR_SAR_MODEL_PATH: {MAJOR_SAR_MODEL_PATH}")
    logger.info(f"MAJOR_NUM_CLASSES: {MAJOR_NUM_CLASSES}")
logger.info("="*80)

# ========================
# 加载预训练大类模型函数
# ========================
def load_pretrained_major_model(model_path, num_classes, device):
    model = FeatureExtractor('resnet101', num_classes=num_classes, dropout=0.6).to(device)
    
    if not os.path.exists(model_path):
        logger.error(f"预训练大类模型权重文件不存在：{model_path}")
        sys.exit(1)
    
    try:
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint)
        logger.info(f"✅ 成功加载预训练大类SAR模型：{model_path}")
        model.eval()
        return model
    except Exception as e:
        logger.error(f"加载预训练大类模型失败：{str(e)}")
        logger.error("请检查：1.模型路径是否正确 2.模型输出类别数是否匹配 3.权重文件是否损坏")
        sys.exit(1)

# ========================
# 2. 数据增强
# ========================
class AddSpeckleNoise:
    def __init__(self, prob=0.5, sigma=0.1):
        self.prob = prob
        self.sigma = sigma

    def __call__(self, tensor):
        if random.random() < self.prob:
            noise = torch.randn_like(tensor) * self.sigma
            tensor = tensor + tensor * noise
            tensor = torch.clamp(tensor, 0.0, 1.0)
        return tensor

def get_transforms(is_eo=True):
    if is_eo:
        return transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.TrivialAugmentWide(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    else:
        return transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            AddSpeckleNoise(prob=0.5, sigma=0.1),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

# ========================
# 3. 数据集类
# ========================
class PairedDataset(torch.utils.data.Dataset):
    def __init__(self, eo_dataset, sar_dataset, eo_transform, sar_transform):
        self.eo_dataset = eo_dataset
        self.sar_dataset = sar_dataset
        self.eo_transform = eo_transform
        self.sar_transform = sar_transform
        assert len(self.eo_dataset) == len(self.sar_dataset)
        self.labels = [self.eo_dataset[idx][1] for idx in range(len(self.eo_dataset))]
        logger.info(f"PairedDataset初始化完成，样本数：{len(self.eo_dataset)}")

    def __len__(self):
        return len(self.eo_dataset)

    def __getitem__(self, idx):
        eo_img_pil, eo_label = self.eo_dataset[idx]
        sar_img_pil, sar_label = self.sar_dataset[idx]
        assert eo_label == sar_label
        
        if self.eo_transform:
            eo_img = self.eo_transform(eo_img_pil)
        if self.sar_transform:
            sar_img = self.sar_transform(sar_img_pil)
        
        return (eo_img, eo_label), (sar_img, sar_label)
    
    def get_label(self, idx):
        return self.labels[idx]

class SAROnlyDataset(torch.utils.data.Dataset):
    def __init__(self, paired_dataset):
        self.paired_dataset = paired_dataset

    def __len__(self):
        return len(self.paired_dataset)

    def __getitem__(self, idx):
        _, (sar_img, sar_label) = self.paired_dataset[idx]
        return sar_img, sar_label

# ========================
# 4. 数据加载函数
# ========================
def sample_dataset_by_ratio(dataset, class_counts, minor_class_indices, sample_total=SAMPLE_TOTAL, minor_max_ratio=0.10):
    """
    优化：
    1. 小类受控保留（而非全量），限制小类占比上限
    2. 大类分层采样：保证每个大类至少采样到指定数量样本，避免极端不平衡
    minor_max_ratio: 小类在最终采样数据中的最大占比（比如0.05=5%）
    major_min_per_cls: 每个大类至少采样的样本数（可根据需求调整）
    """
    major_min_per_cls = max(3000, sample_total // len(MAJOR_CLASS_INDICES) // 10)
    logger.info(f"\n📌 开始样本采样（总目标：{sample_total}）...")
    logger.info(f"   大类分层采样配置：每个大类至少采样{major_min_per_cls}个样本")
    all_indices = np.arange(len(dataset))
    
    # 快速获取标签（用缓存的labels，不执行transform）
    logger.info("   步骤1：加载标签（已缓存，无需处理图像）...")
    class_labels = np.array(dataset.labels)
    logger.info(f"   ✅ 标签加载完成，总样本数：{len(class_labels)}")
    
    minor_mask = np.isin(class_labels, MINOR_CLASS_INDICES)
    minor_indices = all_indices[minor_mask]
    minor_total_original = len(minor_indices)
    minor_max_num = int(sample_total * minor_max_ratio)
    minor_keep_num = min(minor_total_original, minor_max_num)
    if minor_keep_num < minor_total_original:
        minor_indices = np.random.choice(minor_indices, size=minor_keep_num, replace=False)
    minor_total = len(minor_indices)
    logger.info(f"   ✅ 小类样本：原始{minor_total_original} → 保留{minor_total}（占比上限{minor_max_ratio*100}%）")
    
    major_sample_num = max(0, sample_total - minor_total)
    major_mask = np.isin(class_labels, MAJOR_CLASS_INDICES)
    major_indices = all_indices[major_mask]
    major_total = len(major_indices)
    
    sampled_major_indices = []
    for cls_idx in MAJOR_CLASS_INDICES:
        cls_mask = (class_labels == cls_idx) & major_mask
        cls_indices = all_indices[cls_mask]
        cls_total = len(cls_indices)
        
        if cls_total == 0:
            logger.warning(f"   ⚠️ 大类{cls_idx}({RAW_ID_TO_NAME[cls_idx]})无样本，跳过该类分层采样")
            continue
        
        cls_sample_num = major_min_per_cls
        replace = cls_sample_num > cls_total
        if replace:
            logger.warning(f"   ⚠️ 大类{cls_idx}({RAW_ID_TO_NAME[cls_idx]})样本不足（{cls_total} < {cls_sample_num}），允许重复采样")
        
        # 采样当前大类的基础样本
        cls_sampled = np.random.choice(cls_indices, size=cls_sample_num, replace=replace)
        sampled_major_indices.extend(cls_sampled)
        logger.info(f"   📌 大类{cls_idx}({RAW_ID_TO_NAME[cls_idx]})：基础采样{cls_sample_num}个（原始{cls_total}个）")
    
    remaining_major_num = major_sample_num - len(sampled_major_indices)
    if remaining_major_num > 0:
        major_class_labels = class_labels[major_mask]
        major_class_sample_count = np.array([len(np.where(major_class_labels == t)[0]) for t in MAJOR_CLASS_INDICES])
        major_weight = 1. / major_class_sample_count
        major_samples_weight = np.array([major_weight[MAJOR_CLASS_INDICES.index(t)] for t in major_class_labels])
        major_samples_weight = torch.from_numpy(major_samples_weight)
        
        remaining_sampler = WeightedRandomSampler(
            major_samples_weight.type('torch.DoubleTensor'),
            remaining_major_num,
            replacement=True
        )
        remaining_indices = [major_indices[i] for i in remaining_sampler]
        sampled_major_indices.extend(remaining_indices)
        logger.info(f"   ✅ 大类剩余采样：{remaining_major_num}个（加权随机采样）")
    elif remaining_major_num < 0:
        # 若基础采样数超过需要，随机截断
        sampled_major_indices = sampled_major_indices[:major_sample_num]
        logger.info(f"   ✅ 大类基础采样数超额，截断至{major_sample_num}个")
    
    # 转换为numpy数组
    sampled_major_indices = np.array(sampled_major_indices)
    
    if len(sampled_major_indices) < major_sample_num:
        need_more = major_sample_num - len(sampled_major_indices)
        extra_indices = np.random.choice(major_indices, size=need_more, replace=True)
        sampled_major_indices = np.concatenate([sampled_major_indices, extra_indices])
        logger.warning(f"   ⚠️ 大类分层采样后仍不足，额外补充{need_more}个随机样本")
    
    # 合并并打乱
    sampled_indices = np.concatenate([sampled_major_indices, minor_indices])
    np.random.shuffle(sampled_indices)
    
    # 输出采样统计
    final_major_num = len(sampled_major_indices)
    final_minor_num = len(minor_indices)
    logger.info(f"📤 采样完成：原始{len(dataset)} → 采样后{len(sampled_indices)}")
    logger.info(f"   大类采样：{final_major_num}（分层{major_sample_num - remaining_major_num} + 剩余{remaining_major_num if remaining_major_num>0 else 0}） | 小类保留：{final_minor_num}（占比{final_minor_num/len(sampled_indices)*100:.2f}%）")
    
    sampled_labels = [dataset.get_label(idx) for idx in sampled_indices]
    sampled_major_cls = set([l for l in sampled_labels if l in MAJOR_CLASS_INDICES])
    missing_major_cls = set(MAJOR_CLASS_INDICES) - sampled_major_cls
    if missing_major_cls:
        missing_names = [RAW_ID_TO_NAME[idx] for idx in missing_major_cls]
        logger.warning(f"   ⚠️ 以下大类未被采样到：{missing_major_cls}({missing_names})（原始数据可能无该类样本）")
    else:
        logger.info(f"   ✅ 所有大类均已采样到（共{len(MAJOR_CLASS_INDICES)}类）")
    
    # 获取采样后所有样本的标签，并统计每类数量
    sampled_labels = [dataset.get_label(idx) for idx in sampled_indices]  # 从采样索引获取标签
    unique_cls, cls_counts = np.unique(sampled_labels, return_counts=True)  # 统计唯一类别和数量
    cls_count_dict = dict(zip(unique_cls, cls_counts))  # 转为字典（类别索引: 数量）
    
    # 格式化打印每类详细信息
    logger.info("\n📊 第一次采样后每类样本数量统计：")
    # 打印表头（对齐格式）
    logger.info(f"{'类别类型':<6} | {'原始ID':<6} | {'类别名称':<25} | {'样本数':<8}")
    logger.info("-" * 65)
    for cls_idx in range(len(class_counts)):
        cls_name = RAW_ID_TO_NAME[cls_idx]
        count = cls_count_dict.get(cls_idx, 0)
        cls_type = "大类" if cls_idx in MAJOR_CLASS_INDICES else "小类"
        logger.info(f"{cls_type:<6} | {cls_idx:<6} | {cls_name:<25} | {count:<8}")
    
    total_sampled = len(sampled_indices)
    major_total = sum(cls_count_dict.get(idx, 0) for idx in MAJOR_CLASS_INDICES)
    minor_total = sum(cls_count_dict.get(idx, 0) for idx in MINOR_CLASS_INDICES)
    logger.info("-" * 65)
    logger.info(f"📝 采样汇总：总样本数={total_sampled:>8} | 大类总数={major_total:>8} | 小类总数={minor_total:>8}")

    return sampled_indices

def prepare_data_loaders(
        eo_path, sar_path, clean_indices_path=None, batch_size=32, test_size=0.1, num_workers=4,
        eo_transform=None, sar_transform=None):

    class EO_Dataset(torchvision.datasets.ImageFolder):
        def __getitem__(self, idx):
            img_pil, label = super().__getitem__(idx)
            return img_pil, label

    class SAR_Dataset(torchvision.datasets.ImageFolder):
        def __getitem__(self, idx):
            img_pil, label = super().__getitem__(idx)
            return img_pil, label

    train_data_EO = EO_Dataset(root=eo_path, transform=None)
    train_data_SAR = SAR_Dataset(root=sar_path, transform=None)
    logger.info(f"EO数据集初始化完成，样本数：{len(train_data_EO)}")
    logger.info(f"SAR数据集初始化完成，样本数：{len(train_data_SAR)}")
    
    logger.info(f"\n🔍 数据集类别映射验证：{train_data_EO.class_to_idx}")
    for cls_name, idx in train_data_EO.class_to_idx.items():
        if RAW_NAME_TO_ID.get(cls_name, -1) != idx:
            logger.error(f"数据集类别映射错误！{cls_name} 应映射到 {RAW_NAME_TO_ID[cls_name]}，但实际是 {idx}")
            sys.exit(1)
    logger.info("✅ 数据集类别映射与原始数据一致！")
    
    clean_targets = None
    if clean_indices_path and os.path.exists(clean_indices_path):
        clean_indices = np.load(clean_indices_path).tolist()
        train_data_EO = Subset(train_data_EO, clean_indices)
        train_data_SAR = Subset(train_data_SAR, clean_indices)
        logger.info(f"Using cleaned dataset with {len(clean_indices)} samples.")
        clean_targets = [train_data_EO.dataset.imgs[i][1] for i in clean_indices]
        targets = clean_targets
    else:
        targets = [train_data_EO.imgs[i][1] for i in range(len(train_data_EO))]

    # 初始化PairedDataset（会缓存标签）
    paired_dataset = PairedDataset(train_data_EO, train_data_SAR, eo_transform, sar_transform)
    indices = np.arange(len(targets))
    labeled_indices, unlabeled_indices = train_test_split(indices, test_size=test_size, stratify=targets)

    paired_dataset_labeled = Subset(paired_dataset, labeled_indices)
    paired_dataset_unlabeled = Subset(paired_dataset, unlabeled_indices)
    logger.info(f"训练集样本数：{len(paired_dataset_labeled)}，验证集样本数：{len(paired_dataset_unlabeled)}")
    
    train_loader_unlabeled = DataLoader(
        paired_dataset_unlabeled, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=False  # 关闭避免死锁
    )

    y_train_EO = [targets[i] for i in labeled_indices]
    class_sample_count = np.array([len(np.where(np.array(y_train_EO) == t)[0]) for t in np.unique(y_train_EO)])
    weight = 1. / class_sample_count
    samples_weight = np.array([weight[t] for t in y_train_EO])
    samples_weight = torch.from_numpy(samples_weight)

    sampler = WeightedRandomSampler(samples_weight.type('torch.DoubleTensor'), len(samples_weight), replacement=True)
    train_loader = DataLoader(
        paired_dataset_labeled, 
        batch_size=32, 
        sampler=sampler, 
        num_workers=min(os.cpu_count(), 8),  # 降低num_workers
        pin_memory=True,
        persistent_workers=False,  # 关闭避免死锁
        prefetch_factor=2
    )
    train_dataset_size_EO = len(train_data_EO)
    train_dataset_size_SAR = len(train_data_SAR)

    if clean_targets is not None:
        unique_classes = np.unique(clean_targets)
        cleaned_class_counts = np.zeros(NUM_CLASSES, dtype=int)
        for cls in unique_classes:
            cleaned_class_counts[cls] = len(np.where(np.array(clean_targets) == cls)[0])
        logger.info(f"\n🧹 清洗后数据的类别样本数：")
        logger.info(f"{'原始ID':<6} | {'类别名称':<25} | {'样本数':<8}")
        logger.info("-" * 45)
        for cls_idx in range(NUM_CLASSES):
            logger.info(f"{cls_idx:<6} | {RAW_ID_TO_NAME[cls_idx]:<25} | {cleaned_class_counts[cls_idx]:<8}")
    else:
        unique_classes = np.unique(targets)
        cleaned_class_counts = np.zeros(NUM_CLASSES, dtype=int)
        for cls in unique_classes:
            cleaned_class_counts[cls] = len(np.where(np.array(targets) == cls)[0])
        logger.info(f"\n📊 原始数据（清洗前）的类别样本数：")
        logger.info(f"{'原始ID':<6} | {'类别名称':<25} | {'样本数':<8}")
        logger.info("-" * 45)
        for cls_idx in range(NUM_CLASSES):
            logger.info(f"{cls_idx:<6} | {RAW_ID_TO_NAME[cls_idx]:<25} | {cleaned_class_counts[cls_idx]:<8}")
    
    return train_loader, train_loader_unlabeled, train_dataset_size_EO, train_dataset_size_SAR, cleaned_class_counts

# ========================
# 5. 大类/小类划分
# ========================
def split_major_minor_classes(class_counts, class_names, threshold=0.03):
    """改用硬编码映射，不再动态计算大类/小类"""
    # 计算类别占比（仅用于日志，不影响映射）
    total_samples = np.sum(class_counts)
    cls_ratio = {idx: cnt/total_samples for idx, cnt in enumerate(class_counts)}
    
    major_classes = [RAW_ID_TO_NAME[idx] for idx in MAJOR_CLASS_INDICES]
    minor_classes = [RAW_ID_TO_NAME[idx] for idx in MINOR_CLASS_INDICES]
    
    logger.info(f"✅ 大类划分完成（使用硬编码，阈值{threshold*100}%仅用于参考）：")
    logger.info(f"   大类: {major_classes} (共{len(major_classes)}类) | 小类: {minor_classes} (共{len(minor_classes)}类)")
    logger.info(f"   大类模型输出维度：{len(major_classes)+1} (大类数+小类合并类)")
    
    logger.info("\n🔍 数据类别统计与硬编码映射验证（原始数据）：")
    logger.info(f"{'原始ID':<6} | {'类别名称':<25} | {'样本数':<8} | {'占比':<8} | {'类别类型'}")
    logger.info("-" * 70)
    for idx in range(len(class_counts)):
        cls_name = RAW_ID_TO_NAME[idx]
        cls_type = "大类" if idx in MAJOR_CLASS_INDICES else "小类"
        logger.info(f"   {idx:<6} | {cls_name:<25} | {class_counts[idx]:<8} | {cls_ratio.get(idx, 0):.4f} | {cls_type}")
    
    return (major_classes, minor_classes, LABEL2MAJOR, MINOR_UNIFIED_LABEL,
            MAJOR_CLASS_INDICES, MINOR_CLASS_INDICES)

# ========================
# 6. 数据加载器封装
# ========================
def prepare_double_classification_dataloaders(
    eo_path, sar_path, batch_size=32,
    val_ratio=0.1, num_workers=4, minor_threshold=0.03
):
    eo_transform = get_transforms(is_eo=True)
    sar_transform = get_transforms(is_eo=False)
    
    train_loader_base, train_loader_unlabeled, _, _, class_counts = prepare_data_loaders(
        eo_path=eo_path,
        sar_path=sar_path,
        clean_indices_path=None,
        batch_size=batch_size,
        test_size=val_ratio,
        num_workers=num_workers,
        eo_transform=eo_transform,
        sar_transform=sar_transform
    )
    
    (major_classes, minor_classes, label2major, minor_unified_label,
     major_class_indices, minor_class_indices) = split_major_minor_classes(
        class_counts, OFFICIAL_CLASS_NAMES, threshold=minor_threshold
    )
    
    # 采样训练集
    major_dataset = train_loader_base.dataset
    sampled_indices = sample_dataset_by_ratio(
        dataset=major_dataset.dataset,
        class_counts=class_counts,
        minor_class_indices=minor_class_indices,
        sample_total=SAMPLE_TOTAL
    )
    major_dataset_sampled = Subset(major_dataset.dataset, sampled_indices)
    logger.info(f"采样后数据集样本数：{len(major_dataset_sampled)}")
    
    # 重新构建训练加载器
    y_train_sampled = [major_dataset.dataset.get_label(idx) for idx in sampled_indices]
    class_sample_count = np.array([len(np.where(np.array(y_train_sampled) == t)[0]) for t in np.unique(y_train_sampled)])
    weight = 1. / class_sample_count
    samples_weight = np.array([weight[t] for t in y_train_sampled])
    samples_weight = torch.from_numpy(samples_weight)
    sampler = WeightedRandomSampler(samples_weight.type('torch.DoubleTensor'), len(samples_weight), replacement=True)
    
    train_loader_sampled = DataLoader(
        major_dataset_sampled,
        batch_size=24,
        sampler=sampler,
        num_workers=min(os.cpu_count(), 8),
        pin_memory=True,
        persistent_workers=False,
        prefetch_factor=2
    )
    
    # 大类模型数据
    major_val_loader = train_loader_unlabeled
    
    minor_train_loader = None
    minor_val_loader = None
    if len(minor_classes) > 0:
        class MinorSubset(Subset):
            def __init__(self, dataset, original_indices, minor_class_indices):
                super().__init__(dataset, range(len(dataset)))
                self.minor_class_indices = MINOR_CLASS_INDICES
                self.original_indices = original_indices
                self.minor_indices = []
                
                for inner_idx, raw_idx in enumerate(self.original_indices):
                    label = self.dataset.dataset.get_label(raw_idx)
                    if label in self.minor_class_indices:
                        self.minor_indices.append(inner_idx)
                
                logger.info(f"MinorSubset初始化完成：")
                logger.info(f"   采样后总样本数：{len(self.original_indices)}")
                logger.info(f"   筛选出小类样本数：{len(self.minor_indices)}")
            
            def __len__(self):
                return len(self.minor_indices)
            
            def __getitem__(self, idx):
                inner_idx = self.minor_indices[idx]
                return self.dataset[inner_idx]
        
        # 小类训练集
        minor_train_subset = MinorSubset(
            dataset=major_dataset_sampled,
            original_indices=sampled_indices,
            minor_class_indices=minor_class_indices
        )
        
        if len(minor_train_subset) > 0:
            minor_train_sar = SAROnlyDataset(minor_train_subset)
            minor_train_loader = DataLoader(
                minor_train_sar,
                batch_size=batch_size//2,
                shuffle=True,
                num_workers=min(os.cpu_count(), 8),
                pin_memory=True,
                persistent_workers=False,
                prefetch_factor=2
            )
        
        # 小类验证集
        val_original_indices = major_val_loader.dataset.indices
        minor_val_subset = MinorSubset(
            dataset=major_val_loader.dataset,
            original_indices=val_original_indices,
            minor_class_indices=minor_class_indices
        )
        
        if len(minor_val_subset) > 0:
            minor_val_sar = SAROnlyDataset(minor_val_subset)
            minor_val_loader = DataLoader(
                minor_val_sar,
                batch_size=batch_size,
                shuffle=False,
                num_workers=min(os.cpu_count(), 8),
                pin_memory=True,
                persistent_workers=False,
                prefetch_factor=2
            )
    
    num_major_output = len(MAJOR_CLASS_INDICES) + 1
    alpha_t = torch.ones(num_major_output, dtype=torch.float32) / num_major_output
    
    return {
        "major": {
            "train_loader": train_loader_sampled,
            "val_loader": major_val_loader,
            "alpha": alpha_t,
            "num_classes": num_major_output,
            "label_map": LABEL2MAJOR,
            "classes": major_classes,
            "minor_unified_label": MINOR_UNIFIED_LABEL,
            "class_indices": MAJOR_CLASS_INDICES,
            "ood_threshold": MAJOR_OOD_THRESHOLD
        },
        "minor": {
            "train_loader": minor_train_loader,
            "val_loader": minor_val_loader,
            "num_classes": len(MINOR_CLASS_INDICES),
            "classes": minor_classes,
            "class_indices": MINOR_CLASS_INDICES,
            "ood_threshold": MINOR_OOD_THRESHOLD
        },
        "class_names": [RAW_ID_TO_NAME[idx] for idx in range(NUM_CLASSES)],  # 修正为原始数据的类别名
        "class_counts": class_counts
    }

# ========================
# 7. EMA工具类
# ========================
import copy
class EMA:
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.ema_model = copy.deepcopy(model).eval()
        for p in self.ema_model.parameters():
            p.requires_grad_(False)
        logger.info(f"EMA初始化完成，decay={decay}")

    def update(self):
        with torch.no_grad():
            for ema_p, p in zip(self.ema_model.parameters(), self.model.parameters()):
                ema_p.mul_(self.decay).add_(p, alpha=1 - self.decay)
            for ema_b, b in zip(self.ema_model.buffers(), self.model.buffers()):
                ema_b.copy_(b)

    def get_ema_model(self):
        return self.ema_model

# ========================
# 8. 验证函数
# ========================
def validate_major_model(model, val_loader, label2major, minor_unified_label, device, ood_threshold):
    model.eval()
    major_correct = 0
    binary_correct = 0
    total = 0
    minor_class_confidences = []
    
    with torch.no_grad():
        for (_, _), (img_sar, label_sar) in val_loader:
            img_sar, label_sar = img_sar.to(device), label_sar.to(device)
            out = model(img_sar)
            prob = F.softmax(out, dim=1)
            pred = prob.argmax(dim=1)
            
            major_labels = torch.tensor([LABEL2MAJOR[l.item()] for l in label_sar], device=device)
            major_correct += (pred == major_labels).sum().item()
            
            is_minor_gt = (major_labels == MINOR_UNIFIED_LABEL)
            is_minor_pred = (pred == MINOR_UNIFIED_LABEL)
            binary_correct += (is_minor_gt == is_minor_pred).sum().item()
            
            minor_mask = (pred == MINOR_UNIFIED_LABEL)
            if minor_mask.any():
                minor_confs = prob[minor_mask, MINOR_UNIFIED_LABEL].cpu().numpy()
                minor_class_confidences.extend(minor_confs.tolist())
            
            total += label_sar.size(0)
    
    major_acc = major_correct / total
    binary_acc = binary_correct / total
    
    if minor_class_confidences:
        avg_minor_conf = np.mean(minor_class_confidences)
        logger.info(f"📈 小类合并类平均置信度：{avg_minor_conf:.4f} | 阈值：{ood_threshold}")
    
    return {
        "major_class_acc": major_acc,
        "binary_acc": binary_acc,
        "avg_minor_conf": avg_minor_conf if minor_class_confidences else 0.0
    }

def validate_minor_model(model, val_loader, minor_class_indices, device, ood_threshold):
    if val_loader is None or len(minor_class_indices) == 0:
        return 0.0, 0.0
    
    model.eval()
    correct = 0
    total = 0
    minor_confidences = []
    minor_label_map = MINOR_LABEL_MAP
    
    with torch.no_grad():
        for img_sar, label_sar in val_loader:
            img_sar, label_sar = img_sar.to(device), label_sar.to(device)
            minor_labels = torch.tensor([minor_label_map[l.item()] for l in label_sar], device=device)
            
            out = model(img_sar)
            prob = F.softmax(out, dim=1)
            pred = prob.argmax(dim=1)
            
            correct += (pred == minor_labels).sum().item()
            pred_confs = prob[range(len(pred)), pred].cpu().numpy()
            minor_confidences.extend(pred_confs.tolist())
            
            total += label_sar.size(0)
    
    val_acc = correct / total
    avg_minor_conf = np.mean(minor_confidences) if minor_confidences else 0.0
    logger.info(f"📈 小类模型平均置信度：{avg_minor_conf:.4f} | 阈值：{ood_threshold}")
    
    return val_acc, avg_minor_conf

# ========================
# 9. 训练函数
# ========================
def train_major_model(data_dict, device):
    major_data = data_dict["major"]
    logger.info(f"开始训练大类模型，输出类别数：{major_data['num_classes']}")
    
    model_Major_EO = FeatureExtractor('resnet101', num_classes=major_data["num_classes"], dropout=0.6).to(device)
    model_Major_SAR = FeatureExtractor('resnet101', num_classes=major_data["num_classes"], dropout=0.6).to(device)
    logger.info("大类模型（EO/SAR）初始化完成")
    
    def get_optimizer(model):
        return optim.AdamW([
            {'params': model.conv1.parameters(), 'lr': 5e-5},
            {'params': model.bn1.parameters(), 'lr': 5e-5},
            {'params': model.layer1.parameters(), 'lr': 5e-5},
            {'params': model.layer2.parameters(), 'lr': 1e-4},
            {'params': model.layer3.parameters(), 'lr': 1e-4},
            {'params': model.layer4.parameters(), 'lr': 1e-4},
            {'params': model.classifier.parameters(), 'lr': 1e-3}
        ], weight_decay=1e-4)
    
    optim_EO = get_optimizer(model_Major_EO)
    optim_SAR = get_optimizer(model_Major_SAR)
    
    scheduler_EO = CosineAnnealingLR(optim_EO, T_max=30, eta_min=1e-6)
    scheduler_SAR = CosineAnnealingLR(optim_SAR, T_max=30, eta_min=1e-6)
    
    criterion_ce = FocalLoss(alpha=major_data["alpha"].to(device), gamma=2)
    criterion_da = da_loss(num_projections=128)
    
    ema_EO = EMA(model_Major_EO, decay=0.999)
    ema_SAR = EMA(model_Major_SAR, decay=0.999)
    
    save_dir_major_eo = BASE_DIR / 'resnet_Major_EO'
    save_dir_major_sar = BASE_DIR / 'resnet_Major_SAR'
    save_dir_major_eo.mkdir(exist_ok=True)
    save_dir_major_sar.mkdir(exist_ok=True)
    logger.info(f"大类模型保存路径：EO={save_dir_major_eo}, SAR={save_dir_major_sar}")
    
    best_binary_acc = 0.0
    label2major = LABEL2MAJOR
    
    logger.info("\n🚀 开始训练大类模型（ResNet101）...")
    for epoch in tqdm(range(40), desc="Epoch Progress", unit="epoch"):
        model_Major_EO.train()
        model_Major_SAR.train()
        train_loss = correct_EO = correct_SAR = total = 0
        
        progress_bar = tqdm(major_data["train_loader"], desc=f"Epoch {epoch+1}", unit="batch", leave=False)
        for batch in progress_bar:
            (img_eo, label_eo), (img_sar, label_sar) = batch
            img_eo, label_eo = img_eo.to(device), label_eo.to(device)
            img_sar, label_sar = img_sar.to(device), label_sar.to(device)
            
            major_label_eo = torch.tensor([LABEL2MAJOR[l.item()] for l in label_eo], device=device)
            major_label_sar = torch.tensor([LABEL2MAJOR[l.item()] for l in label_sar], device=device)
            
            out_eo = model_Major_EO(img_eo)
            out_sar = model_Major_SAR(img_sar)
            
            loss_ce_eo = criterion_ce(out_eo, major_label_eo)
            loss_ce_sar = criterion_ce(out_sar, major_label_sar)
            
            loss_da = 0
            target_keys = ['layer2', 'layer3', 'layer4']
            for key in target_keys:
                if key not in model_Major_EO.activations or key not in model_Major_SAR.activations:
                    logger.warning(f"⚠️ 激活层{key}不存在，跳过该层的域自适应损失")
                    continue
                f_eo = F.adaptive_avg_pool2d(model_Major_EO.activations[key], (1,1)).flatten(1)
                f_sar = F.adaptive_avg_pool2d(model_Major_SAR.activations[key], (1,1)).flatten(1)
                loss_da += criterion_da(f_eo, f_sar)
            
            loss_total = loss_ce_eo + loss_ce_sar + 0.1 * loss_da
            
            optim_EO.zero_grad()
            optim_SAR.zero_grad()
            loss_total.backward()
            torch.nn.utils.clip_grad_norm_(model_Major_EO.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(model_Major_SAR.parameters(), 1.0)
            optim_EO.step()
            optim_SAR.step()
            ema_EO.update()
            ema_SAR.update()
            
            pred_eo = out_eo.argmax(dim=1)
            pred_sar = out_sar.argmax(dim=1)
            correct_EO += (pred_eo == major_label_eo).sum().item()
            correct_SAR += (pred_sar == major_label_sar).sum().item()
            total += label_eo.size(0)
            train_loss += loss_total.item()
            progress_bar.set_postfix(loss=loss_total.item())
        
        scheduler_EO.step()
        scheduler_SAR.step()
        
        val_metrics = validate_major_model(
            ema_SAR.get_ema_model(), major_data["val_loader"],
            label2major, MINOR_UNIFIED_LABEL, device, major_data["ood_threshold"]
        )
        train_acc_eo = correct_EO / total
        train_acc_sar = correct_SAR / total
        
        logger.info(f'Epoch {epoch+1} | Train Loss: {train_loss/len(major_data["train_loader"]):.4f} | '
                    f'Train Acc EO: {train_acc_eo:.4f} | SAR: {train_acc_sar:.4f} | '
                    f'Val Major Acc: {val_metrics["major_class_acc"]:.4f} | Binary Acc: {val_metrics["binary_acc"]:.4f}')
        
        if val_metrics["binary_acc"] > best_binary_acc:
            best_binary_acc = val_metrics["binary_acc"]
            torch.save(ema_EO.get_ema_model().state_dict(), save_dir_major_eo / 'best_ema.pth')
            torch.save(ema_SAR.get_ema_model().state_dict(), save_dir_major_sar / 'best_ema.pth')
            torch.save(model_Major_EO.state_dict(), save_dir_major_eo / 'best_raw.pth')
            torch.save(model_Major_SAR.state_dict(), save_dir_major_sar / 'best_raw.pth')
            logger.info(f"✅ 新的最佳大类模型保存！Binary Acc: {best_binary_acc:.4f}")
    
    torch.save(model_Major_EO.state_dict(), BASE_DIR / 'Major_EO_final.pth')
    torch.save(model_Major_SAR.state_dict(), BASE_DIR / 'Major_SAR_final.pth')
    logger.info("大类模型训练完成，最终模型已保存")
    
    return {
        "eo_model": model_Major_EO,
        "sar_model": model_Major_SAR,
        "ema_eo_model": ema_EO.get_ema_model(),
        "ema_sar_model": ema_SAR.get_ema_model()
    }

def train_minor_model(major_sar_model, data_dict, device):
    minor_data = data_dict["minor"]
    minor_class_indices = MINOR_CLASS_INDICES
    
    if minor_data["num_classes"] == 0 or minor_data["train_loader"] is None:
        logger.warning("\n⚠️ 无小类数据，跳过小类模型训练")
        return None
    
    logger.info(f"\n🚀 开始训练小类模型（ResNet101骨干+解冻layer4）...")
    logger.info(f"   小类数量：{minor_data['num_classes']} | 小类列表：{minor_data['classes']}")
    
    model_Minor_SAR = FeatureExtractor('resnet101', num_classes=minor_data["num_classes"], dropout=0.6).to(device)
    
    resnet_core_layers = ['conv1', 'bn1', 'layer1', 'layer2', 'layer3', 'layer4']
    for layer_name in resnet_core_layers:
        major_layer = getattr(major_sar_model, layer_name)
        minor_layer = getattr(model_Minor_SAR, layer_name)
        minor_layer.load_state_dict(major_layer.state_dict())
    logger.info("✅ 小类模型加载大类模型骨干权重完成")
    
    for name, param in model_Minor_SAR.named_parameters():
        param.requires_grad = False
        if any(layer in name for layer in UNFREEZE_LAYERS):
            param.requires_grad = True
            logger.info(f"🔓 解冻骨干层：{name}")
    
    for param in model_Minor_SAR.classifier.parameters():
        param.requires_grad = True
    
    backbone_params = []
    for name, p in model_Minor_SAR.named_parameters():
        if p.requires_grad and any(layer in name for layer in UNFREEZE_LAYERS):
            backbone_params.append(p)
    classifier_params = list(model_Minor_SAR.classifier.parameters())
    
    optimizer_params = [
        {'params': backbone_params, 'lr': 5e-5},
        {'params': classifier_params, 'lr': 1e-4}
    ]
    optim_Minor = optim.AdamW(optimizer_params, weight_decay=1e-4)
    scheduler_Minor = CosineAnnealingLR(optim_Minor, T_max=MINOR_TRAIN_EPOCHS, eta_min=1e-6)
    
    alpha_minor = torch.ones(len(MINOR_CLASS_INDICES), dtype=torch.float32) / len(MINOR_CLASS_INDICES)
    criterion_ce = FocalLoss(alpha=alpha_minor.to(device), gamma=2)
    
    ema_Minor = EMA(model_Minor_SAR, decay=0.999)
    
    save_dir_minor_sar = BASE_DIR / 'resnet_Minor_SAR'
    save_dir_minor_sar.mkdir(exist_ok=True)
    logger.info(f"小类模型保存路径：{save_dir_minor_sar}")
    
    best_val_acc = 0.0
    minor_label_map = MINOR_LABEL_MAP
    
    for epoch in tqdm(range(MINOR_TRAIN_EPOCHS), desc="Minor Epoch Progress", unit="epoch"):
        model_Minor_SAR.train()
        train_loss = correct = total = 0
        
        progress_bar = tqdm(minor_data["train_loader"], desc=f"Minor Epoch {epoch+1}", unit="batch", leave=False)
        for batch in progress_bar:
            img_sar, label_sar = batch
            img_sar, label_sar = img_sar.to(device), label_sar.to(device)
            
            minor_labels = torch.tensor([minor_label_map[l.item()] for l in label_sar], device=device)
            
            out_sar = model_Minor_SAR(img_sar)
            loss_ce = criterion_ce(out_sar, minor_labels)
            
            optim_Minor.zero_grad()
            loss_ce.backward()
            optim_Minor.step()
            ema_Minor.update()
            
            pred_sar = out_sar.argmax(dim=1)
            correct += (pred_sar == minor_labels).sum().item()
            total += label_sar.size(0)
            train_loss += loss_ce.item()
            progress_bar.set_postfix(loss=loss_ce.item())
        
        scheduler_Minor.step()
        
        val_acc, avg_minor_conf = validate_minor_model(
            ema_Minor.get_ema_model(), minor_data["val_loader"],
            minor_class_indices, device, minor_data["ood_threshold"]
        )
        train_acc = correct / total
        
        logger.info(f'Minor Epoch {epoch+1} | Train Loss: {train_loss/len(minor_data["train_loader"]):.4f} | '
                    f'Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}')
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(ema_Minor.get_ema_model().state_dict(), save_dir_minor_sar / 'best_ema.pth')
            torch.save(model_Minor_SAR.state_dict(), save_dir_minor_sar / 'best_raw.pth')
            logger.info(f"✅ 新的最佳小类模型保存！Val Acc: {best_val_acc:.4f}")
    
    torch.save(model_Minor_SAR.state_dict(), BASE_DIR / 'Minor_SAR_final.pth')
    logger.info("小类模型训练完成，最终模型已保存")
    return model_Minor_SAR

# ========================
# 10. 主函数
# ========================
if __name__ == "__main__":
    try:
        EO_path = BASE_DIR.parent / 'train/EO_Train'
        SAR_path = BASE_DIR.parent / 'train/SAR_Train'
        logger.info(f"数据路径：EO={EO_path}, SAR={SAR_path}")

        if not EO_path.exists():
            logger.error(f"EO数据路径不存在：{EO_path}")
            sys.exit(1)
        if not SAR_path.exists():
            logger.error(f"SAR数据路径不存在：{SAR_path}")
            sys.exit(1)
        
        logger.info("🔍 开始准备数据加载器...")
        data_dict = prepare_double_classification_dataloaders(
            eo_path=EO_path,
            sar_path=SAR_path,
            batch_size=48,
            val_ratio=0.1,
            num_workers=16,
            minor_threshold=MINOR_THRESHOLD
        )
        
        if LOAD_PRETRAINED_MAJOR:
            logger.info("\n📌 模式：加载预训练大类模型 → 仅训练小类模型")
            major_sar_model = load_pretrained_major_model(
                model_path=MAJOR_SAR_MODEL_PATH,
                num_classes=MAJOR_NUM_CLASSES,
                device=device
            )
            minor_model = train_minor_model(major_sar_model, data_dict, device)
        
        else:
            logger.info("\n📌 模式：从头训练（大类模型 → 小类模型）")
            major_models = train_major_model(data_dict, device)
            minor_model = train_minor_model(major_models["sar_model"], data_dict, device)
        
        logger.info("\n🎉 训练流程完成！")
        logger.info(f"   小类模型保存路径：{BASE_DIR / 'resnet_Minor_SAR'}")
        logger.info(f"   OOD判定阈值：大类{MAJOR_OOD_THRESHOLD} | 小类{MINOR_OOD_THRESHOLD}")
        
    except Exception as e:
        logger.error(f"训练过程中发生错误：{str(e)}", exc_info=True)
        raise