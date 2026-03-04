import torch
import torch.nn as nn
import torch.utils.data as data
from torchvision import transforms
from torch.utils.data import Dataset
import torch.nn.functional as F
import pandas as pd
import numpy as np
import os
import cv2
from pathlib import Path
from collections import Counter
from tqdm import tqdm
import random  # 提前导入，避免运行时缺失
from feature_extractor import FeatureExtractor
import logging

def setup_logging():
    log_format = '%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=[logging.StreamHandler()]
    )
    return logging.getLogger(__name__)

logger = setup_logging()

BASE_DIR = Path(__file__).resolve().parent
NUM_CLASSES = 10
device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
logger.info(f"使用设备：{device}")

OFFICIAL_CLASS_NAMES = [
    "sedan", "SUV", "pickup_truck", "van", "box_truck",
    "motorcycle", "flatbed_truck", "bus", "pickup_truck_w_trailer", "semi_w_trailer"
]

# ========================
# 1. 数据集类
# ========================
class ValDataset(Dataset):
    def __init__(self, img_folder, csv_path, transform=None):
        self.imgs_folder = Path(img_folder)
        self.transform = transform
        self.df = pd.read_csv(csv_path)
        self.df = self.df[self.df['OOD_flag'] == 0].reset_index(drop=True)
        self.image_ids = self.df['image_id'].tolist()
        self.labels = self.df['class'].tolist()
        self.name_to_id = {name: idx for idx, name in enumerate(OFFICIAL_CLASS_NAMES)}
        self.raw_labels = [self.name_to_id[name] for name in self.labels]
        logger.info(f"验证集加载完成：共{len(self.df)}个非OOD样本")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image_id_with_prefix = self.image_ids[idx]
        img_path = self.imgs_folder / f"{image_id_with_prefix}.png"
        img = cv2.imread(str(img_path))
        if img is None:
            logger.warning(f"无法读取图像：{img_path}")
            img = np.zeros((224, 224, 3), dtype=np.uint8)  # 兜底空图像
        if self.transform:
            img = self.transform(img)
        return img, self.raw_labels[idx]

val_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# ========================
# 2. 核心：双模型批量预测函数
# ========================
def predict_dual_model(
    major_model, minor_model, 
    label2major, minor_unified_label, minor_class_indices,
    major_ood_thresh=0.7, minor_ood_thresh=0.6,
    dataloader=None
):
    major_model.eval()
    if minor_model is not None:
        minor_model.eval()
    
    all_preds = []
    all_ood_flags = []
    all_labels = []

    with torch.no_grad():
        for imgs, labels in tqdm(dataloader, desc="双模型预测", leave=False):
            imgs = imgs.to(device)
            labels = labels.to(device)
            batch_size = imgs.shape[0]
            
            major_logits = major_model(imgs)
            major_probs = F.softmax(major_logits, dim=1)
            major_preds = torch.argmax(major_probs, dim=1)
            major_confs = major_probs[range(batch_size), major_preds]

            final_preds = -1 * torch.ones(batch_size, dtype=torch.int64, device=device)
            ood_flags = torch.ones(batch_size, dtype=torch.bool, device=device)
            
            non_minor_mask = (major_preds != minor_unified_label)
            if non_minor_mask.sum() > 0:
                major_pred_indices = major_preds[non_minor_mask].cpu().numpy()
                major2raw = {}
                for raw_idx, major_idx in label2major.items():
                    if major_idx != minor_unified_label:
                        major2raw[major_idx] = raw_idx

                non_minor_preds = [major2raw[idx] for idx in major_pred_indices]
                non_minor_preds_tensor = torch.tensor(non_minor_preds, dtype=torch.int64, device=device)
                final_preds[non_minor_mask] = non_minor_preds_tensor
                
                valid_non_minor = non_minor_mask & (major_confs >= major_ood_thresh)
                ood_flags[valid_non_minor] = False

            minor_mask = (major_preds == minor_unified_label)
            if minor_mask.sum() > 0 and minor_model is not None:
                minor_imgs = imgs[minor_mask]
                minor_logits = minor_model(minor_imgs)
                minor_probs = F.softmax(minor_logits, dim=1)
                minor_preds = torch.argmax(minor_probs, dim=1)
                minor_confs = minor_probs[range(len(minor_preds)), minor_preds]
                
                # 小类内部索引→原始类别ID
                minor_pred_raw = [minor_class_indices[idx] for idx in minor_preds.cpu().numpy()]
                valid_minor = minor_confs >= minor_ood_thresh
                
                minor_batch_indices = torch.nonzero(minor_mask, as_tuple=True)[0]
                valid_minor_batch_indices = minor_batch_indices[valid_minor]
                if len(valid_minor_batch_indices) > 0:
                    valid_minor_preds = [minor_pred_raw[i] for i in range(len(valid_minor)) if valid_minor[i]]
                    valid_minor_preds_tensor = torch.tensor(valid_minor_preds, dtype=torch.int64, device=device)
                    final_preds[valid_minor_batch_indices] = valid_minor_preds_tensor
                
                ood_flags[minor_batch_indices] = ~valid_minor

            all_preds.extend(final_preds.cpu().numpy())
            all_ood_flags.extend(ood_flags.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return np.array(all_preds), np.array(all_ood_flags), np.array(all_labels)

# ========================
# 3. 评估函数
# ========================
def evaluate_dual_model(preds, ood_flags, labels):
    """
    评估双模型性能：
    - preds：预测的原始类别ID（-1表示OOD）
    - ood_flags：是否判定为OOD
    - labels：真实原始类别ID
    返回：总体准确率、非OOD准确率、每类准确率
    """
    valid_mask = (ood_flags == False) & (preds != -1)
    if valid_mask.sum() == 0:
        overall_acc = 0.0
    else:
        overall_acc = (preds[valid_mask] == labels[valid_mask]).mean()
    
    ood_correct = (ood_flags == False).mean()
    
    per_class = {}
    for cls in range(NUM_CLASSES):
        cls_mask = (labels == cls)
        if cls_mask.sum() == 0:
            per_class[cls] = 0.0
            continue
        cls_valid = cls_mask & valid_mask
        if cls_valid.sum() == 0:
            per_class[cls] = 0.0
        else:
            per_class[cls] = (preds[cls_valid] == cls).mean()
    
    logger.info(f"评估完成：有效预测{valid_mask.sum()}/{len(labels)}个样本 | OOD判定准确率{ood_correct:.4f}")
    return {
        "overall_acc": overall_acc,
        "ood_correct_rate": ood_correct,
        "per_class_acc": per_class
    }

# ========================
# 4. 加载大类/小类标签映射
# ========================
def get_label_mappings(class_counts, minor_threshold=0.03):
    total_samples = np.sum(class_counts)
    cls_ratio = {idx: cnt/total_samples for idx, cnt in enumerate(class_counts)}
    
    major_class_indices = [idx for idx in range(len(class_counts)) if cls_ratio[idx] >= minor_threshold]
    minor_class_indices = [idx for idx in range(len(class_counts)) if cls_ratio[idx] < minor_threshold]
    
    label2major = {}
    for new_idx, old_idx in enumerate(major_class_indices):
        label2major[old_idx] = new_idx
    minor_unified_label = len(major_class_indices)
    for old_idx in minor_class_indices:
        label2major[old_idx] = minor_unified_label
    
    logger.info(f"标签映射重建完成：")
    logger.info(f"  大类索引：{major_class_indices} → 大类模型输出索引：{list(range(len(major_class_indices)))}")
    logger.info(f"  小类索引：{minor_class_indices} → 大类模型输出索引：{minor_unified_label}")
    return label2major, minor_unified_label, major_class_indices, minor_class_indices

# ========================
# 主函数
# ========================
def main():
    logger.info("🔍 开始加载双模型验证环境...")
    
    val_img_folder = BASE_DIR.parent / "val"
    val_csv_path = BASE_DIR.parent / "validation_reference.csv"
    val_dataset = ValDataset(val_img_folder, val_csv_path, transform=val_transform)
    val_loader = data.DataLoader(
        val_dataset, 
        batch_size=64, 
        shuffle=False, 
        num_workers=4,
        pin_memory=True,
        persistent_workers=False
    )

    TRAIN_CLASS_COUNTS = np.array([43401,2896,612,898,1441,24158,695,364291,353,16890])
    label2major, minor_unified_label, major_class_indices, minor_class_indices = get_label_mappings(
        class_counts=TRAIN_CLASS_COUNTS,
        minor_threshold=0.0
    )

    logger.info("🚀 加载大类模型（ResNet101 SAR）...")
    num_major_output = len(major_class_indices)
    major_model = FeatureExtractor('resnet101', num_classes=num_major_output, dropout=0.5)
    major_ckpt_path = BASE_DIR / "resnet_SAR/best_ema.pth"
    major_state = torch.load(major_ckpt_path, map_location=device)
    major_model.load_state_dict(major_state)
    major_model.to(device)

    minor_model = None
    if len(minor_class_indices) > 0:
        logger.info(f"🚀 加载小类模型（ResNet101 SAR）...")
        minor_model = FeatureExtractor('resnet101', num_classes=len(minor_class_indices), dropout=0.5)
        minor_ckpt_path = BASE_DIR / "resnet_Minor_SAR/best_ema.pth"
        minor_state = torch.load(minor_ckpt_path, map_location=device)
        minor_model.load_state_dict(minor_state)
        minor_model.to(device)
    else:
        logger.warning("⚠️ 无小类数据，仅加载大类模型")

    logger.info("\n🧪 开始双模型验证预测...")
    preds, ood_flags, labels = predict_dual_model(
        major_model=major_model,
        minor_model=minor_model,
        label2major=label2major,
        minor_unified_label=minor_unified_label,
        minor_class_indices=minor_class_indices,
        major_ood_thresh=0,
        minor_ood_thresh=0,
        dataloader=val_loader
    )

    eval_results = evaluate_dual_model(preds, ood_flags, labels)

    logger.info("\n" + "="*80)
    logger.info("🏆 双模型验证结果")
    logger.info("="*80)
    logger.info(f"总体准确率（非OOD且预测正确）：{eval_results['overall_acc']:.4f}")
    logger.info(f"OOD判定准确率（验证集应全为非OOD）：{eval_results['ood_correct_rate']:.4f}")
    logger.info("\n📊 每类准确率：")
    for cls_idx in range(NUM_CLASSES):
        cls_name = OFFICIAL_CLASS_NAMES[cls_idx]
        cls_acc = eval_results['per_class_acc'][cls_idx]
        logger.info(f"  类别{cls_idx:2d} ({cls_name:>25}): {cls_acc:.4f}")
    logger.info("="*80)

if __name__ == "__main__":
    main()