import re
import time
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import Dataset
from tqdm import tqdm
import numpy as np
import pandas as pd
import os
import cv2
import logging
from pathlib import Path
from collections import Counter
from feature_extractor import FeatureExtractor

# ========================
# 1. 日志配置
# ========================
def setup_logging():
    log_format = '%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=[logging.StreamHandler()]
    )
    return logging.getLogger(__name__)

logger = setup_logging()

# ========================
# 2. 基础配置
# ========================
BASE_DIR = Path(__file__).resolve().parent
NUM_CLASSES = 10
device_ids = [0]
device = torch.device(f'cuda:{device_ids[0]}' if torch.cuda.is_available() else 'cpu')
logger.info(f"使用设备：{device}")

OFFICIAL_CLASS_NAMES = [
    "sedan",                      # 0
    "SUV",                        # 1
    "pickup_truck",               # 2
    "van",                        # 3
    "box_truck",                  # 4
    "motorcycle",                 # 5
    "flatbed_truck",              # 6
    "bus",                        # 7
    "pickup_truck_w_trailer",     # 8
    "semi_w_trailer"              # 9
]

FIRST_CODE_ID_TO_CURRENT_ID = {
    0: 1,
    1: 4,
    2: 7,
    3: 6,
    4: 5,
    5: 2,
    6: 8,
    7: 0,
    8: 9,
    9: 3
}

MINOR_OUTPUT_TO_RAW = {
    0: 1,
    1: 2,
    2: 3,
    3: 4,
    4: 6,
    5: 8
}

TRAIN_CLASS_COUNTS = np.array([
    43401,  # 0:SUV
    2896,   # 1:box_truck
    612,    # 2:bus
    898,    # 3:flatbed_truck
    1441,   # 4:motorcycle
    24158,  # 5:pickup_truck
    696,    # 6:pickup_truck_w_trailer
    364291, # 7:sedan
    353,    # 8:semi_w_trailer
    16890   # 9:van
])
major_class_indices = [0,5,7,9]
minor_class_indices = [1,2,3,4,6,8]
minor_unified_label = 4
label2major = {
    0:0, 5:1, 7:2, 9:3,
    1:4, 2:4, 3:4, 4:4, 6:4, 8:4
}

# ========================
# 3. 数据集类
# ========================
class InfDataset(Dataset):
    def __init__(self, img_folder, transform=None):
        self.imgs_folder = Path(img_folder)
        self.transform = transform
        self.img_paths = []
        img_list = sorted(os.listdir(self.imgs_folder))
        self.img_nums = len(img_list)
        for img_name in img_list:
            self.img_paths.append(str(self.imgs_folder / img_name))

    def __getitem__(self, idx):
        img = cv2.imread(self.img_paths[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.transform:
            img = self.transform(img)
        name = os.path.basename(self.img_paths[idx])
        match = re.search(r'\d+', name)
        if match:
            image_id = match.group()
        else:
            raise ValueError(f"无法从文件名 {name} 提取 image_id")
        return img, image_id

    def __len__(self):
        return self.img_nums

class ValDataset(Dataset):
    def __init__(self, img_folder, csv_path, transform=None):
        self.imgs_folder = Path(img_folder)
        self.transform = transform
        self.df = pd.read_csv(csv_path)
        self.df = self.df[self.df['OOD_flag'] == 0].reset_index(drop=True)
        self.image_ids = self.df['image_id'].tolist()
        self.labels = self.df['class'].tolist()

        self.csv_name_to_current_id = {name: idx for idx, name in enumerate(OFFICIAL_CLASS_NAMES)}
        self.current_id_to_first_code_id = {v: k for k, v in FIRST_CODE_ID_TO_CURRENT_ID.items()}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image_id_with_prefix = self.image_ids[idx]
        img_path = self.imgs_folder / f"{image_id_with_prefix}.png"
        if not img_path.exists():
            raise FileNotFoundError(f"Image not found: {img_path}")
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.transform:
            img = self.transform(img)
        
        label_name = self.labels[idx]
        current_label_id = self.csv_name_to_current_id[label_name]
        first_code_label_id = self.current_id_to_first_code_id[current_label_id]
        
        return img, first_code_label_id, current_label_id, image_id_with_prefix

# ========================
# 4. 数据增强
# ========================
inf_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# ========================
# 5. 双模型预测核心函数
# ========================
def predict_dual_model_for_test(
    major_model, minor_model, 
    label2major, minor_unified_label,
    major_ood_thresh=0, minor_ood_thresh=0,
    dataloader=None
):
    major_model.eval()
    if minor_model is not None:
        minor_model.eval()
    
    all_preds_first_code = []  
    all_logits = []           
    all_image_ids = []

    with torch.no_grad():
        for batch_idx, (imgs, image_ids) in tqdm(enumerate(dataloader), desc="双模型预测"):
            imgs = imgs.to(device)
            batch_size = imgs.shape[0]
            
            # 大类模型预测
            major_logits = major_model(imgs)
            major_probs = F.softmax(major_logits, dim=1)
            major_preds = torch.argmax(major_probs, dim=1)
            major_confs = major_probs[range(batch_size), major_preds]
            final_preds = -1 * torch.ones(batch_size, dtype=torch.int64, device=device)
            
            # 非小类合并类
            non_minor_mask = (major_preds != minor_unified_label)
            if non_minor_mask.sum() > 0:
                major2raw = {0:0, 1:5, 2:7, 3:9}
                major_pred_indices = major_preds[non_minor_mask].cpu().numpy()
                non_minor_preds = [major2raw[idx] for idx in major_pred_indices]
                non_minor_preds_tensor = torch.tensor(non_minor_preds, dtype=torch.int64, device=device)
                final_preds[non_minor_mask] = non_minor_preds_tensor
                valid_non_minor = non_minor_mask & (major_confs >= major_ood_thresh)
                final_preds[~valid_non_minor] = -1

            # 小类合并类
            minor_mask = (major_preds == minor_unified_label)
            if minor_mask.sum() > 0 and minor_model is not None:
                minor_imgs = imgs[minor_mask]
                minor_logits = minor_model(minor_imgs)
                minor_probs = F.softmax(minor_logits, dim=1)
                minor_preds = torch.argmax(minor_probs, dim=1)
                minor_confs = minor_probs[range(len(minor_preds)), minor_preds]
                
                minor_pred_raw = [MINOR_OUTPUT_TO_RAW[idx] for idx in minor_preds.cpu().numpy()]
                minor_batch_indices = torch.nonzero(minor_mask, as_tuple=True)[0]
                valid_minor = minor_confs >= minor_ood_thresh
                valid_minor_batch_indices = minor_batch_indices[valid_minor]
                if len(valid_minor_batch_indices) > 0:
                    valid_minor_preds = [minor_pred_raw[i] for i in range(len(valid_minor)) if valid_minor[i]]
                    valid_minor_preds_tensor = torch.tensor(valid_minor_preds, dtype=torch.int64, device=device)
                    final_preds[valid_minor_batch_indices] = valid_minor_preds_tensor

            # 收集结果
            all_preds_first_code.extend(final_preds.cpu().numpy())
            all_logits.extend(major_logits.cpu().numpy())
            all_image_ids.extend(image_ids)

    # 过滤OOD样本（测试集默认都是ID样本，将-1替换为大类模型最可能的预测）
    all_preds_first_code = np.array(all_preds_first_code)
    all_logits = np.array(all_logits)
    ood_mask = (all_preds_first_code == -1)
    if ood_mask.sum() > 0:
        logger.warning(f"发现{ood_mask.sum()}个OOD预测样本，替换为大类模型最可能结果")
        ood_major_preds = np.argmax(all_logits[ood_mask], axis=1)
        major2raw = {0:0, 1:5, 2:7, 3:9, 4:1}
        ood_preds = [major2raw[idx] for idx in ood_major_preds]
        all_preds_first_code[ood_mask] = ood_preds

    return all_preds_first_code, all_logits, all_image_ids

# ========================
# 6. Energy Score计算
# ========================
def energy_score(logits, temperature=1.0):
    """Compute Energy Score: higher = more likely In-Distribution."""
    logits_tensor = torch.tensor(logits)
    return -temperature * torch.logsumexp(logits_tensor / temperature, dim=1).numpy()

# ========================
# 7. 主测试函数
# ========================
def test():
    logger.info("🚀 开始加载双模型测试环境...")
    img_folder = BASE_DIR.parent / "DATA_ROOT/test"
    inf_dataset = InfDataset(img_folder, transform=inf_transform)
    inf_dataloader = data.DataLoader(inf_dataset, batch_size=64, shuffle=False)
    logger.info(f"测试集加载完成：共{len(inf_dataset)}张图片")

    num_major_output = len(major_class_indices)+1
    major_model = FeatureExtractor('resnet101', num_classes=num_major_output, dropout=0.5)
    major_ckpt_path = BASE_DIR / "resnet_Major_SAR/best_ema.pth"  # 替换为你的大类模型路径
    major_state = torch.load(major_ckpt_path, map_location=device)
    major_model.load_state_dict(major_state)
    major_model.to(device)
    logger.info("✅ 大类模型加载完成")

    minor_model = None
    if len(minor_class_indices) > 0:
        minor_model = FeatureExtractor('resnet101', num_classes=len(minor_class_indices), dropout=0.5)
        minor_ckpt_path = BASE_DIR / "resnet_Minor_SAR/best_ema.pth"  # 替换为你的小类模型路径
        minor_state = torch.load(minor_ckpt_path, map_location=device)
        minor_model.load_state_dict(minor_state)
        minor_model.to(device)
        logger.info("✅ 小类模型加载完成")

    logger.info("🚀 开始双模型预测...")
    start_time = time.time()
    preds_first_code, logits, image_ids = predict_dual_model_for_test(
        major_model=major_model,
        minor_model=minor_model,
        label2major=label2major,
        minor_unified_label=minor_unified_label,
        major_ood_thresh=0,
        minor_ood_thresh=0,
        dataloader=inf_dataloader
    )
    end_time = time.time()

    preds_current_id = [FIRST_CODE_ID_TO_CURRENT_ID[pid] if pid != -1 else 0 for pid in preds_first_code]
    ood_scores = energy_score(logits, temperature=1.0)

    total = len(image_ids)
    assert total == 4000, f"预期4000个预测，实际{total}个，请检查测试集文件夹"

    df = pd.DataFrame({
        'image_id': [int(imid) for imid in image_ids],
        'class_id': preds_current_id,
        'score': ood_scores
    })
    df.to_csv('results.csv', index=False, header=True)
    logger.info("✅ results.csv生成完成")

    class_counts = Counter(preds_current_id)
    logger.info(f"\n📊 预测类别分布（总数：{total}张图片）：")
    logger.info("-" * 60)
    for cls in range(NUM_CLASSES):
        count = class_counts.get(cls, 0)
        name = OFFICIAL_CLASS_NAMES[cls]
        percentage = 100 * count / total
        logger.info(f"类别{cls:2d} ({name:>25}): {count:5d} ({percentage:5.2f}%)")
    logger.info("-" * 60)

    runtime_per_image = (end_time - start_time) / total
    use_gpu = 0 if str(device).startswith('cuda') else 1

    with open("readme.txt", "w") as f:
        f.write(f"runtime per image [s] : {runtime_per_image:.6f}\n")
        f.write(f"CPU[1] / GPU[0] : {use_gpu}\n")
        f.write("Extra Data [1] / No Extra Data [0] : 0\n")
        f.write("Other description : Dual-model (major+minor) inference using ResNet101. "
                "Prediction uses hard-coded mapping to align with official class ID. "
                "OOD score is Energy Score calculated from major model logits. "
                "Preprocessing uses Resize(256)+CenterCrop(224) to align with training protocol.\n")

    logger.info("📄 readme.txt生成完成")
    logger.info(f"⏱ 单张图片平均推理时间：{runtime_per_image:.6f} 秒")
    logger.info("✅ 测试流程全部完成！")

if __name__ == "__main__":
    test()