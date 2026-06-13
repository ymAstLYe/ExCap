import argparse
import random
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup


def set_seed(seed_value):  # 设置随机数种子
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class PreprocessedCocoSubsetDataset(Dataset):
    """
    读取预处理后的 subset_x.pt 文件
    每个 .pt 文件里保存了 (images, targets)，其中:
        images: list[Tensor], 每个 Tensor shape = [C, H, W]
        targets: list[Dict], 适配 torchvision FasterRCNN 格式
    """
    def __init__(self, pt_files, args):
        """
        参数:
            pt_files: str 或 list[str]
                      如果是 str，表示一个文件路径
                      如果是 list[str]，表示多个 subset 文件路径
        """
        if isinstance(pt_files, str):
            pt_files = [pt_files]
        self.pt_files = pt_files

        # 加载所有 subset 文件到内存
        self.images = []
        self.targets = []
        for pt in pt_files:
            imgs, tgts = torch.load(pt, map_location=args.device)
            self.images.extend(imgs)
            self.targets.extend(tgts)

        assert len(self.images) == len(self.targets), "images 和 targets 数量不匹配！"

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        """
        返回:
            image: Tensor [C, H, W], float32
            target: dict {boxes, labels, image_id, area, iscrowd}
        """
        return self.images[idx], self.targets[idx]


def get_model(num_classes, NMS_thresh=0.3):
    # 加载 Faster R-CNN（ResNet-101 + FPN）
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(weights=None)  # 先加载50
    # 1. 替换为 ResNet-101
    from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
    backbone = resnet_fpn_backbone(backbone_name='resnet101', weights="IMAGENET1K_V1")
    model.backbone = backbone

    # 2. 修改 box_head
    in_features = model.roi_heads.box_head[5].in_features   # 12544
    out_features = 2048
    model.roi_heads.box_head[5] = torch.nn.Linear(in_features, out_features)

    # 3. 替换分类头
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
        out_features, num_classes
    )

    # ====== 修改内部 NMS 阈值 ======
    model.roi_heads.nms_thresh = NMS_thresh
    print("IoU threshold is:", model.roi_heads.nms_thresh)
    return model


def collate_fn(batch):
    return tuple(zip(*batch))


def train():
    out_dir = args.output_dir  # 训得模型存储的目录
    print("训练后模型存储的地址是：", out_dir)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    pp_img_tgts_root = args.pp_root  # 预处理的(img_tensor, tgts)数据的目录

    # 加载模型
    num_classes = args.num_classes
    model = get_model(num_classes)
    device = args.device
    model.to(device)

    # 优化器与学习率调度策略
    def get_param_groups(model, base_lr, lr_multiplier, wd=1e-5):
        """
        将模型参数分为不同组，应用不同学习率和权重衰减
        
        Args:
            model: 模型实例
            base_lr: 基础学习率
            lr_multiplier: 新添加层的学习率乘数
            wd: 权重衰减值
        Returns:
            param_groups: 参数组列表，用于传入优化器
        """
        # 初始化参数组
        pretrained_params = []  # 预训练模型参数（低学习率）
        other_params = []  # 其他模块参数（高学习率）
        
        # 获取backbone中的参数
        pretrained_module = model.backbone
        for name, param in pretrained_module.named_parameters():
            pretrained_params.append(param)
        
        # 获取模型其他部分的参数
        for name, param in model.named_parameters():
            if 'backbone' not in name:
                other_params.append(param)
        
        # 构建参数组
        param_groups = [
            {'params': pretrained_params, 'lr': base_lr},
            {'params': other_params, 'lr': base_lr * lr_multiplier}
        ]
        
        return param_groups

    params = get_param_groups(model, args.lr, lr_multiplier=2)
    optimizer = torch.optim.Adam(params)

    # 训练
    if args.resume_path:
        resume_path = os.path.join(out_dir, args.resume_path)
        checkpoint = torch.load(resume_path, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_epoch = checkpoint["epoch"] + 1
        num_training_steps = checkpoint["num_training_steps"]
        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=int(num_training_steps*0.1), num_training_steps=num_training_steps)
        lr_scheduler.load_state_dict(checkpoint["scheduler_state"])
        print(f"🔄 Resumed from epoch {checkpoint['epoch']}")
    else:
        if args.finetune:
            finetune_path = args.finetune
            checkpoint = torch.load(finetune_path, map_location=device)
            model.load_state_dict(checkpoint["model_state"])
            print(f"🔄 Finetune from {finetune_path}")
        start_epoch = 0
        num_training_steps = args.epoch * (args.len_all_dataset // args.batch_size)
        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=int(num_training_steps*0.1), num_training_steps=num_training_steps)

    num_epochs = args.epoch
    for epoch in range(start_epoch, num_epochs):
        model.train()
        epoch_loss = 0.0
        step_count = 0
        random_subset = random.sample(range(args.num_subset), args.num_subset)
        for subset in random_subset:  # 每次随机读入一个子集
            subset_path = os.path.join(pp_img_tgts_root, f"subset_split_{subset+1}.pt")  # 子集名格式为“subset_split_i.pt”，i从1开始
            dataset = PreprocessedCocoSubsetDataset(subset_path, args)
            data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
            for imgs, targets in tqdm(data_loader):
                imgs = list(imgs)
                targets = list(targets)
                loss_dict = model(imgs, targets)

                losses = sum(loss for loss in loss_dict.values())

                optimizer.zero_grad()
                losses.backward()
                optimizer.step()
                lr_scheduler.step()

                epoch_loss += losses.item()
                step_count += 1

        avg_loss = epoch_loss / max(step_count, 1)
        print(f"Epoch {epoch}, Avg Loss: {avg_loss:.4f}")
        
        # 保存模型
        checkpoint = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": lr_scheduler.state_dict(),
            "num_training_steps": num_training_steps,
        }
        save_path = os.path.join(out_dir, f"checkpoint_e{epoch}.pth")
        torch.save(checkpoint, save_path)
        print(f"💾 Saved checkpoint: {save_path}")

        with open(out_dir + f"train_log.txt", 'a+') as f:
            f.writelines('epoch ' + str(epoch) + ': ' + str(avg_loss) + '\r\n')

    print("✅ 训练完成，模型已保存")


if __name__ == "__main__":

    def get_args():  # 定义了一个命令行参数解析器，并为其添加了多个参数
        parser = argparse.ArgumentParser()
        parser.add_argument("--seed", type=int, default=42)
        parser.add_argument("--device", type=str, default="cuda:0")  # <--
        parser.add_argument("--batch_size", type=int, default=4)
        parser.add_argument("--lr", type=float, default=5e-5)
        parser.add_argument("--epoch", type=int, default=20)
        
        parser.add_argument("--num_classes", type=int, default=1601)  # 包括背景类
        parser.add_argument("--resume_path", type=str, default=None)
        parser.add_argument("--finetune", type=str, default=None)

        parser.add_argument("--output_dir", type=str, default="FasterRCNN/")  # <--
        parser.add_argument("--pp_root", type=str, default="Visual_Genome_12/preprocessed_img_targets/")

        parser.add_argument("--len_all_dataset", type=int, default=1e5)
        parser.add_argument("--num_subset", type=int, default=157)

        args = parser.parse_args()
        return args
    
    args = get_args()
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")
    set_seed(args.seed)
    train()
