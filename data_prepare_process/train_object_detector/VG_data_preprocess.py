import json
import os
import shutil
from typing import List
from collections import Counter
import torch
from tqdm import tqdm
from torchvision.datasets import CocoDetection
from torchvision import transforms as T
from pycocotools.coco import COCO
import gc


def get_overlap_with_coco():
    with open("Visual_Genome_12/annotation/image_data.json", "r") as f:
        list_img_info = json.load(f)  
    # image_data is like: 
    # [{"width": 800, "url": "......", "height": 600, "image_id": 1, "coco_id": null, "flickr_id": null}, ...]

    with open("dataset_coco.json", "r") as f:  # input dataset_coco or dataset_flickr30k
        dataset_IC = json.load(f)
    IC_ids = []
    for img_info in dataset_IC['images']:
        IC_ids.append(str(img_info['cocoid']))
        # IC_ids.append(str(img_info['filename'].split('.')[0]))  
    '''
    dataset_coco is like:
    {'filepath': 'val2014', 'sentids': [770337, 771687, 772707, 776154, 781998], 
    'filename': 'COCO_val2014_000000391895.jpg', 
    'imgid': 0, 'split': 'test', 
    'sentences': 
    [{'tokens': ['a', 'man', 'with', 'a', 'red', 'helmet', 'on', 'a', 'small', 'moped', 'on', 'a', 'dirt', 'road'], 'raw': 'A man with a red helmet on a small moped on a dirt road. ', 'imgid': 0, 'sentid': 770337}, 
    {'tokens': ['man', 'riding', 'a', 'motor', 'bike', 'on', 'a', 'dirt', 'road', 'on', 'the', 'countryside'], 'raw': 'Man riding a motor bike on a dirt road on the countryside.', 'imgid': 0, 'sentid': 771687}, 
    {'tokens': ['a', 'man', 'riding', 'on', 'the', 'back', 'of', 'a', 'motorcycle'], 'raw': 'A man riding on the back of a motorcycle.', 'imgid': 0, 'sentid': 772707}, 
    {'tokens': ['a', 'dirt', 'path', 'with', 'a', 'young', 'person', 'on', 'a', 'motor', 'bike', 'rests', 'to', 'the', 'foreground', 'of', 'a', 'verdant', 'area', 'with', 'a', 'bridge', 'and', 'a', 'background', 'of', 'cloud', 'wreathed', 'mountains'], 'raw': 'A dirt path with a young person on a motor bike rests to the foreground of a verdant area with a bridge and a background of cloud-wreathed mountains. ', 'imgid': 0, 'sentid': 776154}, 
    {'tokens': ['a', 'man', 'in', 'a', 'red', 'shirt', 'and', 'a', 'red', 'hat', 'is', 'on', 'a', 'motorcycle', 'on', 'a', 'hill', 'side'], 'raw': 'A man in a red shirt and a red hat is on a motorcycle on a hill side.', 'imgid': 0, 'sentid': 781998}], 
    'cocoid': 391895}
    '''
    '''
    dataset_flickr30k is like:
    {'sentids': [0, 1, 2, 3, 4], 'imgid': 0, 
    'sentences': 
    [{'tokens': ['two', 'young', 'guys', 'with', 'shaggy', 'hair', 'look', 'at', 'their', 'hands', 'while', 'hanging', 'out', 'in', 'the', 'yard'], 'raw': 'Two young guys with shaggy hair look at their hands while hanging out in the yard.', 'imgid': 0, 'sentid': 0}, 
    {'tokens': ['two', 'young', 'white', 'males', 'are', 'outside', 'near', 'many', 'bushes'], 'raw': 'Two young, White males are outside near many bushes.', 'imgid': 0, 'sentid': 1}, 
    {'tokens': ['two', 'men', 'in', 'green', 'shirts', 'are', 'standing', 'in', 'a', 'yard'], 'raw': 'Two men in green shirts are standing in a yard.', 'imgid': 0, 'sentid': 2}, 
    {'tokens': ['a', 'man', 'in', 'a', 'blue', 'shirt', 'standing', 'in', 'a', 'garden'], 'raw': 'A man in a blue shirt standing in a garden.', 'imgid': 0, 'sentid': 3}, 
    {'tokens': ['two', 'friends', 'enjoy', 'time', 'spent', 'together'], 'raw': 'Two friends enjoy time spent together.', 'imgid': 0, 'sentid': 4}], 'split': 'train', 
    'filename': '1000092795.jpg'}
    '''

    set_imgIDs_IC = set(IC_ids)

    list_overlap_VG = []
    list_overlap_IC = []

    for img_info in list_img_info:
        IC_id_of_VG = img_info["coco_id"]
        # IC_id_of_VG = img_info["flickr_id"]
        if IC_id_of_VG and str(IC_id_of_VG) in set_imgIDs_IC:
            list_overlap_VG.append(img_info['image_id'])
            list_overlap_IC.append(IC_id_of_VG)

    dict_VGid2ICid = {k:v for k,v in zip(list_overlap_VG, list_overlap_IC)}
    with open('dict_VGid2COCOid_overlap.json', 'w') as f:
    # with open('dict_VGid2Flickid_overlap.json', 'w') as f:
        json.dump(dict_VGid2ICid, f)
    print(len(dict_VGid2ICid))


def convert_objects_to_cocoformat(objects_json_path, images_dir, output_path):
    """
    将 objects.json 转换为 COCO 格式标注
    Args:
        objects_json_path (str): 输入 objects.json 路径
        images_dir (str): 存放图片的目录 (用于获取宽高)
        output_path (str): 输出 coco_annotations.json 路径
    """
    from PIL import Image

    # 读取 objects.json
    with open(objects_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # ====== 收集类别，建立映射 ======
    categories_set = set()
    for item in data:
        for obj in item["objects"]:
            if obj["names"]:
                categories_set.add(obj["names"][0])

    categories = sorted(list(categories_set))  # 实体类
    # 构造实体类别到id的映射，可能用于分类的label
    label2id = {label: idx + 1 for idx, label in enumerate(categories)}  # COCO 类别 ID 从 1 开始

    coco = {
        "images": [],
        "annotations": [],
        "categories": []
    }

    # 写入 categories，获取分类对照表
    for label, cid in label2id.items():
        coco["categories"].append({  # 实体类
            "id": cid,
            "name": label,
            "supercategory": "none"
        })

    # ann_id = 1
    for item in data:
        image_id = item["image_id"]

        # 获取图片路径
        img_filename = f"{image_id}.jpg"  # 假设文件名就是 image_id.jpg
        img_path = os.path.join(images_dir, img_filename)
        if not os.path.exists(img_path):
            print(f"⚠️ Warning: {img_path} not found, skipping")
            continue

        # 获取图像宽高
        with Image.open(img_path) as img:
            width, height = img.size

        # 添加 image 信息
        coco["images"].append({  # 图像信息
            "id": image_id,
            "file_name": img_filename,
            "width": width,
            "height": height
        })

        # 添加 annotation 信息
        for obj in item["objects"]:  
            if not obj["names"]:
                continue
            label = obj["names"][0]
            category_id = label2id[label]

            x, y, w, h = obj["x"], obj["y"], obj["w"], obj["h"]
            bbox = [x, y, w, h]
            area = w * h
            # 个人修改点
            ann_id = obj["object_id"]
            coco["annotations"].append({  # 图像的实体并不与图像一一对应
                "id": ann_id,
                "image_id": image_id,
                "category_id": category_id,
                "bbox": bbox,
                "area": area,
                "iscrowd": 0,
                "segmentation": []  # 这里不提供分割
            })
            # ann_id += 1

    # 保存 COCO JSON
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(coco, f, indent=2, ensure_ascii=False)

    print(f"✅ 转换完成，COCO形式 标注已保存到 {output_path}")

'''objects_json_path = "Visual_Genome_12/annotation/objects.json"
images_dir = "Visual_Genome_12/images/VG_100K"  # 存放图片的文件夹
output_path = "Visual_Genome_12/annotation_COCOformat/objects_COCOformat.json"
convert_objects_to_cocoformat(objects_json_path, images_dir, output_path)'''


def fliter_overlap():  # 需对coco和flick分别过滤
    with open("dict_VGid2COCOid_overlap.json", "r") as f:
        dict_VGid2ICid = json.load(f) 
    print(len(dict_VGid2ICid))

    # with open('dict_VGid2Flickid_overlap.json', 'r') as f:
    #     dict_VGid2ICid = json.load(f) 
    # print(len(dict_VGid2ICid))        

    def filter_coco_by_overlap(input_json, output_json, set_overlap_imgs):
        # 读取原始 COCO形式 标注
        with open(input_json, "r", encoding="utf-8") as f:
            coco = json.load(f)

        # step1: 过滤 images
        kept_images = [img for img in coco["images"] if img["id"] not in set_overlap_imgs]
        kept_image_ids = {img["id"] for img in kept_images}

        # step2: 过滤 annotations
        kept_annotations = [ann for ann in coco["annotations"] if ann["image_id"] in kept_image_ids]

        # step3: 找到还在使用的类别
        used_category_ids = {ann["category_id"] for ann in kept_annotations}
        kept_categories = [cat for cat in coco["categories"] if cat["id"] in used_category_ids]

        # step4: 重新组装 coco dict
        filtered_coco = {
            "images": kept_images,
            "annotations": kept_annotations,
            "categories": kept_categories
        }

        # 保存结果
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(filtered_coco, f, ensure_ascii=False, indent=2)

        print(f"过滤完成：输入 {len(coco['images'])} 张图像，输出 {len(kept_images)} 张图像")
        print(f"输入 {len(coco['annotations'])} 条标注，输出 {len(kept_annotations)} 条标注")
        print(f"输入 {len(coco['categories'])} 个类别，输出 {len(kept_categories)} 个类别")

    overlap_imgs = set([int(k) for k in dict_VGid2ICid.keys()])  # str类型，需转化为int
    filter_coco_by_overlap(input_json='Visual_Genome_12/annotation_COCOformat/objects_COCOformat.json',
                        output_json='Visual_Genome_12/annotation_COCOformat/objects_COCOformat_filter_overlap.json',
                            set_overlap_imgs=overlap_imgs
                        )


def filter_vg_to_top_categories(coco_input, coco_output, topk=1600):
    # 读取 COCO 格式
    with open(coco_input, "r") as f:
        coco = json.load(f)

    images = coco["images"]
    annotations = coco["annotations"]
    categories = coco["categories"]

    # 统计每个类别出现次数
    cat_counter = Counter([ann["category_id"] for ann in annotations])

    # 选出前 topk 个类别
    top_cats = set([cat_id for cat_id, _ in cat_counter.most_common(topk)])
    print(f"保留前 {topk} 个类别，覆盖的标注数量: {sum(cat_counter[c] for c in top_cats)}")

    # 过滤 annotations
    new_annotations = [ann for ann in annotations if ann["category_id"] in top_cats]

    # 重新编号 categories (从1开始)
    old2new_catid = {old_id: new_id+1 for new_id, old_id in enumerate(top_cats)}
    new_categories = []
    for old_id, new_id in old2new_catid.items():
        cat_name = next((c["name"] for c in categories if c["id"] == old_id), f"unk_{old_id}")
        new_categories.append({"id": new_id, "name": cat_name})
    # 更新 annotation 的类别id
    for ann in new_annotations:
        ann["category_id"] = old2new_catid[ann["category_id"]]

    # 过滤没有标注的 images
    ann_img_ids = set([ann["image_id"] for ann in new_annotations])
    new_images = [img for img in images if img["id"] in ann_img_ids]

    new_coco = {
        "images": new_images,
        "annotations": new_annotations,
        "categories": new_categories
    }

    with open(coco_output, "w") as f:
        json.dump(new_coco, f)

    print(f"完成: 输出 {len(new_images)} 张图像，{len(new_annotations)} 个标注，{len(new_categories)} 个类别")
    return new_coco

# filter_vg_to_top_categories("Visual_Genome_12/annotation_COCOformat/objects_COCOformat_filter_overlap.json", 
#                             "Visual_Genome_12/objects1600_COCOformat_filter_overlap.json", topk=1600)


def preprocess_coco_to_pt(img_folder, ann_file, output_folder, num_subsets=20, device="cpu"):
    """
    将 COCO 格式的标注和图像数据，划分为多个子集，每个子集单独保存为 .pt 文件
    格式： (images, targets)，
        images: list[Tensor(C,H,W)] float32 in [0,1]
        targets: list[dict] 适配 torchvision FasterRCNN
    """

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    coco = COCO(ann_file)
    img_ids = coco.getImgIds()
    total_images = len(img_ids)
    subset_size = (total_images + num_subsets - 1) // num_subsets

    print(f"Total {total_images} images, split into {num_subsets} subsets, "
          f"~{subset_size} images each")

    transform = T.ToTensor()  # 转为 [0,1] float32

    for subset_idx in range(num_subsets):
        start = subset_idx * subset_size
        end = min((subset_idx + 1) * subset_size, total_images)
        sub_img_ids = img_ids[start:end]
        if not sub_img_ids:
            continue

        # ---- 生成子集标注 JSON ----
        sub_ann_file = os.path.join(output_folder, f"subset_{subset_idx}.json")
        coco_subset = coco.loadImgs(sub_img_ids)

        sub_anns = []
        for img_id in sub_img_ids:
            ann_ids = coco.getAnnIds(imgIds=[img_id])
            anns = coco.loadAnns(ann_ids)
            sub_anns.extend(anns)

        sub_coco_dict = {
            "images": coco_subset,
            "annotations": sub_anns,
            "categories": coco.loadCats(coco.getCatIds())
        }

        with open(sub_ann_file, "w") as f:
            json.dump(sub_coco_dict, f)

        # ---- 用子集 ann_file + 原图像文件夹 构造 Dataset ----
        dataset = CocoDetection(img_folder, sub_ann_file)

        images, targets = [], []
        for img, anns in tqdm(dataset, desc=f"Subset {subset_idx}/{num_subsets}"):
            # 转为 [0,1] float32
            img_tensor = transform(img)  # [C,H,W], float32

            boxes = []
            labels = []
            areas = []
            iscrowd = []
            for ann in anns:
                x, y, w, h = ann["bbox"]
                boxes.append([x, y, x + w, y + h])
                labels.append(ann["category_id"])
                areas.append(ann["area"])
                iscrowd.append(ann.get("iscrowd", 0))

            target = {
                "boxes": torch.tensor(boxes, dtype=torch.float32),
                "labels": torch.tensor(labels, dtype=torch.int64),
                "image_id": torch.tensor([ann["image_id"]], dtype=torch.int64),
                "area": torch.tensor(areas, dtype=torch.float32),
                "iscrowd": torch.tensor(iscrowd, dtype=torch.int64),
            }

            images.append(img_tensor)
            targets.append(target)

        out_path = os.path.join(output_folder, f"subset_{subset_idx}.pt")
        torch.save((images, targets), out_path)
        print(f"✅ Saved subset {subset_idx} with {len(images)} images+targets to {out_path}")

        # ---- 内存清理 ----
        del images, targets, dataset
        gc.collect()

    print("🎉 全部子集预处理完成！")

# img_folder = "Visual_Genome_12/images/VG_100K"              # 原始图片路径
# ann_file = "Visual_Genome_12/objects1600_COCOformat_filter_overlap.json" # COCO 格式标注
# output_folder = "Visual_Genome_12/preprocessed_img_targets"  # 输出 .pt 文件路径
# preprocess_coco_to_pt(img_folder, ann_file, output_folder, num_subsets=128)

def split_large_subsets(folder, threshold_gb=2.0):
    """
    自动查找超过阈值的 subset_*.pt 文件，并进行二分拆分。
    
    参数:
        folder: str, 子集文件所在目录
        threshold_gb: float, 拆分的大小阈值 (单位: GB)
    """
    threshold_bytes = threshold_gb * (1024 ** 3)

    pt_files = [f for f in os.listdir(folder) if f.endswith(".pt")]
    pt_files = sorted(pt_files)

    for fname in pt_files:
        fpath = os.path.join(folder, fname)
        fsize = os.path.getsize(fpath)

        if fsize > threshold_bytes:
            print(f"⚠️ File {fname} is {fsize/1e9:.2f} GB (> {threshold_gb} GB), splitting...")

            # 加载数据
            images, targets = torch.load(fpath, map_location="cpu")
            total = len(images)
            mid = total // 2

            # 拆分为两个新子集
            subset_a = (images[:mid], targets[:mid])
            subset_b = (images[mid:], targets[mid:])

            # 构造新文件名
            base, _ = os.path.splitext(fname)
            fpath_a = os.path.join(folder, f"{base}_a.pt")
            fpath_b = os.path.join(folder, f"{base}_b.pt")

            torch.save(subset_a, fpath_a)
            torch.save(subset_b, fpath_b)

            print(f"✅ Saved {fpath_a} ({len(subset_a[0])} samples)")
            print(f"✅ Saved {fpath_b} ({len(subset_b[0])} samples)")

            # 删除原文件
            os.remove(fpath)
            print(f"🗑️ Deleted {fpath}")

            # 内存清理
            del images, targets, subset_a, subset_b
            gc.collect()

    print("🎉 Finished checking and splitting large subsets.")
# folder = "Visual_Genome_12/preprocessed_img_targets"
# split_large_subsets(folder, threshold_gb=1.9)


def rename_pt_files(directory):
    """
    将指定目录下所有.pt文件重命名为subset_split_i.pt格式
    
    Args:
        directory: 要处理的目录路径
    """
    # 检查目录是否存在
    if not os.path.isdir(directory):
        print(f"错误：目录 '{directory}' 不存在")
        return
    
    # 获取目录下所有.pt文件，并按名称排序
    pt_files = [f for f in os.listdir(directory) 
                if os.path.isfile(os.path.join(directory, f)) 
                and f.lower().endswith('.pt')]
    
    # 按文件名排序，确保编号顺序可预测
    pt_files.sort()
    
    # 重命名文件
    for i, filename in enumerate(pt_files, start=1):
        # 构建旧文件路径和新文件路径
        old_path = os.path.join(directory, filename)
        new_filename = f"subset_split_{i}.pt"
        new_path = os.path.join(directory, new_filename)
        
        # 避免文件名冲突
        if os.path.exists(new_path):
            print(f"警告：文件 '{new_filename}' 已存在，跳过 '{filename}'")
            continue
        
        # 执行重命名
        os.rename(old_path, new_path)
        print(f"已重命名: {filename} -> {new_filename}")
    
    print(f"处理完成，共重命名 {len(pt_files)} 个文件")
# rename_pt_files(directory='Visual_Genome_12/preprocessed_img_targets')