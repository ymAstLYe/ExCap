import argparse
import os
from tqdm import tqdm
import torch
import json
from PIL import Image
from transformers import CLIPProcessor, CLIPModel


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--subset_idx", type=int, default=0)
    args = parser.parse_args()
    return args


def main(args, subset_size, clip_model, preprocess, mlp_layer, postlayernorm, img_captions, imgs_path, out_root):
    device = args.device

    # 子集分割
    img_captions = img_captions[subset_size*args.subset_idx: subset_size*(args.subset_idx+1)]
    imgs_path = imgs_path[subset_size*args.subset_idx: subset_size*(args.subset_idx+1)]

    # 批量与批次
    batch_size = args.batch_size
    if len(imgs_path) % batch_size == 0:
        iter_num = len(imgs_path) // batch_size
    else:
        iter_num = (len(imgs_path) // batch_size) + 1

    # 流程
    cap2imgAllF = {}
    for i in tqdm(range(iter_num)):
        with torch.no_grad():
            batch_img_path = imgs_path[i * batch_size: (i + 1) * batch_size]
            batch_img_instance = [Image.open(img_path) for img_path in batch_img_path]
            batch_caption = img_captions[i * batch_size: (i + 1) * batch_size]
            # 批量预处理图像
            imgs_tensor = preprocess(images=batch_img_instance, return_tensors="pt", padding=True)
            imgs_tensor = imgs_tensor['pixel_values'].to(device)
            # 获取图像所有特征
            vision_outputs = clip_model.vision_model(imgs_tensor)
            image_features = vision_outputs.last_hidden_state
            image_features = postlayernorm(image_features)
            mapped_features = mlp_layer(image_features)
            # 保存数据
            for caption, img_all_feature in zip(batch_caption, mapped_features):
                cap2imgAllF[caption] = img_all_feature.cpu().float()
    torch.save(cap2imgAllF, out_root + f"/train_cap2imgAllF_{args.subset_idx}.pt")


if __name__ == "__main__":
    '''
    输入图像目录与配对关系{caption: imgID}，输出对应{caption: CLIPfeats, ...}
    '''
    # 数据准备
    image_root = "SynImg/"  # 合成图像所在文件夹
    with open("data_EMHM/dict_cap2imgIDrep_full.json", 'r') as f:
        dict_cap2imgID = json.load(f)

    img_captions = list(dict_cap2imgID.keys())  # Caption列表
    imgs_path = [image_root + str(i) + ".jpg" for i in dict_cap2imgID.values()]  # SynImage_idx列表

    out_root = 'data_EMHM/CLIPfeats'
    os.makedirs(out_root, exist_ok=True)

    n_subset = 49
    subset_size = len(dict_cap2imgID) // n_subset + 2

    args = get_args()
    # 模型准备
    device = args.device
    model_path = "openai/clip-vit-base-patch32"
    clip_model = CLIPModel.from_pretrained(model_path)
    preprocess = CLIPProcessor.from_pretrained(model_path)
    clip_model.to(device)
    clip_model.eval()
    mlp_layer = clip_model.visual_projection
    postlayernorm = clip_model.vision_model.post_layernorm

    for i in range(n_subset):
        args.subset_idx = i
        main(args, subset_size, clip_model, preprocess, mlp_layer, postlayernorm,
             img_captions, imgs_path, out_root)