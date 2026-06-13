import os
import json
import torch
from math import ceil
from tqdm import tqdm
import argparse
from diffusers import StableDiffusionPipeline


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda:3")
    parser.add_argument("--batch_size", type=int, default=30)
    parser.add_argument("--model_path", type=str,
                        default="sd-legacy/stable-diffusion-v1-5")
    parser.add_argument("--save_dir", type=str, default='SynImg')
    parser.add_argument("--condition_texts", type=str,
                        default="list_corpus_trian.json")
    args = parser.parse_args()
    return args


def generate_syn_img(args):
    device = args.device
    torch_dtype = torch.float16

    # 创建输出文件夹
    image_root = args.save_dir
    os.makedirs(image_root, exist_ok=True)

    # 加载 Stable Diffusion 模型
    pipe = StableDiffusionPipeline.from_pretrained(
        args.model_path, torch_dtype=torch_dtype
    ).to(device)

    # 关闭安全检查
    pipe.safety_checker = lambda images, clip_input: (images, [False] * len(images))

    # 设置随机数种子
    generator = torch.Generator(device).manual_seed(args.seed)

    # 读取文本
    with open(args.condition_texts, "r") as f:
        list_texts = json.load(f)

    # 保存 id 到条件文本的映射
    dict_imgID2text = {i: text for i, text in enumerate(list_texts)}
    with open("dict_imgID2text.json", "w") as f:
        json.dump(dict_imgID2text, f)

    dict_imgID2text = list(dict_imgID2text.items())
    batch_size = args.batch_size

    # 批量生成
    with torch.no_grad():
        for i in tqdm(range(ceil(len(dict_imgID2text) / batch_size))):
            batch_data = dict_imgID2text[i * batch_size:(i + 1) * batch_size]
            if not batch_data:
                continue

            index_list = [item[0] for item in batch_data]
            text_list = [item[1] for item in batch_data]

            images = pipe(
                text_list,
                generator=generator,
                num_inference_steps=20
            ).images

            for idx, img in zip(index_list, images):
                img.save(os.path.join(image_root, f"{idx}.jpg"))


if __name__ == "__main__":
    args = get_args()
    generate_syn_img(args)
