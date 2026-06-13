import torch
from tqdm import tqdm
from transformers import AutoProcessor, AutoModel


def get_siglip2_feat(ckpt, device, list_texts, path_siglip_text_feat):
    '''
    输入数据：
    list_texts: 文本列表
    输出数据：
    dict_cap2capF_siglip2.pt, 位置是path_siglip_text_feat
    '''
    model = AutoModel.from_pretrained(ckpt, attn_implementation="sdpa")
    model.eval()
    model.to(device)
    processor = AutoProcessor.from_pretrained(ckpt)

    img_captions = list_texts  # Caption列表

    prompt = "a photo of "

    batch_size = 128
    dict_cap2capF = {}

    for i in tqdm(range(0, len(list_texts), batch_size)):
        batch_captions = img_captions[i:i+batch_size]
        batch_prompt_caps = [prompt + cap.strip().lower() for cap in batch_captions]
        inputs_cap = processor(text=batch_prompt_caps, padding="max_length", max_length=64, truncation=True, return_tensors="pt").to(device)
        with torch.no_grad():
            caption_embeddings = model.get_text_features(**inputs_cap)
        for cap, capF in zip(batch_captions, caption_embeddings):
            dict_cap2capF[cap] = capF
    torch.save(dict_cap2capF, path_siglip_text_feat)

