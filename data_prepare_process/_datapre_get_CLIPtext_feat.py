import torch
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel


def get_CLIP_text_feat(device, list_caps, path_dict_cap2capF_clip):
    ckpt = "openai/clip-vit-base-patch32"
    model = CLIPModel.from_pretrained(ckpt)
    model.eval()
    model.to(device)
    processor = CLIPProcessor.from_pretrained(ckpt)

    prompt = "a photo of "
    list_prompt_caps = [prompt + cap.strip() for cap in list_caps]

    batch_size = 512
    dict_cap2capF = {}

    for i in tqdm(range(0, len(list_caps), batch_size)):
        batch_captions = list_caps[i:i+batch_size]
        batch_prompt_caps = list_prompt_caps[i:i+batch_size]
        inputs_cap = processor(text=batch_prompt_caps, padding="max_length", truncation=True, max_length=64, return_tensors="pt").to(device)
        with torch.no_grad():
            caption_embeddings = model.get_text_features(**inputs_cap)
        for cap, capF in zip(batch_captions, caption_embeddings):
            dict_cap2capF[cap] = capF

    torch.save(dict_cap2capF, path_dict_cap2capF_clip)

    return dict_cap2capF