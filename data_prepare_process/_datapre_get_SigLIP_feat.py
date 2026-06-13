import json
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, AutoModel


def get_siglip2_text_feat(list_caps, processor, model, dir_siglip_feat, device):

    img_captions = list_caps

    prompt = "a photo of "

    batch_size = 128
    dict_cap2capF = {}

    for i in tqdm(range(0, len(img_captions), batch_size)):
        batch_captions = img_captions[i:i+batch_size]
        batch_prompt_caps = [prompt + cap.strip().lower() for cap in batch_captions]
        inputs_cap = processor(text=batch_prompt_caps, padding="max_length", max_length=64, truncation=True, return_tensors="pt").to(device)

        with torch.no_grad():
            caption_embeddings = model.get_text_features(**inputs_cap)
        for cap, capF in zip(batch_captions, caption_embeddings):
            dict_cap2capF[cap] = capF

    torch.save(dict_cap2capF, dir_siglip_feat+"/dict_cap2capF_siglip2.pt")


def get_siglip2_img_feat(image_root, list_imgIDs, processor, model, device, dir_siglip_feat):
    img_paths = [image_root + str(i) + ".jpg" for i in list_imgIDs]  # SynImage_idx列表

    batch_size = 128
    dict_imgID2imgF = {}

    for i in tqdm(range(0, len(list_imgIDs), batch_size)):

        batch_imgIDs = list_imgIDs[i:i+batch_size]
        batch_imgPaths = img_paths[i:i+batch_size]
        
        batch_imgs = [Image.open(img_path).convert("RGB") for img_path in batch_imgPaths]
        inputs_img = processor(images=batch_imgs, return_tensors="pt").to(device)
        
        with torch.no_grad():
            image_embeddings = model.get_image_features(**inputs_img)

        for imgID, imgF in zip(batch_imgIDs, image_embeddings):
            dict_imgID2imgF[imgID] = imgF

    torch.save(dict_imgID2imgF, dir_siglip_feat+"/dict_imgID2imgF_siglip2.pt")


def get_siglip2_feat(ckpt, device, image_root, list_imgIDs, list_caps, dir_siglip_feat):
    '''
    输入数据：
    list_caps, 即caption的列表
    list_imgIDs, 即图像名称的列表
    输出数据：
    dict_imgID2imgF_siglip2.pt/dict_cap2capF_siglip2.pt, 在dir_siglip_feat目录下
    '''
    model = AutoModel.from_pretrained(ckpt, attn_implementation="sdpa")
    model.eval()
    model.to(device)
    processor = AutoProcessor.from_pretrained(ckpt)

    if image_root is not None:
        get_siglip2_img_feat(image_root=image_root, list_imgIDs=list_imgIDs, device=device,
                            dir_siglip_feat=dir_siglip_feat, 
                            processor=processor, model=model)
    
    if list_caps is not None:
        get_siglip2_text_feat(list_caps=list_caps, device=device, 
                            dir_siglip_feat=dir_siglip_feat,
                            processor=processor, model=model)
