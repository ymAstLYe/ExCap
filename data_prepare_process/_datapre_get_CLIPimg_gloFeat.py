import torch
from PIL import Image
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel


def get_CLIP_global_imgF(device, list_imgIDs, image_root, path_dict_imgID2imgF_clipGlo):
    ckpt = "openai/clip-vit-base-patch32"
    model = CLIPModel.from_pretrained(ckpt)
    model.eval()
    model.to(device)
    processor = CLIPProcessor.from_pretrained(ckpt)

    img_paths = [image_root + str(i) +'.jpg' for i in list_imgIDs]  # SynImage_idx列表

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
            
    if path_dict_imgID2imgF_clipGlo is not None:
        torch.save(dict_imgID2imgF, path_dict_imgID2imgF_clipGlo)

    return dict_imgID2imgF

