import json
import os
import torch
import torchvision
from tqdm import tqdm
from PIL import Image
import torchvision.transforms as T
from sentence_transformers import SentenceTransformer


def get_model(num_classes, ckpt_path, device='cuda:0', NMS_thresh=0.3):
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

    if ckpt_path and os.path.exists(ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        print(f"✅ Loaded checkpoint from {ckpt_path}")

    # ====== 修改内部 NMS 阈值 ======
    model.roi_heads.nms_thresh = NMS_thresh
    # ==============================

    model.to(device).eval()
    return model


def preprocess_image(transform, img_path):
    img = Image.open(img_path).convert("RGB")
    return transform(img)


def data_prepare():
    # 数据准备
    image_root = "SynImg/"  # 合成图像所在文件夹

    with open("dict_imgID2text.json") as f:  # {"index": caption, ...}
        sdf_num_caption_map = json.load(f)  # {'imgID': caption, ...}

    imgs_path = [image_root + str(i) + ".jpg" for i in sdf_num_caption_map.keys()]  # SynImage_idx列表

    imgs_id = list(sdf_num_caption_map.keys())

    dict_imgID2imgPath = {k:v for k,v in zip(imgs_id, imgs_path)}

    return dict_imgID2imgPath
        

@torch.no_grad()
def inference_batch(model, batch_img_tensor, device="cuda:0", score_thresh=0.0):
    """
    对批量图像进行目标检测推理
    
    参数:
        model: FasterRCNN模型
        batch_img_tensor: 图像张量列表，每个元素为单张图像的tensor（shape: [C, H, W]）
        device: 计算设备
        score_thresh: 分数阈值，过滤低置信度检测结果
    
    返回:
        列表，每个元素为对应图像的检测结果字典，包含"labels"
    """
    
    # 1. 将批量图像移至目标设备
    batch_img_tensor = [img.to(device) for img in batch_img_tensor]
    
    # 2. 图像预处理（resize + normalize），支持批量处理
    images_list, _ = model.transform(batch_img_tensor, None)  # images_list包含批量处理后的图像信息
    
    # 3. Backbone提取批量特征
    features = model.backbone(images_list.tensors)  # 输入为拼接后的批量tensor
    if isinstance(features, torch.Tensor):
        features = {"0": features}  # 统一为字典格式，适配后续处理
    
    # 4. RPN生成批量候选框（每个元素对应一张图像的proposals）
    proposals, _ = model.rpn(images_list, features, targets=None)
    
    # 5. RoI Heads计算批量检测结果（transform尺度下）
    detections, _ = model.roi_heads(features, proposals, images_list.image_sizes, targets=None)  # 列表，长度为batch_size
    
    # 6. 对每张图像的检测结果进行分数过滤
    filtered_detections = []
    for det in detections:
        boxes_tf = det["boxes"]  # transform尺度下的框
        labels = det["labels"]
        scores = det["scores"]
        
        # 过滤低置信度结果
        keep = scores >= score_thresh
        filtered_boxes = boxes_tf[keep]
        filtered_labels = labels[keep]
        filtered_scores = scores[keep]
        
        filtered_detections.append({
            "boxes": filtered_boxes,
            "labels": filtered_labels,
            "scores": filtered_scores
        })

    batch_results = []
    for det in filtered_detections:
        batch_results.append({
            "labels": det["labels"].cpu(),  # 类别标签 (N,)
            # "scores": det["scores"].cpu()   # 置信度分数 (N,)
        })
    
    
    return batch_results


def object_detect():
    device = "cuda:1"
    num_classes = 1601   # 类别数
    batch_size = 32
    ckpt_path = "FasterRCNN/checkpoint_e19.pth"
    dict_imgID2imgPath = data_prepare()  # {"imgID": img_path, ...}
    list_imgID2imgPath = list(dict_imgID2imgPath.items())
    
    out_dir = "data_ODres/"
    os.makedirs(out_dir, exist_ok=True)

    transform = T.ToTensor()  # 转 [0,1] float
    model = get_model(num_classes, ckpt_path, device, NMS_thresh=0.5)

    dict_imgID2OD_res = {}

    n_step = len(list_imgID2imgPath) // batch_size + 1
    for i in tqdm(range(n_step)):
        batch_imgID2imgPath = list_imgID2imgPath[i*batch_size:(i+1)*batch_size]
        batch_imgIDs = [item[0] for item in batch_imgID2imgPath]
        batch_imgPaths = [item[1] for item in batch_imgID2imgPath]
        batch_imgs_tensor = [preprocess_image(transform, img_path) for img_path in batch_imgPaths]
        batch_ODres = inference_batch(model, batch_imgs_tensor, device=device, score_thresh=0.3)

        for imgID, ODres in zip(batch_imgIDs, batch_ODres):
            dict_imgID2OD_res[imgID] = ODres

    torch.save(dict_imgID2OD_res, os.path.join(out_dir, f"dict_imgID2OD_res.pt"))

    print(f"✅ All Object Detection results saved to {out_dir}")

    def get_OD_result_label_imgIDkey(dict_imgID2OD_res, out_dir):
        count = 0
        dict_imgID2OD_labels = {}
        for imgID, ODres in tqdm(dict_imgID2OD_res.items()):
            labels = ODres['labels']
            labels = [label.item() for label in labels]
            if labels:
                dict_imgID2OD_labels[imgID] = labels
            else:
                count += 1
        print(count)
        with open(out_dir+f'dict_imgID2OD_labels.json', 'w') as f:
            json.dump(dict_imgID2OD_labels, f)

    get_OD_result_label_imgIDkey(dict_imgID2OD_res=dict_imgID2OD_res,
                                 out_dir=out_dir)


def get_vertex_feature():

    def text2emb_sBERT(list_texts, output_path):
        '''
        list_texts: text list is like: ["text1", "text2", ...]
        output_path: the path of output, output is like: {"text": text_feature, ...}
        '''
        model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        model.to(device)
        model.eval()

        num_step = len(list_texts) // batch_size + 1
        dict_text2textF_sBERT = {}
        with torch.no_grad():
            for i in tqdm.tqdm(range(num_step)):
                batch_texts = list_texts[i*batch_size: (i+1)*batch_size]
                batch_embeddings = model.encode(batch_texts, device=device, convert_to_tensor=True)
                for text, embedding in zip(batch_texts, batch_embeddings):
                    dict_text2textF_sBERT[text] = embedding
        torch.save(dict_text2textF_sBERT, output_path)

    device = "cuda:1"
    batch_size = 512
    with open("Visual_Genome_12/objects1600_COCOformat_filter_overlap.json", "r") as f:
        objects_ann = json.load(f)
    dict_cid2name = {int(category_info['id']):category_info['name'] for category_info in objects_ann['categories']}
    list_texts = list(dict_cid2name.values())
    text2emb_sBERT(list_texts=list_texts,
                output_path="data_ODres/dict_objectTag2feature_sBERT_OD1600.pt")
    

def get_MaxSim_vertexAlabel():

    device = 'cuda:1'

    dict_label2feat = torch.load("data_ODres/dict_objectTag2feature_sBERT_OD1600.pt",
                                  map_location=device)

    label_feats = list(dict_label2feat.values())
    label_feats = torch.stack(label_feats)
    label_feats = label_feats / label_feats.norm(dim=-1, keepdim=True)  # [N, D]

    dict_vertex2feat = torch.load("dict_vertex2feat_WOempty_sBERT.pt",
                                  map_location=device)
    
    dict_vertex2MaxSimID = {}
    for vertex, vf in dict_vertex2feat.items():
        vf = vf / vf.norm(dim=-1, keepdim=True)  # [D]
        sim = vf @ label_feats.T  # [1600,]
        MaxSim, MaxSim_id = torch.max(sim, dim=0)
        dict_vertex2MaxSimID[vertex] = (MaxSim, MaxSim_id)
    
    torch.save(dict_vertex2MaxSimID, "dict_vertex2MaxSimID_between1600.pt")
