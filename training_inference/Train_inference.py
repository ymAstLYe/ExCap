import json
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
import argparse
import os
import tqdm
import Model_Captioning as CapModel
from transformers import get_linear_schedule_with_warmup
from torch.nn.utils.rnn import pad_sequence
EPS = 1e-7


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


class TrainDataset(Dataset):  # 训练集，输出为单位化的caption、图像特征、实体特征

    def __init__(self, args, shared_data):
        self.shared_data = shared_data  # 共享数据（已加载）

        self.dict_cap2imgAllF = torch.load(args.repair_dir + f"CLIPfeats/train_cap2imgAllF_{args.subset_idx}.pt", map_location=args.device)
        self.list_cap2imgAllF = list(self.dict_cap2imgAllF.items())

        self.dict_capOri2capFin = self.shared_data['dict_capOri2capFin']
        self.dict_capFin2imgIDrep = self.shared_data['dict_capFin2imgIDrep']  # 用以取图像的全局特征

        # 其他数据直接从 shared_data 中获取
        self.prefix = self.shared_data["prefix"]
        # 文本相关
        self.support_features_capF = self.shared_data["support_features_capF"]  # 文本特征支撑集
        self.dict_imgID2ngblist_I2T = self.shared_data["neighbors_cap_dict"]  # 图像的文本特征近邻
        # 各向异性加噪相关
        self.dict_capOri2ngbs_index = self.shared_data["neighbors_dict"]  # 图像近邻
        self.support_features_img = self.shared_data["support_features"]  # 图像特征支撑集
        # 图像全局特征
        self.imgID2imgF_use_or_noisy = self.shared_data["train_cap2img_feature_dict"]  # 合成图像特征

        # 实体相关
        self.dict_entity2textF = shared_data['dict_entity2textF']
        self.list_caps = shared_data['list_caps']
        self.dict_cap2vertexs = shared_data['dict_cap2vertexs']

        self.memorybank_entityStr = list(self.dict_entity2textF.keys())
        self.memorybank_entityF = torch.stack(list(self.dict_entity2textF.values()))
    
    def __len__(self) -> int:
        return len(self.dict_cap2imgAllF)  # 训练集的长度

    def __getitem__(self, index: int):
        caption, img_all_features = self.list_cap2imgAllF[index]  # img_features.shape=[50, 768]

        # 各向异性加噪相关
        ngb_idx_img = self.dict_capOri2ngbs_index[caption]

        if caption in self.dict_capOri2capFin:
            caption = self.dict_capOri2capFin[caption]

        imgID = self.dict_capFin2imgIDrep[caption]
        img_gol_feature = self.imgID2imgF_use_or_noisy[imgID]
        img_gol_feature_fix = img_gol_feature.clone()

        if args.use_noisy:
            # 1.高斯加噪
            img_gol_feature, noise_gauss = self.add_noise_gauss(img_feature=img_gol_feature, img_feature_fix=img_gol_feature_fix)
            # 2.各向异性加噪
            img_gol_feature, _ = self.add_noise_neighbor(neighbor=ngb_idx_img, img_feature=img_gol_feature, img_feature_fix=img_gol_feature_fix, noise_gauss=noise_gauss)

        # 检索投影
        proj_feature, ret_cap_idx = self.cap_retrieve_project(imgID=imgID, img_feature=img_gol_feature)
        # 归一化
        proj_feature_norm = proj_feature / proj_feature.norm(dim=-1, keepdim=True)

        entities = get_entity_from_sentence(ret_cap_idx=ret_cap_idx, list_caps=self.list_caps,
                                            dict_cap2vertexs=self.dict_cap2vertexs, n_entities=args.n_entities)
        
        if not entities:  # 如果检索到的caption都没解析出实体
            entities = retrieve_img2text(query=img_gol_feature, memory_bank=self.memorybank_entityF,
                                         memory_keys=self.memorybank_entityStr,
                                         num_answer=args.n_entities)
        
        entities_feats = torch.stack([self.dict_entity2textF[entity] for entity in entities])
        entities = entities[:args.n_hp]
        hp_entities = ''
        for i, entity in enumerate(entities):
            if i < len(entities)-1:
                hp_entities += ' ' + entity + ','
            else:
                hp_entities += ' ' + entity + ' '
        hard_prompt = 'there are' + hp_entities + 'in the image.'

        caption = hard_prompt.lower() + self.prefix + (caption.strip().split('.')[0] + '.').lower().lstrip("'\"\\@&:()#")  # 格式化

        img_all_features_norm = img_all_features / img_all_features.norm(dim=-1, keepdim=True)

        return caption, proj_feature_norm, img_all_features_norm[1:, :], entities_feats

    def add_noise_gauss(self, img_feature, img_feature_fix):
        # 1.高斯噪声
        noisy = torch.randn_like(img_feature)
        noisy_level1 = args.noisy_level1
        delta = noisy * noisy_level1
        output_feat = img_feature + delta
        return output_feat, noisy
    
    def add_noise_neighbor(self, neighbor, img_feature, img_feature_fix, noise_gauss):
        # 2.各向异性加噪--仍然以近邻特征与输入特征之差为方向
        noisy_level2 = args.noisy_level2
        ngb_noise = torch.zeros_like(img_feature, device=img_feature.device, dtype=img_feature.dtype)
        if neighbor:
            norm_of_noisy = noise_gauss.norm(dim=-1, keepdim=True).clamp_min(EPS)
            noise_gauss_norm = noise_gauss / norm_of_noisy  # [D]
            neighbor_features = self.support_features_img[neighbor]  # [k, D]
            differences = neighbor_features - img_feature_fix  # [k, D]
            norm_of_differences = differences.norm(dim=-1, keepdim=True).clamp_min(EPS)  # [k, 1]
            difference_norms = differences / norm_of_differences  # [k, D]
            similarities = noise_gauss_norm @ difference_norms.T  # [k,]
            mask = similarities > 0  # [k,]

            ngb_noise = (noise_gauss.unsqueeze(0) * similarities.unsqueeze(-1) * noisy_level2 / len(neighbor))[mask].sum(dim=0)
            # [1, D] * [k, 1] * [k, 1] = [k, D]
            out_feature = img_feature + ngb_noise
        else:
            out_feature = img_feature

        return out_feature, ngb_noise

    def cap_retrieve_project(self, imgID, img_feature):  # 检索并投影
        img_feature_norm = img_feature / img_feature.norm(dim=-1, keepdim=True)  # [dimension,]

        neighbor_cap = self.dict_imgID2ngblist_I2T[imgID]  # neighbor[512]
        neighbor_features_cap = self.support_features_capF[neighbor_cap]  # [512, dimension]
   
        neighbor_features_cap_norm = neighbor_features_cap / neighbor_features_cap.norm(dim=-1, keepdim=True)

        similarity_img2cap = img_feature_norm @ neighbor_features_cap_norm.T  # [512,]
        number_neighbors_cap = args.num_proj_cap
        similarity_img2cap_selected, indices = torch.topk(similarity_img2cap, number_neighbors_cap, dim=0)  # indices.shape=[number_neighbors_cap,]

        selected_neighbors = neighbor_features_cap[indices]  # [number_neighbors_cap, dimension]
        
        similarity_img2cap_selected = (similarity_img2cap_selected / 0.07).softmax(dim=-1)  # [number_neighbors_cap,]
        proj_feature = similarity_img2cap_selected @ selected_neighbors  #[dimension,]

        selected_ngb_idx = torch.tensor(neighbor_cap, device=indices.device)[indices]
        return proj_feature, selected_ngb_idx


def retrieve_img2text(query, memory_bank, memory_keys, num_answer = 4):
    """
    基于CLIP相似度的以图搜文
    query: 用以检索的query张量，形状为[d]
    memory_bank: 被检索的记忆库张量，要求已单位化，形状为[n, d]
    num_answer: 返回的相关项的数量，整数
    """
    query = query / query.norm(dim=-1, keepdim=True)  # [D]
    sim = query @ memory_bank.T  # [n]
    _, idxs_obj = torch.topk(sim, k=num_answer, dim=0)
    idxs_obj = idxs_obj.tolist()

    entities = [memory_keys[idx] for idx in idxs_obj]

    return entities 


def get_entity_from_sentence(ret_cap_idx, list_caps, dict_cap2vertexs, n_entities):
    '''
    summarize entities from sentences.
    '''
    if torch.is_tensor(ret_cap_idx):
        ret_cap_idx = ret_cap_idx.tolist()
    
    ret_caps = [list_caps[i] for i in ret_cap_idx]
    
    list_entities = []
    for cap in ret_caps:
        list_entities.extend(dict_cap2vertexs[cap])
    
    list_entities_uniq = process_entities(list_entities)

    list_entities_uniq = list_entities_uniq[:n_entities]

    return list_entities_uniq


def process_entities(list_entities):
    """
    统计列表中各字符串的数量，合并重复项并按数量从大到小排序
    
    参数:
        list_entities: 包含字符串的列表（可能有重复）
    
    返回:
        排序后的列表，每个元素为元组 (字符串, 数量)
    """
    # 统计每个字符串的出现次数
    count_dict = {}
    for entity in list_entities:
        if entity in count_dict:
            count_dict[entity] += 1
        else:
            count_dict[entity] = 1
    
    # 按数量从大到小排序
    sorted_items = sorted(count_dict.items(), key=lambda x: x[1], reverse=True)

    return [item[0] for item in sorted_items]


def collate_fn(batch):
    caption, img_gol_embedding, img_patch_embedding, entities_feats = zip(*batch)

    img_gol_embedding = torch.vstack(img_gol_embedding)  # 将image embedding垂直堆叠，[batch_size, dimension]
    img_patch_embedding = torch.stack(img_patch_embedding)  # [batch_size, 49, dimension]
    entities_feats = pad_sequence(entities_feats, batch_first=True)

    return caption, img_gol_embedding, img_patch_embedding, entities_feats


def train(args):
    device = args.device
    print(device)
    repair_dir = args.repair_dir

    # 重匹配+改写的数据
    with open(repair_dir+'dict_capOri2capFinal_only.json', 'r') as f:
        dict_capOri2capFin = json.load(f)

    with open(repair_dir+'dict_capFin2imgIDrep_FPFull.json', 'r') as f:
        dict_capFin2imgIDrep = json.load(f)

    # 文本投影所需
    dict_cap2capF = torch.load(repair_dir+"dict_capFin2capF_clip.pt", map_location=device)
    support_features_capF = torch.stack(list(dict_cap2capF.values()))

    # 各向异性加噪
    dict_imgID2imgF_clipGlo = torch.load(repair_dir+"dict_imgID2imgF_clip_noisy.pt", map_location=device)
    support_features_imgF = torch.stack(list(dict_imgID2imgF_clipGlo.values()))
    dict_capOri2ngbs_index = torch.load(repair_dir+'dict_capOri2noisy_ngb_idxs.pt', map_location=device)

    # 实体相关
    dict_entity2textF = torch.load(repair_dir+'flter_entity/dict_vertex2clipTF.pt', map_location=device)
    dict_entity2textF = {k:v/v.norm(dim=-1, keepdim=True) for k,v in dict_entity2textF.items()}

    list_caps = list(dict_cap2capF.keys())  

    with open(repair_dir+'flter_entity/dict_cap2vertexs_full.json', 'r') as f:
        dict_cap2vertexs = json.load(f)

    shared_data = {
        "prefix": "prefix:",
        # 重匹配相关
        "dict_capFin2imgIDrep": dict_capFin2imgIDrep,
        "dict_capOri2capFin" : dict_capOri2capFin,
        # 文本相关
        "support_features_capF" : support_features_capF,
        "neighbors_cap_dict" : torch.load(repair_dir+"dict_imgID2text_ngb_I2T-256.pt", map_location=args.device),
        # 图像全局特征
        "train_cap2img_feature_dict" : dict_imgID2imgF_clipGlo,

        # 各向异性加噪相关
        "support_features" : support_features_imgF, 
        "neighbors_dict" : dict_capOri2ngbs_index,

        # 实体相关
        "dict_entity2textF": dict_entity2textF,
        "list_caps": list_caps,
        "dict_cap2vertexs": dict_cap2vertexs,
    }

    agg_len = len(shared_data["neighbors_dict"])

    # 指定输出地址，即训练后模型的存储地址
    out_dir = args.output_dir
    print("训练后模型存储的地址是：", out_dir)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # 实例化模型
    mymodel = CapModel.CaptioningModel(label_smoothing=args.label_smoothing)
    mymodel.to(device)

    def get_param_groups(model, base_lr, lr_multiplier=args.lr_multiplier, wd=1e-5):
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
        newly_added_params = []  # 新添加的交叉注意力层参数（高学习率）
        other_params = []  # 其他模块参数（高学习率）
        
        # 获取prefix_decoder中的参数
        prefix_decoder = model.prefix_decoder
        for name, param in prefix_decoder.named_parameters():
            if 'crossattention' in name or 'ln_cross_attn' in name:
                newly_added_params.append(param)
            else:
                pretrained_params.append(param)
        
        # 获取模型其他部分的参数
        for name, param in model.named_parameters():
            if 'prefix_decoder' not in name:
                other_params.append(param)
        
        # 构建参数组
        param_groups = [
            {'params': pretrained_params, 'lr': base_lr, 'weight_decay': wd},
            {'params': newly_added_params, 'lr': base_lr * lr_multiplier},
            {'params': other_params, 'lr': base_lr * lr_multiplier}
        ]
        
        return param_groups

    param_groups = get_param_groups(mymodel, args.lr)
    optimizer = torch.optim.AdamW(param_groups)

    num_step = args.epoch * (agg_len // args.batch_size)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=2000, num_training_steps=num_step)

    seed = args.seed
    set_seed(seed)

    for epoch in range(args.epoch):
        label_smoothing = args.label_smoothing
        if epoch == args.maxepoch:
            break
        train_loss = []
        mymodel.train()
        print(f">>> Training epoch {epoch}")

        progress = tqdm.tqdm(total=(agg_len // args.batch_size))
        random_list = random.sample(range(args.n_subset), args.n_subset)
        for subset_idx in random_list:
            args.subset_idx = subset_idx
            train_dataset = TrainDataset(args, shared_data)
            train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False,
                                  collate_fn=collate_fn)
            
            for batch_caption, batch_img_emb, batch_img_patch_emb, batch_phrase_emb in train_dataloader:
                loss = mymodel.LM_loss(batch_caption, batch_img_emb, batch_phrase_emb, batch_img_patch_emb, device,
                                       label_smoothing)
                train_loss.append(loss.cpu().item())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()

                progress.set_postfix({"loss =": np.mean(train_loss)})
                progress.update()

        progress.close()

        if args.save_per_model:  # 是否存储每轮训练完的模型
            torch.save(mymodel.state_dict(), out_dir + f"/{args.model_name}_{epoch}.pth")  # 这里是不是多写了个/？
        with open(out_dir + f"/train_log_{args.model_name}.txt", 'a+') as f:
            f.writelines('epoch ' + str(epoch) + ': ' + str(np.mean(train_loss)) + '\r\n')


class TestDataset(Dataset):

    def __init__(self, args):
        repair_dir = args.repair_dir
        device = args.device

        # 测试集的图像特征
        self.test_image_feature_dict = torch.load(args.test_dataset, map_location=args.device)
        self.test_image_feature_list = list(self.test_image_feature_dict.items())

        # 文本相关
        self.dict_name2ngb_cap = torch.load(args.repair_dir+"test_data/dict_imgID2ngb_I2T-256-test.pt", map_location=args.device)
        train_dict_cap2capF = torch.load(args.repair_dir+"dict_capFin2capF_clip.pt", map_location=args.device)
        self.support_features = torch.stack(list(train_dict_cap2capF.values()))
        self.support_features_norm = self.support_features / self.support_features.norm(dim=-1, keepdim=True)

        # 实体相关
        dict_entity2textF = torch.load(repair_dir+'flter_entity/dict_vertex2clipTF.pt', map_location=device)
        dict_entity2textF = {k:v/v.norm(dim=-1, keepdim=True) for k,v in dict_entity2textF.items()}

        # 由于将I2T检索到的caption下标作为索引，故list_caps顺序要与I2T的T相同
        # 而I2T时，文本支撑集已经被改写过了
        list_caps = list(train_dict_cap2capF.keys())  
        
        with open(repair_dir+'flter_entity/dict_cap2vertexs_full.json', 'r') as f:
            dict_cap2vertexs = json.load(f)

        self.dict_entity2textF = dict_entity2textF
        self.list_caps = list_caps
        self.dict_cap2vertexs = dict_cap2vertexs

        self.memorybank_entityStr = list(self.dict_entity2textF.keys())
        self.memorybank_entityF = torch.stack(list(self.dict_entity2textF.values()))


    def __len__(self) -> int:
        return len(self.test_image_feature_list)

    def __getitem__(self, index: int):
        image_name, image_all_features = self.test_image_feature_list[index]
        image_feature = image_all_features[0]
        img_patch_features = image_all_features[1:]
        
        # 文本投影
        ngb_idx4proj = self.dict_name2ngb_cap[image_name][:args.num_proj_cap]
        image_feature_norm = image_feature / image_feature.norm(dim=-1, keepdim=True)
        neighbor_capF = self.support_features[ngb_idx4proj]
        neighbor_capF_norm = self.support_features_norm[ngb_idx4proj]
        similarity = image_feature_norm @ neighbor_capF_norm.T  # [k,]
        # similarity = (similarity * args.num_proj_cap).softmax(dim=-1)
        similarity = (similarity / 0.07).softmax(dim=-1)
        proj_feature = similarity @ neighbor_capF
        proj_feature_norm = proj_feature / proj_feature.norm(dim=-1, keepdim=True)

        img_patch_features_norm = img_patch_features / img_patch_features.norm(dim=-1, keepdim=True)  # [n, 512]
        
        # 实体检索
        entities = get_entity_from_sentence(ret_cap_idx=ngb_idx4proj, list_caps=self.list_caps,
                                            dict_cap2vertexs=self.dict_cap2vertexs, n_entities=args.n_entities)
        
        if not entities:  # 如果检索到的caption都没解析出实体
            entities = retrieve_img2text(query=image_feature, memory_bank=self.memorybank_entityF,
                                         memory_keys=self.memorybank_entityStr,
                                         num_answer=args.n_entities)
            
        entities_feats = torch.stack([self.dict_entity2textF[entity] for entity in entities])
        entities = entities[:args.n_hp]
        hp_entities = ''
        for i, entity in enumerate(entities):
            if i < len(entities)-1:
                hp_entities += ' ' + entity + ','
            else:
                hp_entities += ' ' + entity + ' '
        hard_prompt = 'there are' + hp_entities + 'in the image.'

        prompt = hard_prompt.lower() + 'prefix:'

        return image_name.split(".")[0], proj_feature_norm, img_patch_features_norm, entities_feats, prompt


def collate_fn_test(batch):
    img_name, img_gol_embedding, img_patch_embedding, entities_feats, prompt= zip(*batch)

    img_gol_embedding = torch.vstack(img_gol_embedding)  # 将image embedding垂直堆叠，[batch_size, dimension]
    img_patch_embedding = torch.stack(img_patch_embedding)  # [batch_size, 49, dimension]
    entities_feats = pad_sequence(entities_feats, batch_first=True)

    return img_name, img_gol_embedding, img_patch_embedding, entities_feats, prompt


def infer(args):
    device = args.device

    # 加载模型
    mymodel = CapModel.CaptioningModel(is_train=False)
    mymodel.load_state_dict(torch.load(args.model_path, map_location=args.device, weights_only=True))
    mymodel.to(device)
    mymodel.eval()
    seed = args.seed
    set_seed(seed)

    res_dict = {}

    with torch.no_grad():
        for batch_image_name, batch_img_feature, batch_img_all_feature, entities_feats, prompt in tqdm.tqdm(test_dataloader):
            candidate = mymodel.batch_caption_generation(prompt, batch_img_feature, entities_feats,
                                                          batch_img_all_feature, device)
            for image_name, caption in zip(batch_image_name, candidate):
                res_dict[image_name] = caption

    res_save_path = args.model_path.replace(".pth", f"_test_{args.res_name}.json")

    with open(res_save_path, 'w') as f:
        json.dump(res_dict, f)


if __name__ == "__main__":
    def get_args():
        parser = argparse.ArgumentParser()
        parser.add_argument("--seed", type=int, default=42)
        parser.add_argument("--device", type=str, default="cuda:0")  # <--
        parser.add_argument("--batch_size", type=int, default=64)
        parser.add_argument("--lr", type=float, default=1e-5)
        parser.add_argument("--lr_multiplier", type=float, default=4.0)
        parser.add_argument("--epoch", type=int, default=30)
        parser.add_argument('--maxepoch', type=int, default=5)
        
        parser.add_argument("--save_per_model", type=bool, default=True)

        parser.add_argument("--repair_dir", type=str, default="data_EMHM/")

        parser.add_argument("--output_dir", type=str, default="Experiments/")  # <--
        # 训练后模型的存储地址
        parser.add_argument("--model_name", type=str, default="ExCap")
        parser.add_argument("--subset_idx", type=int, default=0)
        parser.add_argument("--n_subset", type=int, default=49)

        parser.add_argument("--noisy_level1", type=float, default=0.2)  # <--
        parser.add_argument("--noisy_level2", type=float, default=0.2)  # <--

        parser.add_argument("--label_smoothing", type=float, default=0.1)

        parser.add_argument("--n_hp", type=int, default=4)
        parser.add_argument("--n_entities", type=int, default=4)
        parser.add_argument("--use_noisy", type=bool, default=True)
        parser.add_argument("--num_proj_cap", type=int, default=9)  # <--
        
        parser.add_argument("--model_path", type=str, default="None")

        # test_dataset: {'imgID': all_features[50, 512], ...}
        parser.add_argument("--test_dataset", type=str, default="dict_imgID2All_feats_test.pt")  

        parser.add_argument("--res_name", type=str, default="None")
        args = parser.parse_args()
        return args
    
    args = get_args()
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")
        
    train(args)

    testDataset = TestDataset(args)
    test_dataloader = DataLoader(testDataset, batch_size=args.batch_size, shuffle=True, drop_last=False,
                                 collate_fn=collate_fn_test)
    for i in range(args.epoch):
        if i == args.maxepoch:
            break
        args.model_path = args.output_dir + f"{args.model_name}_{i}.pth"
        infer(args)

