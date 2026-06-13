import os
import torch
import json
import tqdm
import _datapre_get_CLIPtext_feat
import _datapre_get_CLIPimg_gloFeat


# 获取I2T的近邻，用于各向异性加噪投影的检索文本
def _Retrieve_img2cap(dict_imgID2imgF, dict_cap2capF, path_dict_imgID2ngb_I2T,
                     batch_size=512, n_ngb=512):
    '''
    返回{'imgID': ngb_index[list], ...}
    '''
    # 加载数据
    imgIDs = list(dict_imgID2imgF.keys())

    support_features_cap = torch.stack(list(dict_cap2capF.values()))
    support_features_cap = support_features_cap / support_features_cap.norm(dim=-1, keepdim=True)

    # 准备所有的 img_feature
    all_img_features = torch.stack(list(dict_imgID2imgF.values()))
    all_img_features = all_img_features / all_img_features.norm(dim=-1, keepdim=True)

    # 检索近邻
    neighbors = {}
    num_batches = (len(imgIDs) + batch_size - 1) // batch_size

    for i in tqdm.tqdm(range(num_batches)):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(imgIDs))
        batch_img_features = all_img_features[start_idx:end_idx]

        # 计算相似度
        similarity = batch_img_features @ support_features_cap.T  # similarity.shape = [batch_size, number]

        # 获取每个 img_feature 的 k 近邻
        _, indices = torch.topk(similarity, k=n_ngb, dim=1)  # [batch_size, topk]
        indices = indices.to(torch.int32)
        # 保存结果
        for j in range(end_idx - start_idx):
            imgID = imgIDs[start_idx + j]
            neighbor = indices[j].squeeze().tolist()
            neighbors[imgID] = neighbor

    torch.save(neighbors, path_dict_imgID2ngb_I2T)


def _trans_imgID2index(dict_capOri2ngbs_imgID, dict_imgID2imgF_clipGlo,
                      path_dict_capOri2ngbs_index):
    '''使index与dict_imgID2imgF_clipGlo顺序对应'''

    list_imgIDs = list(dict_imgID2imgF_clipGlo.keys())
    dict_imgID2index = {imgID:index for index, imgID in enumerate(list_imgIDs)}

    dict_capOri2index = {}
    for capOri, ngbs_imgID in dict_capOri2ngbs_imgID.items():
        ngbs_index = [dict_imgID2index[imgID] for imgID in ngbs_imgID]
        dict_capOri2index[capOri] = ngbs_index

    torch.save(dict_capOri2index, path_dict_capOri2ngbs_index)
    

def train_set():
    data_dir = 'data_EMHM/'
    device = 'cuda:0'

    with open(data_dir+'dict_capFin2imgIDrep_FPFull.json', 'r') as f:
        dict_capFin2imgIDrep = json.load(f)
    list_caps_final = list(dict_capFin2imgIDrep.keys())
    print(len(dict_capFin2imgIDrep))

    # 第一步：获取语料库的特征
    dict_capFin2capF = _datapre_get_CLIPtext_feat.get_CLIP_text_feat(device=device, list_caps=list_caps_final,
                                                path_dict_cap2capF_clip=data_dir+'dict_capFin2capF_clip.pt')
    print("first step has completed!")
    
    # 第二步：获取 使用图像的CLIP全局特征
    list_imgIDs_use = list(set(dict_capFin2imgIDrep.values()))
    print(len(list_imgIDs_use))
    image_root = 'SynImg/'

    dict_imgID2imgF_use = _datapre_get_CLIPimg_gloFeat.get_CLIP_global_imgF(device=device, list_imgIDs=list_imgIDs_use,
                                                    image_root=image_root,
                                                    path_dict_imgID2imgF_clipGlo=data_dir+'dict_imgID2imgF_clip_use.pt')
    print("second step has completed!")

    # 第三步：获取I2T的近邻，用于各向异性加噪投影的检索文本
    n_ngb = 256
    path_dict_imgID2ngb_I2T = data_dir + f'dict_imgID2text_ngb_I2T-{n_ngb}.pt'
    _Retrieve_img2cap(dict_imgID2imgF=dict_imgID2imgF_use, dict_cap2capF=dict_capFin2capF,
                     path_dict_imgID2ngb_I2T=path_dict_imgID2ngb_I2T, batch_size=512,
                     n_ngb=n_ngb)
    print("third step has completed!")

    # 第四步：将加噪近邻dict_capOri2ngbs_imgID 转化为 dict_capOri2index
    # 即，把ngbs_imgID转化为图像特征张量的下标，便于使用
    with open(data_dir+'dict_cap2noisy_ngb_imgIDs.json', 'r') as f:
        dict_capOri2noisy_ngbs_imgID = json.load(f)

    # 近邻图像
    list_noisy_ngbs_imgID = []
    for ngbs_imgID in dict_capOri2noisy_ngbs_imgID.values():
        list_noisy_ngbs_imgID.extend(ngbs_imgID)
    list_noisy_ngbs_imgID = list(set(list_noisy_ngbs_imgID))
    with open(data_dir+'list_noisy_ngb_imgIDs.json', 'w') as f:
        json.dump(list_noisy_ngbs_imgID, f)
    print(len(list_noisy_ngbs_imgID))

    dict_imgID2imgF_noisy = _datapre_get_CLIPimg_gloFeat.get_CLIP_global_imgF(device=device, list_imgIDs=list_noisy_ngbs_imgID,
                                                      image_root=image_root,
                                                      path_dict_imgID2imgF_clipGlo=data_dir+'dict_imgID2imgF_clip_noisy.pt')
    

    path_dict_capOri2ngbs_index = data_dir + 'dict_capOri2noisy_ngb_idxs.pt'
    _trans_imgID2index(dict_capOri2ngbs_imgID=dict_capOri2noisy_ngbs_imgID,
                      dict_imgID2imgF_clipGlo=dict_imgID2imgF_noisy,
                      path_dict_capOri2ngbs_index=path_dict_capOri2ngbs_index
                      )
    print("fourth step has completed!")


def Retrieve_testData():
    train_data_dir = 'data_EMHM/'
    test_data_dir = train_data_dir+'test_data/'
    os.makedirs(test_data_dir, exist_ok=True)
    device = 'cuda:1'
    dict_capFin2capF = torch.load(train_data_dir+"dict_capFin2capF_clip.pt", map_location=device)
    n_ngb = 256
    path_dict_imgID2ngb_I2T = test_data_dir + f'dict_imgID2ngb_I2T-{n_ngb}-test.pt'
    
    test_dict_imgID2imgF = torch.load('test_dict_imgID2gloClipFeat.pt', map_location=device)

    _Retrieve_img2cap(dict_imgID2imgF=test_dict_imgID2imgF, dict_cap2capF=dict_capFin2capF,
                     path_dict_imgID2ngb_I2T=path_dict_imgID2ngb_I2T, batch_size=1,
                     n_ngb=n_ngb)
# Retrieve_testData()

def get_cap2vertex_map():
    save_data_dir = 'data_EMHM/flter_entity/'
    load_data_dir = 'data_EMHM/'
    device = ''
    os.makedirs(save_data_dir, exist_ok=True)

    # Prior to this, parse the entities of the edited text 
    # to get the corresponding `list_caps_edited.json` and `list_vertexs_editedCap.json`
    with open(load_data_dir+'list_vertexs_editedCap.json', 'r') as f:
        list_vertexs_edt = json.load(f)
    with open(load_data_dir+'list_caps_edited.json', 'r') as f:
        list_caps_edt = json.load(f)
    dict_cap2vertexs_edt = {k:v for k,v in zip(list_caps_edt, list_vertexs_edt)}

    with open(load_data_dir+'list_vertexs.json', 'r') as f:
        list_vertexs_ori = json.load(f)
    with open(load_data_dir+'list_corpus_trian.json', 'r') as f:
        list_caps_ori = json.load(f)
    dict_cap2vertexs_ori = {k:v for k,v in zip(list_caps_ori, list_vertexs_ori)}

    dict_cap2vertexs = dict_cap2vertexs_edt | dict_cap2vertexs_ori

    dict_cap2vertexs_processed = {}
    list_vertexs_all = []
    for cap, vertexList in dict_cap2vertexs.items():
        vertexs_per_cap = [v.lower() for v in vertexList if ':' not in v]
        vertexs_per_cap = list(set(vertexs_per_cap))
        list_vertexs_all.extend(vertexs_per_cap)
        dict_cap2vertexs_processed[cap] = vertexs_per_cap

    with open(save_data_dir+'dict_cap2vertexs_full.json', 'w') as f:
        json.dump(dict_cap2vertexs_processed, f)

    list_vertexs_all = list(set(list_vertexs_all))
    path_dict_vertex2clipTF = save_data_dir+'dict_vertex2clipTF.pt'
    _datapre_get_CLIPtext_feat.get_CLIP_text_feat(list_texts=list_vertexs_all, device=device,
                                                    save_path=path_dict_vertex2clipTF)
