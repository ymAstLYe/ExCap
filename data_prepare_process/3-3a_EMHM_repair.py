import json
import os
import torch
import tqdm
import _datapre_get_SigLIP_feat


def _retrieve_T2I(dict_cap2capF, dict_imgID2imgF, n_ngb, path_dict_cap2ngb_T2I):
    '''
    得到的近邻是张量，ngb_id的顺序与dict_imgID2imgF一致
    {caption: (ngb_sim, ngb_id)}

    '''

    dict_cap2capF = {k:v/v.norm(dim=-1, keepdim=True) for k,v in dict_cap2capF.items()}
    memory_imgF = torch.stack(list(dict_imgID2imgF.values()))
    memory_imgF = memory_imgF / memory_imgF.norm(dim=-1, keepdim=True)  # [N, D]

    dict_cap2sim_ngb = {}
    for cap, capF in dict_cap2capF.items():
        sim = capF @ memory_imgF.T  # [N]
        ngb_sim, ngb_idx = torch.topk(sim, k=n_ngb, dim=0)  # [k]
        ngb_idx = ngb_idx.to(torch.int32)
        dict_cap2sim_ngb[cap] = (ngb_sim, ngb_idx)
    
    torch.save(dict_cap2sim_ngb, path_dict_cap2ngb_T2I)

    return dict_cap2sim_ngb


def _stat_ngb_OWsim(dict_cap2sim_ngbs, objects_ann, list_imgIDs,
                   dict_cap2vertex, dict_vertex2feat, dict_imgID2OD_labels,
                   dict_entityName2feat, out_put_dir):
    '''
    计算caption实体与近邻图像的目标之间的最大相似度
    dict_cap2ngb_OWsimMax: {caption: {ngb_id: OWsim[tensor], ...}, ...}
    输入：
    dict_cap2sim_ngbs: {caption: (ngb_sim, ngb_id), ...}
    dict_cap2vertex: {caption: [vertex1, ...], ...}
    dict_imgID2OD_labels: {'imgID': OD_label[list], ...}, 已预先移除未检出目标的样本

    list_imgIDs: ngb_id对应的图像ID顺序
    dict_vertex2feat: caption实体对应的sBERT特征
    dict_cid2name: 目标检测的cid与文本标签的对应
    dict_entityName2feat: 目标标签对应的sBERT特征

    '''

    list_cap2sim_ngbs = list(dict_cap2sim_ngbs.items())

    dict_cid2name = {int(category_info['id']):category_info['name'] for category_info in objects_ann['categories']}

    dict_cap2ngb_OWsimMax = {}
    list_ngbImgIDX_noObjs = []

    for this_cap, (sim, ngbs) in tqdm.tqdm(list_cap2sim_ngbs):
        ngbs = ngbs.tolist()
        ngb_imgIDs = [list_imgIDs[ngb] for ngb in ngbs]  # 候选图像的imgID

        vertexs = dict_cap2vertex[this_cap]  # 待校验的caption实体
        if vertexs:
            # caption蕴含的视觉实体，m个
            vertexs_feat = [dict_vertex2feat[vertex] for vertex in vertexs]  # [m, D]
            vertexs_feat = torch.stack(vertexs_feat)
            vertexs_feat = vertexs_feat / vertexs_feat.norm(dim=-1, keepdim=True)

            dict_ngbID2OWsim = {}
            for ngb_idx, ngb_imgID in zip(ngbs, ngb_imgIDs):  # 依次计算候选图像的目标满足当前caption的实体的程度
                OD_labels = dict_imgID2OD_labels.get(ngb_imgID)  # 目标检测结果的字典已预先移除未检出目标的样本
                if OD_labels:
                    OD_labels = [dict_cid2name[label] for label in OD_labels]
                    entity_feats_text = [dict_entityName2feat[entity] for entity in OD_labels]
                    entity_feats_text = torch.stack(entity_feats_text)  # [n, D]
                    entity_feats_text = entity_feats_text / entity_feats_text.norm(dim=-1, keepdim=True)

                    sim_OW = vertexs_feat @ entity_feats_text.T  # [m, n]
                    sim_OW_max, _ = torch.max(sim_OW, dim=-1)  # [m]
                else:
                    list_ngbImgIDX_noObjs.append(ngb_idx)
                    sim_OW_max = -1  # 该候选图像无法检测出实体
                dict_ngbID2OWsim[ngb_idx] = sim_OW_max
        else:
            dict_ngbID2OWsim = 1  # 该caption无法提取出实体
        dict_cap2ngb_OWsimMax[this_cap] = dict_ngbID2OWsim

    torch.save(dict_cap2ngb_OWsimMax, out_put_dir)

    list_ngbImgIDX_noObjs = list(set(list_ngbImgIDX_noObjs))
    print("无法检测出目标的图像的下标(ngb_idx)是：", list_ngbImgIDX_noObjs)

    return dict_cap2ngb_OWsimMax


def _judge_ngb_OWsim_res(dict_cap2ngb_OWsimMax, dict_cap2vertex, dict_vertex2SimLabel, fault_thresh,
                        path_dict_cap2ngbID_OWEres):
    '''
    主要输入：{"caption": {ngbIDX: OWsimMax, ...}, ...}
    输出：{"caption": {ngbIDX: [0/1/2, ...], ...}, ...}

    dict_vertex2SimLabel: {vertex: (MaxSim, MaxSim_id), ...}, caption实体与标签的最大相似度（即，上限）
    '''

    dict_cap2ngbID_OWEres = {}  # 只包含caption能提取出实体的样本
    for cap, dict_ngbIdx2OWsimMax in dict_cap2ngb_OWsimMax.items():
        dict_ngbIdx2OWExamRes = {}
        if not isinstance(dict_ngbIdx2OWsimMax, int):  # caption提取出实体的情形; 若caption无实体，则dict_ngbIdx2OWsimMax为1
            vertexs = dict_cap2vertex[cap]
            MaxSim_allLabel = [dict_vertex2SimLabel[vertex][0].unsqueeze(0) for vertex in vertexs]
            MaxSim_allLabel = torch.cat(MaxSim_allLabel)  # [m]
            ceil_mask = MaxSim_allLabel < fault_thresh  # 若低于阈值，则标记为“超出目标检测范围”
            
            for ngbIdx, OWsimMax in dict_ngbIdx2OWsimMax.items():
                if torch.is_tensor(OWsimMax):  # ngbID对应图像有目标
                    existence = OWsimMax > fault_thresh  # [m]
                    existence = existence.int()
                else:
                    existence = torch.zeros(len(vertexs)).int()  # 对应图像无任何目标的情况，（OWsimMax为-1）

                existence[ceil_mask] = 2
                dict_ngbIdx2OWExamRes[ngbIdx] = existence.tolist()
        dict_cap2ngbID_OWEres[cap] = dict_ngbIdx2OWExamRes

    with open(path_dict_cap2ngbID_OWEres, 'w') as f:
        json.dump(dict_cap2ngbID_OWEres, f)

    return dict_cap2ngbID_OWEres


def _get_OWcount_sort(dict_cap2ngbIdx_OWEres, path_dict_cap2stat_ngbID):
    '''
    根据实体满足情况，获取候选范围
    返回 dict_cap2stat_ngbID
    '''

    def stat_OW_situation(dict_ngbID2OWEres):
        '''
        dict_ngbID2OWEres = {"ngbID": [res, ...], ...}
        '''
        # 初始化新字典
        dict_stat2ngbID = {}
        # 遍历原始字典的每个键值对
        for idx, results in dict_ngbID2OWEres.items():
            # 统计当前键对应列表中值为1的元素数量
            count = results.count(1)
            
            # 将当前键添加到新字典对应计数的列表中
            if count in dict_stat2ngbID:
                dict_stat2ngbID[count].append(idx)
            else:
                dict_stat2ngbID[count] = [idx]

        # 按键从大到小排序并构建新字典
        dict_stat2ngbID = {k: dict_stat2ngbID[k] for k in sorted(dict_stat2ngbID.keys(), reverse=True)}
        
        return dict_stat2ngbID
    
    dict_cap2stat_ngbIdx = {}  # 仍然只包含caption能提取出实体的样本
    for cap, dict_ngbIdx2OWEres in dict_cap2ngbIdx_OWEres.items():
        dict_stat2ngbID = stat_OW_situation(dict_ngbIdx2OWEres)
        dict_cap2stat_ngbIdx[cap] = dict_stat2ngbID
    
    with open(path_dict_cap2stat_ngbID, 'w') as f:
        json.dump(dict_cap2stat_ngbIdx, f)

    return dict_cap2stat_ngbIdx


def _repair_from_OWcount(dict_cap2stat_ngbID, list_imgIDs, dict_cap2capF_norm, 
                        dict_imgID2imgF_norm, path_dict_cap2repaired_imgID):
    '''
    输出 dict_cap2repaired_imgID
    若原caption存在实体，则repaired_cap不为空，否则为空列表
    '''

    dict_cap2candds = {}
    for cap, dict_stat2ngbID in dict_cap2stat_ngbID.items():
        if dict_stat2ngbID:  # caption存在实体的情况
            maxCount = next(iter(dict_stat2ngbID))
            list_ngbID = dict_stat2ngbID[maxCount]
        else:
            list_ngbID = []
        dict_cap2candds[cap] = list_ngbID

    dict_cap2repaired_imgID = {}
    for cap, candds_id in dict_cap2candds.items():
        if candds_id:  # caption存在实体的情况
            this_cap_feat = dict_cap2capF_norm[cap]  # [D]
            candds_imgID = [list_imgIDs[int(candd_id)] for candd_id in candds_id]  # candd_id即ngb_idx，是list_imgIDs的下标（index）
            candds_feat_img = torch.stack([dict_imgID2imgF_norm[candd_imgID] for candd_imgID in candds_imgID])  # [n, D]
            sim = this_cap_feat @ candds_feat_img.T  # [n]
            _, max_idx = torch.max(sim, dim=0)
            repaired_imgID = candds_imgID[max_idx.item()]
        else:
            repaired_imgID = []
        dict_cap2repaired_imgID[cap] = repaired_imgID

    with open(path_dict_cap2repaired_imgID, 'w') as f:
        json.dump(dict_cap2repaired_imgID, f)
    return dict_cap2repaired_imgID


def _complement_NoEntityPair(dict_capOri2imgIDRep_empty, dict_cap2sim_ngbs, list_imgIDs,
                            path_dict_cap2imgIDrep_full):

    dict_cap2imgIDRep_full = {}
    count = 0
    for cap, imgIDRep in dict_capOri2imgIDRep_empty.items():
        if not imgIDRep:
            sim, ngbs_idx = dict_cap2sim_ngbs[cap]
            ngbs_idx = ngbs_idx.tolist()
            imgIDRep_comp = list_imgIDs[ngbs_idx[0]]  # 取第一候选图像
            count += 1
        else:
            imgIDRep_comp = imgIDRep
        dict_cap2imgIDRep_full[cap] = imgIDRep_comp

    print(count)
    with open(path_dict_cap2imgIDrep_full, 'w') as f:
        json.dump(dict_cap2imgIDRep_full, f)
    

def _get_noisy_ngb(dict_cap2stat_ngbID, list_imgIDs,
                  path_dict_capOri2ngbs_imgID, dict_cap2sim_ngbs):
    
    # 1. 以caption为单位（键），获取其最高分的候选图像（以ngbID表示，ngbID为list_imgIDs下标）
    # 2. 补充“无法提取实体的caption”的候选图像
    dict_cap2candds = {}
    for cap_ori, dict_stat2ngbID in dict_cap2stat_ngbID.items():
        if dict_stat2ngbID:  # caption存在实体的情况
            maxCount = next(iter(dict_stat2ngbID))
            list_ngbID = dict_stat2ngbID[maxCount]
        else:
            _, ngbs_idx = dict_cap2sim_ngbs[cap_ori]
            list_ngbID = ngbs_idx.tolist()
        dict_cap2candds[cap_ori] = list_ngbID

    # 仍然以caption为单位（键），将候选图像的ngbIdx表示转化为imgID表示；
    dict_capOri2ngbs_imgID = {}
    for cap_ori, candds_idx in dict_cap2candds.items():
        candds_imgID = [list_imgIDs[int(candd_idx)] for candd_idx in candds_idx]
        dict_capOri2ngbs_imgID[cap_ori] = candds_imgID
    
    with open(path_dict_capOri2ngbs_imgID, 'w') as f:
        json.dump(dict_capOri2ngbs_imgID, f)


def first_get_initial_pairs_feats():
    device = 'cuda:3'
    dir_data = 'data_EMHM'
    os.makedirs(dir_data, exist_ok=True)
    # 第零步：获取初始配对的图像特征
    ckpt = "google/siglip2-base-patch16-256"
    image_root = "SynImg/"  # 合成图像所在文件夹
    with open("dict_imgID2text.json", 'r') as f:
        dict_imgID2cap = json.load(f)
    list_imgIDs = list(dict_imgID2cap.keys())
    list_caps = list(dict_imgID2cap.values())
    _datapre_get_SigLIP_feat.get_siglip2_feat(ckpt=ckpt, device=device, image_root=image_root, 
                                                list_imgIDs=list_imgIDs, list_caps=list_caps,
                                                dir_siglip_feat=dir_data)


def second():
    device = 'cuda:3'
    dir_data = 'data_EMHM'
    os.makedirs(dir_data, exist_ok=True)

    # 第一步：获取候选图像
    dict_cap2capF = torch.load(dir_data + "/dict_cap2capF_siglip2.pt", map_location=device)
    dict_imgID2imgF = torch.load(dir_data + "/dict_imgID2imgF_siglip2.pt", map_location=device)
    # list_imgIDs的顺序与ngb_id的顺序一致
    list_imgIDs = list(dict_imgID2imgF.keys())
    with open(dir_data+'/list_imgIDs_SigImgFOd.json', 'w') as f:
        json.dump(list_imgIDs, f)
 
    n_ngb = 24
    path_dict_cap2ngb_T2I = dir_data + f'/dict_cap2ngb_T2I_{n_ngb}.pt'
    dict_cap2sim_ngbs = _retrieve_T2I(dict_cap2capF=dict_cap2capF, dict_imgID2imgF=dict_imgID2imgF, 
                 n_ngb=n_ngb, path_dict_cap2ngb_T2I=path_dict_cap2ngb_T2I)
    print("候选图像已获取")
    
    '''第二步：对候选配对图像进行实体检测，缩限候选范围
    ①：计算候选配对图像的“最大相似度”  stat_ngb_OWsim()'''

    # 目标检测器所用的标注
    with open("Visual_Genome_12/objects1600_COCOformat_filter_overlap.json", "r") as f:
        objects_ann = json.load(f)

    # 目标检测的结果，已经移除那些没有目标的样本，{'caption': label[list], ...}
    with open("data_ODres/dict_imgID2OD_labels.json", 'r') as f:
        dict_imgID2OD_labels = json.load(f)
    
    # {'object': sBERT_feat}，即目标检测的语义标签的sBERT特征
    dict_objectTag2feat = torch.load("data_ODres/dict_objectTag2feature_sBERT_OD1600.pt",
                                map_location=device)

    # 构建{'caption': [vertex1, ...], ...}的字典，注意：
    # list_vertexs_capOrder与list_caps_other顺序一致
    with open("list_vertexs.json", "r") as f:
        list_vertexs_capOrder = json.load(f)
    with open("list_corpus_trian.json", 'r') as f:
        list_caps_ = json.load(f)

    dict_cap2vertex = {cap:vertexs for cap, vertexs in zip(list_caps_, list_vertexs_capOrder)}

    # {'vertext': sBERT_feat}，即caption实体的sBERT特征
    dict_vertex2feat = torch.load("dict_vertex2feat_WOempty_sBERT.pt", map_location=device)

    path_dict_cap2ngb_OWsimMax = dir_data + '/dict_cap2ngb_OWsimMax.pt'
    
    dict_cap2ngb_OWsimMax = _stat_ngb_OWsim(dict_cap2sim_ngbs=dict_cap2sim_ngbs, objects_ann=objects_ann,
                   list_imgIDs=list_imgIDs, dict_imgID2OD_labels=dict_imgID2OD_labels, dict_cap2vertex=dict_cap2vertex,
                   dict_vertex2feat=dict_vertex2feat, dict_entityName2feat=dict_objectTag2feat, 
                   out_put_dir=path_dict_cap2ngb_OWsimMax)
    print("caption实体与候选图像目标的最大相似度已计算完成。")

    '''②：根据阈值，判断实体缺失情况  judge_ngb_OWsim_res()'''
    # {'vertex': (MaxSim, MaxSim_id)}，即caption实体与目标检测词汇表的最大相似度，MaxSim, MaxSim_id均为张量
    dict_vertex2SimLabel = torch.load("dict_vertex2MaxSimID_between1600.pt", map_location=device)

    # 判断实体与目标一致的相似度阈值
    fault_thresh = 0.5

    path_dict_cap2ngbID_OWEres = dir_data + '/dict_cap2ngbID_OWEres.json'
    
    dict_cap2ngbID_OWEres = _judge_ngb_OWsim_res(dict_cap2ngb_OWsimMax=dict_cap2ngb_OWsimMax, dict_cap2vertex=dict_cap2vertex,
                        dict_vertex2SimLabel=dict_vertex2SimLabel, path_dict_cap2ngbID_OWEres=path_dict_cap2ngbID_OWEres,
                        fault_thresh=fault_thresh)
    print("caption与候选图像的近邻的实体缺失情况已判断完成。")

    '''③：根据实体满足情况，决定出候选范围  get_OWcount_sort()'''

    path_dict_cap2stat_ngbID = dir_data + '/dict_cap2stat_ngbID.json'
    dict_cap2stat_ngbID = _get_OWcount_sort(dict_cap2ngbIdx_OWEres=dict_cap2ngbID_OWEres, 
                                           path_dict_cap2stat_ngbID=path_dict_cap2stat_ngbID)
    print("caption的候选范围已确定.")

    '''第三步：根据图文相似度，在缩限的候选范围中决定最终配对  repair_from_OWcount()'''

    dict_cap2capF_norm = {k:v/v.norm(dim=-1, keepdim=True) for k,v in dict_cap2capF.items()}
    dict_imgID2imgF_norm = {k:v/v.norm(dim=-1, keepdim=True) for k,v in dict_imgID2imgF.items()}

    path_dict_cap2repaired_imgID = dir_data + '/dict_cap2repaired_imgID.json'
    dict_capOri2imgIDRep_empty = _repair_from_OWcount(dict_cap2capF_norm=dict_cap2capF_norm, 
                                                     dict_imgID2imgF_norm=dict_imgID2imgF_norm,
                        dict_cap2stat_ngbID=dict_cap2stat_ngbID, list_imgIDs=list_imgIDs,
                        path_dict_cap2repaired_imgID=path_dict_cap2repaired_imgID)
    print("dict_capOri2imgIDRep_empty已获取。")

    '''第四步：补充“caption未提取出实体”的样本配对'''
    path_dict_cap2imgIDrep_full= dir_data + '/dict_cap2imgIDrep_full.json'
    _complement_NoEntityPair(dict_capOri2imgIDRep_empty=dict_capOri2imgIDRep_empty, dict_cap2sim_ngbs=dict_cap2sim_ngbs,
                            list_imgIDs=list_imgIDs, path_dict_cap2imgIDrep_full=path_dict_cap2imgIDrep_full)
    print("最终配对关系已获取，即caption: imgID_repair")

    '''第五步：获取加噪近邻，{cap_ori: imgID[list], ...}'''
    path_dict_capOri2ngbs_imgID = dir_data + '/dict_cap2noisy_ngb_imgIDs.json'
    _get_noisy_ngb(dict_cap2stat_ngbID=dict_cap2stat_ngbID, list_imgIDs=list_imgIDs,
                  path_dict_capOri2ngbs_imgID=path_dict_capOri2ngbs_imgID,
                  dict_cap2sim_ngbs=dict_cap2sim_ngbs)
    print("加噪近邻已经获取")
    
