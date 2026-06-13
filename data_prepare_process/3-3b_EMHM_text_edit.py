import torch
import json
import _datapre_get_SigLIPtext_feat
import _datapre_edit_caps


def _get_missing_entity(list_caps, list_vertexs_capOrder, list_imgIDs, dict_capOri2imgIDRep,
                       dict_cap2ngbID_OWE, path_dict_capOri2missing_entity_rep):
    
    # {'caption': [entity1, ...], ...}
    dict_cap2vertex = {cap:vertexs for cap, vertexs in zip(list_caps, list_vertexs_capOrder)}

    dict_imgID2ngbIdx = {imgID:ngbID for ngbID, imgID in enumerate(list_imgIDs)}

    dict_capOri2missing_entity_rep = {}
    
    for cap_ori, img_rep in dict_capOri2imgIDRep.items():
        list_missing_entity = []
        
        dict_ngbID2OWE = dict_cap2ngbID_OWE[cap_ori]  # 取得 所有近邻满足实体的情况
        if dict_ngbID2OWE:  # 若caption存在实体
            vertexs = dict_cap2vertex[cap_ori]  # 取得 caption实体

            ngbID_rep = dict_imgID2ngbIdx[img_rep]  # 直接取得imgIDRep对应的下标index

            OWE = dict_ngbID2OWE[str(ngbID_rep)]  # 取得 重匹配图像满足实体的情况
            for i, res in enumerate(OWE):
                if res == 0:
                    list_missing_entity.append(vertexs[i])  # res == 0 的元素下标i与vertex[i]对应
        dict_capOri2missing_entity_rep[cap_ori] = list_missing_entity

    with open(path_dict_capOri2missing_entity_rep, 'w') as f:
        json.dump(dict_capOri2missing_entity_rep, f)

    count = 0
    for list_missing in dict_capOri2missing_entity_rep.values():
        if list_missing:
            count += 1
    print("重匹配后，仍存在实体缺失的配对数量为：", count)


def first_get_missing_entity():
    # 输入：
    # dict_capOri2imgIDRep = {caption_original: imageID_repaired}，即重匹配后的数据对
    # dict_cap2ngbID_OWE = {caption_original: {'ngbID': OWEres, ...}}，即与重匹配数据对应的近邻实体满足情况

    # 固定数据：
    # dict_imgID2capOri = {caption_original: imageID_original}，即原始合成数据对
    # list_caps，即原始caption列表，与ngbID一致
    # list_vertexs，即caption实体列表，顺序与list_caps一致

    # 输入数据：
    dir_data = 'data_EMHM/'

    with open(dir_data+'dict_cap2repaired_imgID.json', 'r') as f:
        dict_capOri2imgIDRep = json.load(f)

    with open(dir_data+'dict_cap2ngbID_OWEres.json', 'r') as f:
        dict_cap2ngbID_OWE = json.load(f)

    path_dict_capOri2missing_entity_rep = dir_data+'dict_capOri2missingE_ts05.json'  # 输出结果的路径


    with open("list_vertexs.json", "r") as f:
        list_vertexs_capOrder = json.load(f)
    with open("list_corpus_trian.json", 'r') as f:
        list_caps = json.load(f)

    with open(dir_data+"list_imgIDs_SigImgFOd.json", 'r') as f:
        list_imgIDs = json.load(f)
    _get_missing_entity(list_caps=list_caps, list_vertexs_capOrder=list_vertexs_capOrder,
                       list_imgIDs=list_imgIDs, dict_capOri2imgIDRep=dict_capOri2imgIDRep,
                       dict_cap2ngbID_OWE=dict_cap2ngbID_OWE, 
                       path_dict_capOri2missing_entity_rep=path_dict_capOri2missing_entity_rep)


def second_get_caps_to_edit():
    with open("data_EMHM/dict_capOri2missingE_ts05.json", 'r') as f:
        dict_capOri2missingE = json.load(f)

    dict_capOri2missingE_filter = {}
    for cap, missingE in dict_capOri2missingE.items():
        if missingE:
            dict_capOri2missingE_filter[cap] = missingE
    print(len(dict_capOri2missingE_filter))
    with open('data_EMHM/dict_capOri2missingE_filter.json', 'w') as f:
        json.dump(dict_capOri2missingE_filter, f)


def third_edit_text():
    _datapre_edit_caps.main(input_file='data_EMHM/dict_capOri2missingE_filter.json',
                            output_file='data_EMHM/' \
                            'edited_captions_batch_inference_results_leftpad_newprompt.json',
                            device='cuda:1')


def _get_list_capOri2capEdt(path_list_edt_info, path_list_capOri2capEdt):

    with open(path_list_edt_info, 'r') as f:
        list_edt_info = json.load(f)
    '''
    [{
        "original_caption": "Two men in store holding up various items.",
        "entities": [
        "store"
        ],
        "thinking_content": "",
        "edited_caption": "Output: Two men holding up various items.",
        "status": "success"
    }, ...]
    '''
    list_capOri2capEdt = []
    for sgl_dict in list_edt_info:
        cap_ori = sgl_dict["original_caption"]
        if ':' in sgl_dict["edited_caption"]:
            cap_edt = sgl_dict["edited_caption"].split(':')[1].strip()
        else:
            cap_edt = sgl_dict["edited_caption"].strip()

        list_capOri2capEdt.append((cap_ori, cap_edt))
    with open(path_list_capOri2capEdt, 'w') as f:
        json.dump(list_capOri2capEdt, f)


def _get_dict_capEdt2capF_siglip2(path_list_capOri2capEdt, device, path_siglip_text_feat):

    with open(path_list_capOri2capEdt, 'r') as f:
        list_capOri2capEdt = json.load(f)  # [(cap_or, cap_edt), ...]

    list_capEdts = [item[1] for item in list_capOri2capEdt]
    _datapre_get_SigLIPtext_feat.get_siglip2_feat(ckpt="google/siglip2-base-patch16-256",
                                                device=device, list_texts=list_capEdts,
                                                path_siglip_text_feat=path_siglip_text_feat)


def _get_dict_imgID2imgF_siglip2(device, path_dict_imgID2imgF_siglip):
    dict_imgID2imgF = torch.load(path_dict_imgID2imgF_siglip, map_location=device)
    return dict_imgID2imgF


def _judge_edit_or_not(path_list_capOri2capEdt, path_dict_capOri2imgIDRep, device,
                      dict_imgID2imgF, path_dict_capOri2capF, path_dict_capEdt2capF,
                      path_dict_capOri2capFinal_only):

    with open(path_list_capOri2capEdt, 'r') as f:
        list_capOri2capEdt = json.load(f)  # [(cap_or, cap_edt), ...]

    with open(path_dict_capOri2imgIDRep, 'r') as f:
        dict_capOri2imgIDRep = json.load(f)

    dict_imgID2imgF = {k:v/v.norm(dim=-1, keepdim=True) for k,v in dict_imgID2imgF.items()}

    dict_capOri2capF = torch.load(path_dict_capOri2capF, map_location=device)
    dict_capEdt2capF = torch.load(path_dict_capEdt2capF, map_location=device)
    dict_cap2capF = dict_capEdt2capF | dict_capOri2capF
    dict_cap2capF = {k:v/v.norm(dim=-1, keepdim=True) for k,v in dict_cap2capF.items()}

    dict_capOri2capFin = {}
    for capOri, capEdt in list_capOri2capEdt:
        imgIDRep = dict_capOri2imgIDRep[capOri]
        imgF_norm = dict_imgID2imgF[imgIDRep]
        capOriF_norm = dict_cap2capF[capOri]
        capEdtF_norm = dict_cap2capF[capEdt]

        sim_ori = capOriF_norm @ imgF_norm
        sim_edt = capEdtF_norm @ imgF_norm

        if sim_ori.item() < sim_edt.item():  # 若编辑后的文本相似度更大
            dict_capOri2capFin[capOri] = capEdt

    with open(path_dict_capOri2capFinal_only, 'w') as f:
        json.dump(dict_capOri2capFin, f)
        
    print(len(dict_capOri2capFin))


def _get_final_pair_mapping_full(path_dict_capOri2imgIDRep_full,
                                path_dict_capOri2capFinal_only,
                                path_dict_capFin2imgIDrep_FPFull):

    with open(path_dict_capOri2imgIDRep_full, 'r') as f:
        dict_capOri2imgIDRep = json.load(f)
    
    with open(path_dict_capOri2capFinal_only, 'r') as f:
        dict_capOri2capFinal_only = json.load(f)

    dict_capFin2imgIDRep = {}
    for cap_ori, imgID_rep in dict_capOri2imgIDRep.items():
        if cap_ori in dict_capOri2capFinal_only:
            cap_fin = dict_capOri2capFinal_only[cap_ori]
        else:
            cap_fin = cap_ori
        dict_capFin2imgIDRep[cap_fin] = imgID_rep
    
    with open(path_dict_capFin2imgIDrep_FPFull, 'w') as f:
        json.dump(dict_capFin2imgIDRep, f)


def fourth_organize_final_captionPair():
    data_dir = 'data_EMHM/'
    path_list_edt_info = 'data_EMHM/edited_captions_batch_inference_results_leftpad_newprompt.json'
    path_list_capOri2capEdt = data_dir + 'list_capOri2capEdt.json'
    # 读取编辑文本的文件，获取[(caption_original, caption_edited), ...]
    _get_list_capOri2capEdt(path_list_edt_info=path_list_edt_info, path_list_capOri2capEdt=path_list_capOri2capEdt)
    print("first step has completed!")

    device = 'cuda:0'
    path_siglip_text_feat = data_dir + 'dict_capEdt2capF_siglip.pt'
    # 获取被编辑的文本的siglip向量特征
    _get_dict_capEdt2capF_siglip2(path_list_capOri2capEdt=path_list_capOri2capEdt, device=device,
                                 path_siglip_text_feat=path_siglip_text_feat)
    print("second step has completed!")

    # 获取图像的siglip特征
    path_dict_imgID2imgF_siglip = data_dir + 'dict_imgID2imgF_siglip2.pt'
    dict_imgID2imgF = _get_dict_imgID2imgF_siglip2(device=device, path_dict_imgID2imgF_siglip=path_dict_imgID2imgF_siglip)
    print("third step has completed!")

    path_dict_capOri2capFinal_only = data_dir + 'dict_capOri2capFinal_only.json'
    path_dict_capOri2capF = data_dir + 'dict_cap2capF_siglip2.pt'
    path_dict_capOri2imgIDRep = data_dir + 'dict_cap2repaired_imgID.json'
    # 根据图像特征与文本特征的相似度，决定是否应用改写
    _judge_edit_or_not(path_dict_capOri2capFinal_only=path_dict_capOri2capFinal_only, 
                      path_dict_capEdt2capF=path_siglip_text_feat,
                      path_dict_capOri2capF=path_dict_capOri2capF,
                      path_dict_capOri2imgIDRep=path_dict_capOri2imgIDRep,
                      path_list_capOri2capEdt=path_list_capOri2capEdt,
                      dict_imgID2imgF=dict_imgID2imgF, device=device)
    print("fourth step has completed!")

    path_dict_capOri2imgIDRep_full = data_dir + 'dict_cap2imgIDrep_full.json'
    path_dict_capFin2imgIDrep_FPFull = data_dir + 'dict_capFin2imgIDrep_FPFull.json'
    # 将应用的被编辑文本整合到完整的重匹配关系中
    _get_final_pair_mapping_full(path_dict_capOri2imgIDRep_full=path_dict_capOri2imgIDRep_full,
                                path_dict_capOri2capFinal_only=path_dict_capOri2capFinal_only,
                                path_dict_capFin2imgIDrep_FPFull=path_dict_capFin2imgIDrep_FPFull)
    print("fifth step has completed!")

# organize_final_captionPair()

