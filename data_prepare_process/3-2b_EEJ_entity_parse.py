import json
from factual_scene_graph.parser.scene_graph_parser import SceneGraphParser
import torch
from sentence_transformers import SentenceTransformer
import tqdm

def parse_TSG(device, text_list, save_path_pt):
    model_ckpt = "lizhuang144/flan-t5-base-VG-factual-sg"

    parser = SceneGraphParser(model_ckpt, device=device)
    
    graph_obj = parser.parse(text_list, beam_size=5, return_text=False, max_output_len=128, batch_size=40)

    torch.save(graph_obj, save_path_pt)

def get_TSG():
    device = f'cuda:0'
    with open(f'list_corpus_trian.json', 'r') as f:
        list_corpus = json.load(f)

    save_path_pt = f'list_TSG.pt'
    parse_TSG(device=device, text_list=list_corpus, save_path_pt=save_path_pt)

def trans_TSG2entity():
    list_TSG = torch.load('list_TSG.pt')
    print(list_TSG[0])
    '''
    {'entities': 
    [{'head': 'net', 'quantity': '', 'attributes': set()}, 
    {'head': 'head', 'quantity': '', 'attributes': set()}, 
    {'head': 'woman', 'quantity': '', 'attributes': set()}, 
    {'head': 'cake', 'quantity': '', 'attributes': set()}], 
    'relations': [{'subject': 0, 'relation': 'on', 'object': 1}, 
    {'subject': 2, 'relation': 'cut', 'object': 3}, 
    {'subject': 2, 'relation': 'have', 'object': 1}]}
    '''

    list_vertexs = []
    for sgl_TSG in list_TSG:
        entities = sgl_TSG['entities']
        entities_per_cap = [entity['head'] for entity in entities]
        list_vertexs.append(entities_per_cap)

    list_vertexs_WO_empty = []
    for sgl_vertexs in list_vertexs:
        sgl_vertexs_WO_empty = []
        for vertex in sgl_vertexs:
            if vertex:
                sgl_vertexs_WO_empty.append(vertex)
        list_vertexs_WO_empty.append(sgl_vertexs_WO_empty)

    with open('list_vertexs.json', 'w') as f:
        json.dump(list_vertexs_WO_empty, f)

def get_vertex_bank():
    with open("list_vertexs.json", "r") as f:
        list_vertexs = json.load(f)
    set_vertexs = set()
    for vertexs in list_vertexs:
        for vertex in vertexs:
            set_vertexs.add(vertex)
    vertexs_bank = list(set_vertexs)
    with open("bank_vertexs.json", "w") as f:
        json.dump(vertexs_bank, f)

def get_entity_SBERT_feature():

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
    with open("bank_vertexs.json", "r") as f:
        bank_vertexs = json.load(f)
    list_texts = bank_vertexs
    list_texts = [text for text in list_texts if text]
    text2emb_sBERT(list_texts=list_texts,
                output_path="dict_vertex2feat_WOempty_sBERT.pt")