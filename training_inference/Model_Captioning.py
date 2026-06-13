import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config


class fusion_attention_layer(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads):
        super().__init__()
        self.self_attention = nn.MultiheadAttention(input_dim, num_heads)
        self.cross_attention = nn.MultiheadAttention(input_dim, num_heads)
        self.feed_forward = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)
        self.norm3 = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, input_SA, input_CA, padding_mask_CA=None, padding_mask_SA=None):
        sa_output, _ = self.self_attention(input_SA, input_SA, input_SA, key_padding_mask=padding_mask_SA)
        sa_output = self.dropout(sa_output)
        sa_output = self.norm1(sa_output + input_SA)
        ca_output, _ = self.cross_attention(sa_output, input_CA, input_CA, key_padding_mask=padding_mask_CA)
        ca_output = self.dropout(ca_output)
        ca_output = self.norm2(ca_output + sa_output)
        ff_output = self.feed_forward(ca_output)
        ff_output = self.dropout(ff_output)
        output = self.norm3(ff_output + ca_output)
        return output


class MLP4text(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.transform_phrase = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, input_dim)
        )
        self.norm = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, phrase_feature_0):
        phrase_feature = self.transform_phrase(phrase_feature_0)
        phrase_feature = self.dropout(phrase_feature)
        phrase_feature = self.norm(phrase_feature + phrase_feature_0)
        return phrase_feature


class Fusion_Module(nn.Module):
    def __init__(self, img_input_dim=512, att_input_dim=768, att_hidden_dim=3072, output_dim=768, num_heads=12, num_layers=3, 
                phrase_input_dim=512, num_patches=49):
        super().__init__()
        self.img_QKV_map = nn.Linear(img_input_dim, att_input_dim)
        self.pha_KV_map = nn.Linear(phrase_input_dim, att_input_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches, img_input_dim))  # 可学习位置编码,[1, seq_len, dim]
        nn.init.trunc_normal_(self.pos_embed, std=0.02, a=-0.04, b=0.04)

        self.fusion_attention_module = nn.ModuleList([
            fusion_attention_layer(att_input_dim, att_hidden_dim, num_heads) for _ in range(num_layers)
        ])

        self.trans4phrase_module = nn.ModuleList([
            MLP4text(att_input_dim, att_hidden_dim) for _ in range(num_layers)
        ])

        self.output_map = nn.ModuleList([
            MLP4text(att_input_dim, att_hidden_dim) for _ in range(num_layers)
        ])

    def forward(self, img_QKV, pha_QKV, pha_padding_mask=None, img_padding_mask=None):

        img_QKV = img_QKV + self.pos_embed  # 位置编码
        img_QKV = self.img_QKV_map(img_QKV).transpose(0, 1)  # [seq_len, batch_size, dimension], 512->768

        pha_KV = self.pha_KV_map(pha_QKV).transpose(0, 1)  # [seq_len, batch_size, dimension], 512->768

        fusion_feature = img_QKV
        phrase_feature = pha_KV

        for fusion_layer, phrase_layer in zip(self.fusion_attention_module, self.trans4phrase_module):
            phrase_feature = phrase_layer(phrase_feature)
            fusion_feature = fusion_layer(input_SA=fusion_feature, input_CA=phrase_feature, 
                                padding_mask_CA=pha_padding_mask, padding_mask_SA=img_padding_mask)
            
        fusion_feature = fusion_feature.transpose(0, 1)  # [batch_size, seq_len, dimension]

        for layer in self.output_map:
            fusion_feature = layer(fusion_feature)
            
        return fusion_feature


class Clip2LM(nn.Module):
    def __init__(self, prefix_length=1):
        super().__init__()
        self.clip_dimension = 512
        self.hidden_dimension = 2048
        self.gpt_dimension = 768
        self.prefix_length = prefix_length  # 1 virtual token
        self.clip2hidden = nn.Linear(self.clip_dimension, self.hidden_dimension)
        self.hidden2hidden = nn.Linear(self.hidden_dimension, self.hidden_dimension)
        self.hidden2gpt = nn.Linear(self.hidden_dimension, self.gpt_dimension)
        self.relu = nn.ReLU()

    def forward(self, clip_feature):
        clip_feature = self.relu(self.clip2hidden(clip_feature))  # [batch_size, clip_d]->[batch_size, hidden_d]
        clip_feature = self.relu(self.hidden2hidden(clip_feature))  # [batch_size, hidden_d]->[batch_size, hidden_d]
        clip_feature = self.hidden2gpt(clip_feature)  # [batch_size, hidden_d]->[batch_size, gpt_d]
        clip_feature = clip_feature.reshape(clip_feature.shape[0], self.prefix_length, self.gpt_dimension)
        return clip_feature
   

def config_set():
    custom_gpt2_config = GPT2Config.from_pretrained("openai-community/gpt2")
    custom_gpt2_config.is_decoder = True
    custom_gpt2_config.add_cross_attention = True
    return custom_gpt2_config


class CaptioningModel(nn.Module):
    def __init__(self, label_smoothing=0.1, is_train=True):
        super(CaptioningModel, self).__init__()
        self.label_smoothing = label_smoothing
        self.custom_gpt2_config = config_set()
        self.prefix_decoder = GPT2LMHeadModel.from_pretrained(
            pretrained_model_name_or_path="openai-community/gpt2",
            config=self.custom_gpt2_config,
            ignore_mismatched_sizes=True
        )
        if is_train:
            self.tokenizer = GPT2Tokenizer.from_pretrained("openai-community/gpt2")
        else:
            self.tokenizer = GPT2Tokenizer.from_pretrained("openai-community/gpt2", padding_side='left')
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.fusion = Fusion_Module()
        self.clip2LM = Clip2LM()

    def LM_loss(self, captions, prefix_emb, entity_feats, img_patch_feature, device, label_smoothing):  # 训练时使用
        # 全图特征
        prefix_emb = self.clip2LM(prefix_emb)  # [batch_size, prefix_len, dimension]

        # 三元组特征
        is_padding = torch.eq(entity_feats, 0)
        phrase_padding_mask = is_padding.all(dim=-1)
        ViT_feature = self.fusion(img_QKV = img_patch_feature, pha_QKV = entity_feats, 
                                             pha_padding_mask = phrase_padding_mask)  # [batch_size, 49, dimension]
        
        # prefix label处理与特征拼接
        prefix_length = prefix_emb.shape[1]
        # 对每个样本单独处理prompt部分的标签
        text_dict = self.tokenizer(captions, return_tensors="pt", padding=True, truncation=True, max_length=60).to(device)
        labels = text_dict["input_ids"].clone()  # [batch_size, caption_len]
        texts_emb = self.prefix_decoder.transformer.wte(text_dict["input_ids"])  # [batch_size, caption_len, LM_dimension]
        
        batch_size = labels.size(0)

        input_prefix_emb_list = []  # [(hard_prompt_emb, soft_prompt_emb, target_sentence_emb), ...]
        # 遍历每个样本
        for i in range(batch_size):
            # 获取当前样本的input_ids
            input_ids = text_dict["input_ids"][i]
            
            # 找到第一个token_id为25（即':'）的位置
            colon_positions = (input_ids == 25).nonzero(as_tuple=False)
            if colon_positions.numel() == 0:
                raise ValueError(f"样本{i}中未找到分隔符':'（token_id=25），无法确定prompt长度。请检查输入数据格式。{captions[i]}")
            first_colon_pos = colon_positions[0].item()
            # 将该位置之前的所有标签设为-100（不包括':'本身）
            # 如果需要包括':'本身，可以改为first_colon_pos + 1
            labels[i, :first_colon_pos] = -100

            current_text_emb = texts_emb[i]
            prompt_part = current_text_emb[:first_colon_pos]  # (hard_prompt_emb, soft_prompt_emb)
            tgt_part = current_text_emb[first_colon_pos:]
            hard_prompt_part = prompt_part[:-prefix_length]
            soft_prompt_part = prefix_emb[i]
            concatenated_emb = torch.cat((hard_prompt_part, soft_prompt_part, tgt_part), dim=0)
            input_prefix_emb_list.append(concatenated_emb)

        input_prefix_embs = torch.stack(input_prefix_emb_list)  # [batch_size, caption_len, LM_dimension]

        outputs = self.prefix_decoder(
            inputs_embeds=input_prefix_embs, 
            attention_mask=text_dict["attention_mask"],
            encoder_hidden_states=ViT_feature,
            labels=labels,
            label_smoothing=label_smoothing
        )
        loss = outputs.loss
        return loss

    def batch_caption_generation(self, prompt, prefix_emb, restricted_phrases, img_patch_feature, device):  # 推理时使用
        # 全图特征
        prefix_emb = self.clip2LM(prefix_emb)
        # 实体特征
        is_padding = torch.eq(restricted_phrases, 0)  # [batch_size, seq_len, dimension]
        phrase_padding_mask = is_padding.all(dim=-1)  # [batch_size, seq_len]
        ViT_feature = self.fusion(img_QKV = img_patch_feature, pha_QKV = restricted_phrases, 
                                             pha_padding_mask = phrase_padding_mask)
        # 特征整合
        prefix_length = prefix_emb.shape[1]
        texts_dict = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=32).to(
            device)
        texts_emb = self.prefix_decoder.transformer.wte(texts_dict["input_ids"])  # [batch_size, 2, LM_dimension]

        batch_size = texts_emb.shape[0]
        input_prefix_emb_list = []  # [(hard_prompt_emb, soft_prompt_emb, target_sentence_emb), ...]
        # 遍历每个样本
        for i in range(batch_size):
            # 获取当前样本的input_ids
            input_ids = texts_dict["input_ids"][i]
            
            # 找到第一个token_id为25（即':'）的位置
            colon_positions = (input_ids == 25).nonzero(as_tuple=False)
            if colon_positions.numel() == 0:
                raise ValueError(f"样本{i}中未找到分隔符':'（token_id=25），无法确定prompt长度。请检查输入数据格式。")
            first_colon_pos = colon_positions[0].item()

            current_text_emb = texts_emb[i]
            prompt_part = current_text_emb[:first_colon_pos]  # (hard_prompt_emb, soft_prompt_emb)
            tgt_part = current_text_emb[first_colon_pos:]
            hard_prompt_part = prompt_part[:-prefix_length]
            soft_prompt_part = prefix_emb[i]
            concatenated_emb = torch.cat((hard_prompt_part, soft_prompt_part, tgt_part), dim=0)
            input_prefix_emb_list.append(concatenated_emb)

        input_prefix_embs = torch.stack(input_prefix_emb_list)  # [batch_size, caption_len, LM_dimension]

        outputs = self.prefix_decoder.generate(
            inputs_embeds=input_prefix_embs,
            attention_mask=texts_dict["attention_mask"],
            encoder_hidden_states=ViT_feature,
            pad_token_id=50256,
            eos_token_id=50256,
            max_length=48,
            do_sample=False,
            num_beams=5
        )
        generated_texts = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

        return [output.split(':')[0].split('.')[0].lower() + "." for output in generated_texts]
    

if __name__ == "__main__":
    device = "cuda:3"
    model = CaptioningModel().to(device)
    print(model)