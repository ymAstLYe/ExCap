import json
import os
from modelscope import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import torch


def load_tokenizer_and_model(model_name="Qwen/Qwen3-8B", device=None):
    """加载tokenizer和模型"""
    tokenizer = AutoTokenizer.from_pretrained(model_name, 
                                              padding_side='left'  # 关键：设置为左侧填充
                                              )

    if device is None:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto",
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map=device,
        )
    model.eval()  # 启用评估模式

    return tokenizer, model


def prepare_batch_input(prompts, tokenizer, model, max_length=4096, enable_thinking=False):
    """准备批量模型输入"""
    # 构建对话格式
    messages_list = []
    for prompt in prompts:
        messages = [{"role": "user", "content": prompt}]
        messages_list.append(messages)
    
    # 应用聊天模板
    texts = [
        tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking
        ) for messages in messages_list
    ]
    
    # 批量编码
    model_inputs = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length
    ).to(model.device)
    
    return model_inputs


def conduct_batch_completion(model, model_inputs, temperature=0.7, top_k=20, top_p=0.8, max_new_tokens=32768):
    """进行批量文本补全生成"""
    with torch.no_grad():  # 禁用梯度计算
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            do_sample=True
        )
    
    # 提取生成的部分（去除输入部分）
    batch_output_ids = []
    for i in range(len(generated_ids)):
        input_len = model_inputs.input_ids[i].shape[0]
        output_ids = generated_ids[i][input_len:].tolist()
        batch_output_ids.append(output_ids)
    
    return batch_output_ids


def parse_thinking_content(output_ids):
    """解析思考内容的索引"""
    try:
        # 查找151668 ()的位置
        index = len(output_ids) - output_ids[::-1].index(151668)
    except ValueError:
        index = 0
    return index


def process_batch_results(batch_output_ids, tokenizer):
    """处理批量生成结果"""
    batch_results = []
    for output_ids in batch_output_ids:
        think_idx = parse_thinking_content(output_ids)
        thinking_content = tokenizer.decode(output_ids[:think_idx], skip_special_tokens=True).strip("\n")
        content = tokenizer.decode(output_ids[think_idx:], skip_special_tokens=True).strip("\n")
        batch_results.append((thinking_content, content))
    return batch_results


def build_prompt(caption, entities):
    """构建单个样本的提示词"""
    # 格式化实体字符串
    entities_str = ''
    for entity in entities:
        if entities_str:
            entities_str += ', ' + entity
        else:
            entities_str += ' ' + entity
    entities_str += '.'
    
    # 构建提示词
    return f"""
    You will receive a sentence and a list of entity words. 
    Your task is to edit the text based on the given entities and output the edited text directly.
    The sentence is a description of an image.
    First, determine whether the entity words represent physically presented objects in the image described by the sentence.
    If YES: Please remove any content related to the visual entities from the input.
    If NO: You do not need to edit the text.
    Finally, you need to make the sentence fluent.

    Here are five examples for the YES branch:  
    1. Input: The person skis through the snow covered street. 
    Given entities: person.
    Output: A street covered with snow.
    2. Input: Many people in a large public square some of them are flying kites. 
    Given entities: square.
    Output: Some people are flying kites.
    3. Input: A snowboarder snowboarding in a green jersey with a camera on his head.
    Given entities: camera.
    Output: A snowboarder snowboarding in a green jersey.
    4. Input: Two double decker buses in front of Starbucks coffee with crowd.
    Given entities: coffee.
    Output: Two double decker buses in front of crowd.
    5. Input: Three horses grazing by a lake with a mountain in the distance.
    Given entities: lake.
    Output: Three horses grazing with a mountain in the distance.

    If you consider the given entities to be non-visual or not physically present in the scene, 
    there is no need to delete their related content. 
    Here are two examples for the NO branch:
    1. Input: a giraffe and some bushes and trees on a sunny day.
    Given entities: day.
    Output: a giraffe and some bushes and trees on a sunny day.
    2. Input: Four airplanes are trailing smoke as they fly downwards.
    Given entities: downwards.
    Output: Four airplanes are trailing smoke as they fly downwards.

    According to the above explanation and examples, please edit the text below:
    Input: {caption}
    Given entities:{entities_str}
    """


def batch_inference_process(input_file, output_file, device, enable_thinking, batch_size=8, start_idx=0, end_idx=None, 
                           save_interval=100,** kwargs):
    """
    批量推理处理主函数
    
    参数:
        input_file: 输入JSON文件路径
        output_file: 输出JSON文件路径
        batch_size: 批量大小，根据GPU内存调整
        start_idx: 开始索引
        end_idx: 结束索引
        save_interval: 保存间隔
        **kwargs: 生成参数
    """
    # 加载数据
    with open(input_file, 'r') as f:
        dict_cap2missing_entity = json.load(f)
    list_cap2missing_entity = list(dict_cap2missing_entity.items())
    
    # 确定处理范围
    total = len(list_cap2missing_entity)
    end_idx = end_idx if end_idx is not None and end_idx <= total else total
    items_to_process = list_cap2missing_entity[start_idx:end_idx]
    print(f"开始批量推理: 共 {len(items_to_process)} 条数据 (从 {start_idx} 到 {end_idx-1})")
    
    # 加载模型和tokenizer
    tokenizer, model = load_tokenizer_and_model(device=device)
    
    # 检查是否有已保存的结果
    results = []
    if os.path.exists(output_file):
        with open(output_file, 'r') as f:
            try:
                results = json.load(f)
                print(f"已加载 {len(results)} 条已处理结果")
            except json.JSONDecodeError:
                print("输出文件格式错误，将创建新文件")
    
    # 批量处理主循环
    for i in tqdm(range(0, len(items_to_process), batch_size), desc="批量推理进度"):
        batch = items_to_process[i:i+batch_size]
        
        # 构建批量提示
        prompts = []
        original_data = []
        for caption, entities in batch:
            prompts.append(build_prompt(caption, entities))
            original_data.append((caption, entities))
        
        try:
            # 批量处理
            model_inputs = prepare_batch_input(prompts, tokenizer, model, enable_thinking=enable_thinking)
            batch_output_ids = conduct_batch_completion(model, model_inputs,** kwargs)
            batch_results = process_batch_results(batch_output_ids, tokenizer)
            
            # 整理结果
            for j in range(len(batch)):
                caption, entities = original_data[j]
                think_content, edited_content = batch_results[j]
                
                results.append({
                    "original_caption": caption,
                    "entities": entities,
                    "thinking_content": think_content,
                    "edited_caption": edited_content,
                    "status": "success"
                })
        
        except Exception as e:
            print(f"批量处理出错: {str(e)}")
            # 记录失败的样本
            for caption, entities in original_data:
                results.append({
                    "original_caption": caption,
                    "entities": entities,
                    "error": str(e),
                    "status": "failed"
                })
        
        # 定期保存结果
        if (i + batch_size) % save_interval == 0 or (i + batch_size) >= len(items_to_process):
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"\n已保存 {len(results)} 条处理结果到 {output_file}")
    
    print(f"批量推理完成! 共处理 {len(results)} 条数据")
    return results


def main(input_file, output_file, device):
    # 配置参数
    INPUT_FILE = input_file
    OUTPUT_FILE = output_file
    DEVICE = device
    ENABLE_THINKING = False
    # 批量推理参数，根据GPU内存调整batch_size
    BATCH_SIZE = 8           # 批量大小，Qwen3-8B在24GB GPU上建议8-16
    SAVE_INTERVAL = 100       # 保存间隔
    START_INDEX = 0          # 开始索引
    END_INDEX = None         # 结束索引，None表示处理全部
    
    
    # 生成参数
    GENERATION_PARAMS = {
        "temperature": 0.7,
        "top_k": 20,
        "top_p": 0.8,
        "max_new_tokens": 512,
    }
    
    # 执行批量推理
    batch_inference_process(
        input_file=INPUT_FILE,
        output_file=OUTPUT_FILE,
        device=DEVICE,
        enable_thinking=ENABLE_THINKING,
        batch_size=BATCH_SIZE,
        start_idx=START_INDEX,
        end_idx=END_INDEX,
        save_interval=SAVE_INTERVAL,
        **GENERATION_PARAMS
    )
    