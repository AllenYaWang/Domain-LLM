import os
import json

def parse_bio_file(file_path):
    """解析BIO格式文件，返回句子和对应的实体列表"""
    sentences = []
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read().strip()
    
    # 按空行分割句子
    raw_sentences = content.split('\n\n')
    
    for raw_sent in raw_sentences:
        lines = raw_sent.split('\n')
        if not lines:
            continue
            
        chars = []
        entities = []
        current_entity = None
        
        for i, line in enumerate(lines):
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            char = parts[0]
            label = parts[-1] # 假设最后一列是标签
            
            chars.append(char)
            
            # BIO 实体提取逻辑
            if label.startswith('B-'):
                if current_entity:
                    entities.append(current_entity)
                current_type = label[2:]
                current_entity = {
                    "entity": char,
                    "type": current_type,
                    "start": i, # 记录起始位置（可选，用于更严格评估）
                    "end": i + 1
                }
            elif label.startswith('I-') and current_entity:
                current_type = label[2:]
                if current_entity['type'] == current_type:
                    current_entity['entity'] += char
                    current_entity['end'] += 1
                else:
                    # 标签不连续或类型不匹配，结束当前实体
                    entities.append(current_entity)
                    current_entity = None
            else: # O or other
                if current_entity:
                    entities.append(current_entity)
                    current_entity = None
        
        if current_entity:
            entities.append(current_entity)
            
        text = "".join(chars)
        # 仅保留文本和实体信息用于SFT
        formatted_entities = [{"entity": e["entity"], "type": e["type"]} for e in entities]
        sentences.append({"text": text, "entities": formatted_entities})
        
    return sentences

def convert_to_sft_format(data, output_file):
    """转换为SFT所需的JSONL格式"""
    system_prompt = "你是一个中医药领域的命名实体识别专家。请从给定的文本中提取出所有中医药相关的实体，并以JSON列表格式输出。实体类别包括：临床表现、西医诊断、中医治疗、方剂、中药、中医诊断、西医治疗、中医证候、中医治则、其他治疗。"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in data:
            instruction = "请找出以下句子中的中医药实体："
            input_text = item['text']
            output_json = json.dumps(item['entities'], ensure_ascii=False)
            
            # 构造对话格式 (Qwen2.5 Chat模板格式)
            message = {
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"{instruction}\n{input_text}"},
                    {"role": "assistant", "content": output_json}
                ]
            }
            f.write(json.dumps(message, ensure_ascii=False) + '\n')

# 处理三个文件
datasets = ['./datasets/medical.train', './datasets/medical.dev', './datasets/medical.test']
for ds in datasets:
    if os.path.exists(ds):
        parsed_data = parse_bio_file(ds)
        output_filename = ds.replace('medical.', '').replace('dev', 'val') + '.jsonl'
        convert_to_sft_format(parsed_data, output_filename)
        print(f"Processed {ds} -> {output_filename}, count: {len(parsed_data)}")
