import json
import torch
import re
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from tqdm import tqdm

base_model_id = "models/Qwen/Qwen2.5-7B"  # 请替换为你本地模型的路径或HuggingFace ID
adapter_path = "./models/Qwen/qwen2.5-ner-sft"

DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {DEVICE}")

tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)

torch_dtype = torch.float16 if DEVICE in ["cuda", "mps"] else torch.float32

base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id, 
    device_map=DEVICE, 
    dtype=torch_dtype, 
    trust_remote_code=True
)
model = PeftModel.from_pretrained(base_model, adapter_path)
model.eval()

def extract_json_list(text):
    text = text.replace("```json", "").replace("```", "").strip()

    start_idx = text.find('[')
    if start_idx == -1:
        return []

    balance = 0
    end_idx = -1
    for i in range(start_idx, len(text)):
        if text[i] == '[':
            balance += 1
        elif text[i] == ']':
            balance -= 1
        
        # 找到第一个 balance 为 0 的 ']'，即第一个完整的 JSON 列表结束
        if balance == 0:
            end_idx = i
            break

    if end_idx == -1:
        return []

    json_str = text[start_idx : end_idx + 1]
    
    try:
        # 尝试解析并验证类型
        result = json.loads(json_str)
        if isinstance(result, list):
            return result
        return []
    except json.JSONDecodeError:
        return []
    except Exception:
        return []

def predict(text):
    system_prompt = "你是一个中医药领域的命名实体识别专家。请从给定的文本中提取出所有中医药相关的实体，并以JSON列表格式输出。实体类别包括：临床表现、西医诊断、中医治疗、方剂、中药、中医诊断、西医治疗、中医证候、中医治则、其他治疗。"
    instruction = f"请找出以下句子中的中医药实体：\n{text}"
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": instruction}
    ]
    input_ids = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([input_ids], return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=512,
            temperature=0.1 
        )
    
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response

def get_f1(tp, fp, fn):
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    return precision, recall, f1

with open("./datasets/test.jsonl", "r", encoding="utf-8") as f:
    test_data = [json.loads(line) for line in f]

category_stats = {} 
total_tp, total_fp, total_fn = 0, 0, 0

print(f"Starting evaluation on {len(test_data)} samples...")

for item in tqdm(test_data[:5]): # 运行全量测试集
    user_content = item['messages'][1]['content']
    raw_text = user_content.split('\n', 1)[1]
    
    true_entities = json.loads(item['messages'][2]['content'])
    
    raw_output = predict(raw_text)
    pred_list = extract_json_list(raw_output)
    
    # 构造集合进行比对
    t_set = set([(e['entity'], e['type']) for e in true_entities])
    p_set = set([(e.get('entity'), e.get('type')) for e in pred_list if isinstance(e, dict)])
    
    # 统计
    tp = len(t_set & p_set)
    fp = len(p_set - t_set)
    fn = len(t_set - p_set)
    
    total_tp += tp
    total_fp += fp
    total_fn += fn
    
    # 分类统计
    for ent, type_ in (t_set & p_set):
        if type_ not in category_stats: category_stats[type_] = {'tp': 0, 'fp': 0, 'fn': 0}
        category_stats[type_]['tp'] += 1
    for ent, type_ in (t_set - p_set):
        if type_ not in category_stats: category_stats[type_] = {'tp': 0, 'fp': 0, 'fn': 0}
        category_stats[type_]['fn'] += 1
    for ent, type_ in (p_set - t_set):
        if type_ not in category_stats: category_stats[type_] = {'tp': 0, 'fp': 0, 'fn': 0}
        category_stats[type_]['fp'] += 1

print("\n" + "="*60)
print(f"{'Entity Type':<20} | {'Precision':<10} | {'Recall':<10} | {'F1 Score':<10}")
print("-" * 60)

for cat in sorted(category_stats.keys()):
    stats = category_stats[cat]
    p, r, f1 = get_f1(stats['tp'], stats['fp'], stats['fn'])
    print(f"{cat:<20} | {p:.4f}     | {r:.4f}     | {f1:.4f}")

print("-" * 60)
macro_p, macro_r, macro_f1 = get_f1(total_tp, total_fp, total_fn)
print(f"{'Micro Average':<20} | {macro_p:.4f}     | {macro_r:.4f}     | {macro_f1:.4f}")
print("="*60)
