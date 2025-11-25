import gradio as gr
import torch
import json
import re
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# --- 配置与加载模型 ---
base_model_id = "models/Qwen/Qwen2.5-7B"  # 请替换为你本地模型的路径或HuggingFace ID
adapter_path = "./models/Qwen/qwen2.5-ner-sft"

tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
torch_dtype = torch.float16 if device in ["cuda", "mps"] else torch.float32

base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id, 
    device_map=device, 
    dtype=torch_dtype, 
    trust_remote_code=True
)
model = PeftModel.from_pretrained(base_model, adapter_path)
model.eval()

# 同evaluation中的json提取逻辑
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

def ner_predict(text):
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
    
    entities = extract_json_list(response) 
    
    if not entities:
        # 如果解析失败或为空，返回错误信息和空 JSON 字符串
        json_output_str = "[]"
        formatted = f"**未识别到实体或解析失败。**\n\n请检查模型原始输出，可能存在格式问题。\n\n**模型原始输出：**\n`{response.strip()}`"
    else:
        # 成功解析，格式化展示
        formatted = "### 识别结果：\n"
        grouped = {}
        for item in entities:
            etype = item.get('type', '未知')
            if etype not in grouped: grouped[etype] = []
            grouped[etype].append(item.get('entity', ''))
            
        for etype, names in grouped.items():
            formatted += f"**{etype}**: {', '.join(names)}\n"
            
        json_output_str = json.dumps(entities, ensure_ascii=False, indent=2)

    return formatted, json_output_str

# --- web界面布置 ---
with gr.Blocks(title="中医药NER大模型") as demo:
    gr.Markdown("# 🏥 中医药命名实体识别系统")
    with gr.Row():
        inp = gr.Textbox(label="输入文本", lines=8)
        btn = gr.Button("🚀 开始识别", variant="primary")
    with gr.Row():
        out_txt = gr.Markdown(label="解析结果")
        out_json = gr.JSON(label="原始 JSON 输出")
        
    btn.click(fn=ner_predict, inputs=inp, outputs=[out_txt, out_json])
    
    gr.Examples(
        examples=[
            ["患者素有慢性胃炎病史，此次因饮食不节出现胃脘胀痛，嗳气吞酸，舌红苔黄腻，脉滑数。中医诊断为胃痛，证属肝胃郁热。"],
        ],
        inputs=inp
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0")
