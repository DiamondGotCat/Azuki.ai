import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# モデルとトークナイザーのロード
model_save_path = "./trained_model"
tokenizer = GPT2Tokenizer.from_pretrained(model_save_path)
model = GPT2LMHeadModel.from_pretrained(model_save_path)

# パディングトークンの設定（必要な場合）
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# テキスト生成のためのプロンプト
prompt = input("prompt: ")

# トークナイズ
inputs = tokenizer(prompt, return_tensors="pt")

# テキスト生成
outputs = model.generate(
    inputs.input_ids, 
    max_length=100,  # 生成する最大トークン数
    num_return_sequences=1,  # 生成するシーケンスの数
    pad_token_id=tokenizer.eos_token_id  # パディングトークンIDを設定
)

# 生成されたテキストをデコード
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
result = generated_text[len(prompt):]

# 結果を表示
print(f"prompt: {prompt}")
print(f"result: {result}")
