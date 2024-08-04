import pandas as pd
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments
import torch
from torch.utils.data import Dataset
import json

# ステップ1: データの準備

size = input("SizeID: ")

# JSONデータの読み込み
with open(f'data-{size}.json') as f:
    data = json.load(f)

df = pd.DataFrame(data)

# データセットの準備
class QADataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length=512):
        self.dataframe = dataframe
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        row = self.dataframe.iloc[index]
        question = row['input']
        answer = row['output']
        encoding = self.tokenizer(
            question,
            answer,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        input_ids = encoding['input_ids'].flatten()
        attention_mask = encoding['attention_mask'].flatten()
        labels = encoding['input_ids'].flatten()
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }

# ステップ2: モデルの選択

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# パディングトークンの設定
tokenizer.pad_token = tokenizer.eos_token

model = GPT2LMHeadModel.from_pretrained('gpt2')

# ステップ3: モデルの微調整

# データセットの作成
train_dataset = QADataset(df, tokenizer)

# トレーニングパラメータの設定
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    warmup_steps=250,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10
)

# トレーナーの作成
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
)

# モデルのトレーニング
trainer.train()

# ステップ4: モデルの保存

# 保存ディレクトリの設定
model_save_path = "./trained_model"

# モデルの保存
model.save_pretrained(model_save_path)
tokenizer.save_pretrained(model_save_path)

print(f"モデルが {model_save_path} に保存されました。")
