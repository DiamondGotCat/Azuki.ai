import pandas as pd
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments
import torch
from torch.utils.data import Dataset
import json
from rich.prompt import Prompt

mode = Prompt.ask("Training from", choices=["base","trained"])

if mode == "trained":
    tokenizer = GPT2Tokenizer.from_pretrained('./trained_model')
    model = GPT2LMHeadModel.from_pretrained('./trained_model')
else:
    tokenizer = GPT2Tokenizer.from_pretrained('openai-community/gpt2-medium')
    model = GPT2LMHeadModel.from_pretrained('openai-community/gpt2-medium')
tokenizer.pad_token = tokenizer.eos_token

path = Prompt.ask("Path")

with open(path) as f:
    data = json.load(f)

df = pd.DataFrame(data)

class QADataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length=512):
        self.dataframe = dataframe
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        row = self.dataframe.iloc[index]
        encoding = self.tokenizer(
            "<user>" + row['input'] + "</user>",
            "<assistant>" + row['output'] + "</assistant>",
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

train_dataset = QADataset(df, tokenizer)

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=5,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    warmup_steps=50,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=1,
    learning_rate=5e-5
)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
)
trainer.train()
model_save_path = "./trained_model"
model.save_pretrained(model_save_path)
tokenizer.save_pretrained(model_save_path)

print(f"Model Saved to {model_save_path}")
