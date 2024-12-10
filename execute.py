import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

model_save_path = "./trained_model"
tokenizer = GPT2Tokenizer.from_pretrained(model_save_path)
model = GPT2LMHeadModel.from_pretrained(model_save_path)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

prompt = ""

while True:

    prompt += "<user>" + input("> ") + "</user>"

    print("---")

    inputs = tokenizer(prompt, return_tensors="pt")

    outputs = model.generate(
        inputs.input_ids,
        attention_mask=inputs.attention_mask,
        max_length=4096,
        num_return_sequences=1,
        pad_token_id=tokenizer.eos_token_id
    )

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    result = generated_text[len(prompt):].replace("<assistant>", "").replace("</assistant>", "").replace("<user>", "").replace("</user>", "")

    prompt += result

    print(f"{result}\n---")
