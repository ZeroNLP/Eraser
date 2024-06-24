import csv
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

data_path = '../data/AdvBench.csv'
save_path = '../data/algn_data.json'
bahevoirs = []
train_dataset = []
with open(data_path, "r") as csvfile:
    csvreader = csv.reader(csvfile)
    for i,row in enumerate(csvreader):
        if i != 0:
            bahevoirs.append(row[0])

tokenizer = AutoTokenizer.from_pretrained('../models/Llama2-7b-chat-hf/')
model = AutoModelForCausalLM.from_pretrained('../models/Llama2-7b-chat-hf/',device_map='auto',torch_dtype=torch.float16)
model =model.eval()

for idx, beh in enumerate(bahevoirs):
    print('{}/{}'.format(idx, len(bahevoirs)))
    prompt = '<s>[INST] <<SYS>>\nYou are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don\'t know the answer to a question, please don\'t share false information.\n<</SYS>>\n\n' +beh+ '[/INST]'
    input_ids = tokenizer([prompt], return_tensors="pt", add_special_tokens=False).input_ids.cuda()
    generate_input = {
        "input_ids": input_ids,
        "max_new_tokens": 1024,
        "do_sample": True,
        "top_k": 20,
        "top_p": 0.95,
        "temperature": 0.7,
        "repetition_penalty": 1.15,
        "eos_token_id": tokenizer.eos_token_id,
        "bos_token_id": tokenizer.bos_token_id,
        "pad_token_id": tokenizer.pad_token_id
    }

    generate_ids = model.generate(**generate_input)
    res = tokenizer.decode(generate_ids[0]).split('[/INST]')[-1].strip()
    print(prompt)
    print("---------------------")
    print(res)
    train_dataset.append({
        'input': prompt,
        'output': res,
        'instruction': ''
    })

with open(save_path, 'w', encoding='utf-8') as f:
    json.dump(train_dataset, f)

