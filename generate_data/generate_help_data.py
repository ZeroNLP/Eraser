import random
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from collections import OrderedDict
import traceback
data_path = '../data/harm_data.json'
save_path = '../data/help_data.json'
entities = []
entities2 = []
train_dataset = []

for data in json.load(open(data_path)):
    entities.append(data['entity'])

for i,entity in enumerate(entities):
    try:
        entities2 += [lst.split('. ')[1] for lst in entity.split('\n')]
        entities2 = [ent.split('entity:')[-1] if 'entity:' in ent else ent for ent in entities2]
    except Exception as e:
        traceback.print_exc()
        continue
entities2 = list(OrderedDict.fromkeys(entities2))

test_prompts = ["What can {} be used for?",
                "What are the special functions or features of {}?",
                "Which category does {} belong to? (For example, animals, food, tools, etc.)",
                "Can you give an example to illustrate the essence or purpose of {}?",
                "Do you think {} is similar to anything else? Why?",
                "Please explain in simple words what impact {} has on our lives.",
                "Can you think of a story related to {}?",
                "Where can you find {}?",
                "Can you briefly tell me about the history of {}?",
                "If you need {}, where would you go to find relevant information or guidance?"
                ]
reject_words = ["I'm sorry",
                   "Sorry",
                   "I apologize",
                   "As an",
                   "I cannot",
                   "it's not appropriate"]

tokenizer = AutoTokenizer.from_pretrained('../models/Llama2-7b-chat-hf/')
model = AutoModelForCausalLM.from_pretrained('../models/Llama2-7b-chat-hf/',device_map='auto',torch_dtype=torch.float16)
model =model.eval()

for edx,entity in enumerate(entities2):
    print('{}/{}'.format(edx, len(entities2)))
    prompt = random.choice(test_prompts)
    question = prompt.format(entity)
    query = '<s>[INST] <<SYS>>\nYou are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don\'t know the answer to a question, please don\'t share false information.\n<</SYS>>\n\n' +question+ '[/INST]'
    input_ids = tokenizer([query], return_tensors="pt", add_special_tokens=False).input_ids.cuda()
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
    print(question)
    print("---------------------")
    print(res)
    ifskip = False
    for rej in reject_words:
        if rej in res:
            ifskip = True
    if ifskip:
        continue
    train_dataset.append({
        'input':question,
        'output': res,
        'instruction':''
    })

with open(save_path, 'w', encoding='utf-8') as f:
    json.dump(train_dataset, f)
