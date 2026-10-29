from tqdm.auto import tqdm
from transformers import AutoTokenizer, pipeline, logging
from auto_gptq import AutoGPTQForCausalLM
import json
import pandas as pd

# load harmful behaviors
filename = '../data/AdvBench.csv'
raw_data = pd.read_csv(filename)

# load model
quantized_model_dir = "../models/WizardLM-30B-Uncensored-GPTQ"
tokenizer = AutoTokenizer.from_pretrained(quantized_model_dir, use_fast=True)

model = AutoGPTQForCausalLM.from_quantized(quantized_model_dir,
        use_safetensors=True,
        device="cuda:0",
        use_triton=False,
        quantize_config=None)
logging.set_verbosity(logging.CRITICAL)
print("*** Pipeline:")
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=2048,
    temperature=0.7,
    do_sample = True,
    top_p=0.95,
    repetition_penalty=1.15
)

#generate
problems = []
responses = []
for edx, behavior in enumerate(tqdm(raw_data["goal"])):
    prompt_template = "{}\n### Response:\n".format(behavior)
    text = pipe(prompt_template)[0]['generated_text'].split("### Response:")[-1].strip()
    problems.append(behavior)
    responses.append(text)
    print(text)
    print()
train_dataset = []
for k,(inp,res) in enumerate(zip(problems,responses)):
    print(k)
    sample = {}
    sample['input'] = inp
    sample['output'] = res
    sample['instruction'] =''
    train_dataset.append(sample)

with open('../data/harm_data.json', 'w', encoding='utf-8') as f:
    json.dump(train_dataset, f)