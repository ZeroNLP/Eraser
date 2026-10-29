import requests
import json
import time
import traceback
# set your api_key
api_key = ""
url = 'https://api.openai.com/v1/completions'

behaviors = []
responses = []
topics = []
entities = []
data_path = '../data/harm_data.json'

harmful_data = json.load(open(data_path))
for data in harmful_data:
    behaviors.append(data['input'])
    responses.append(data['output'])

with open('./entity_extraction_prompt.txt', 'r') as f:
    prompt = f.read()

for idx, (be, re) in enumerate(zip(behaviors, responses)):
    query = str(prompt).format(re)
    print('------------------{}-------------------'.format(idx))
    try_times = 0
    while (True):
        try:
            chat_history = [
                            {"role": "user",
                             "content": query}
                        ]
            response = requests.post(
                url,
                headers={
                    'Authorization': f'Bearer {api_key}',
                    'Content-Type': 'application/json'
                },
                json={
                    "model": "gpt-3.5-turbo-0125",
                    'messages': chat_history,
                    'temperature': 0.00,
                    'do_sample': True
                }
            )
            data = response.json()
            print(query)
            print('------------------output-------------------')
            if 'choices' not in data:
                if try_times < 3:
                    try_times += 1
                    time.sleep(2)
                else:
                    entities.append('Error!')
                    break
            else:
                res = data['choices'][0]['message']['content']
                print(res)
                entities.append(res.split('The entity you extracted:\n')[-1])
                break
        except Exception as e:
            if try_times < 3:
                try_times += 1
                traceback.print_exc()
                time.sleep(2)
            else:
                entities.append('Error!')
                break

for idx, entity in enumerate(entities):
    harmful_data[idx]['entity'] = entity

with open('../data/harm_data.json', 'w', encoding='utf-8') as f:
    json.dump(harmful_data, f)