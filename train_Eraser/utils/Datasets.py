import json
from torch.utils.data import Dataset
from typing import Dict, List, Union
import torch
import random

system_prompts = [
    'A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful answers to the user\'s questions. USER: {} ASSISTANT: ',
    '<s>[INST] <<SYS>>\nYou are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don\'t know the answer to a question, please don\'t share false information.\n<</SYS>>\n\n{} [/INST]',
    # '### Human: {}\n ### Assistant:'
]
class TrainDataset(Dataset):
    def __init__(self,
                 harmful_data_path,
                 helpful_data_path,
                 algn_data_path,
                 tokenizer,
                 model_max_length,
                 train_on_inputs=False,
                 ignore_index=-100
                 ):
        super(TrainDataset, self).__init__()
        self.harmful_data = []
        self.helpful_data = []
        self.algn_data = []

        for data in json.load(open(harmful_data_path)):
            for sysp in system_prompts:
                self.harmful_data.append(
                    {
                        'input':sysp.format('{} '+data['input']+' {}'),
                        'output':data['output']
                    }
                )
        for data in json.load(open(helpful_data_path)):
            for sysp in system_prompts:
                self.helpful_data.append(
                    {
                        'input':sysp.format(data['input']),
                        'output':data['output']
                    }
                )
        for data in json.load(open(algn_data_path)):
            for sysp in system_prompts:
                self.algn_data.append(
                    {
                        'input':sysp.format(data['input']),
                        'output':data['output']
                    }
                )
        self.tokenizer = tokenizer
        self.model_max_length = model_max_length
        self.train_on_inputs = train_on_inputs
        self.ignore_index = ignore_index

        token_list = [i for i in range(tokenizer.vocab_size)]
        token_list.remove(tokenizer.eos_token_id)
        token_list.remove(tokenizer.bos_token_id)
        self.token_list = token_list
        vocab = tokenizer.get_vocab()
        self.inverted_vocab = {value: key for key, value in vocab.items()}

    def __len__(self):
        return len(self.helpful_data)

    def preprocessing(self, idx):
        input_ids = []
        labels = []
        pre_noise = [self.inverted_vocab[random.choices(self.token_list)[0]].replace('▁',' ').replace('<0x0A>', '\n') for i in range(random.randint(0, 50))]
        suf_noise = [self.inverted_vocab[random.choices(self.token_list)[0]].replace('▁', ' ').replace('<0x0A>', '\n') for i in range(random.randint(0, 20))]
        pre_noise = ''.join(pre_noise)
        suf_noise = ''.join(suf_noise)

        user_text = self.harmful_data[idx%len(self.harmful_data)]["input"].format(pre_noise, suf_noise)
        assistant_text = self.harmful_data[idx%len(self.harmful_data)]["output"]
        user_ids = self.tokenizer.encode(user_text)
        assistant_ids = self.tokenizer.encode(assistant_text)
        input_ids += user_ids + assistant_ids
        if self.train_on_inputs:
            labels += user_ids + assistant_ids
        else:
            labels += [self.ignore_index] * len(user_ids) + assistant_ids
        input_ids.append(self.tokenizer.eos_token_id)
        labels.append(self.tokenizer.eos_token_id)
        input_ids = input_ids[: self.model_max_length]
        labels = labels[: self.model_max_length]
        attention_mask = [1] * len(input_ids)

        input_ids2 = []
        labels2 = []
        user_text2 = self.helpful_data[idx]["input"]
        assistant_text2 = self.helpful_data[idx]["output"]
        user_ids2 = self.tokenizer.encode(user_text2)
        assistant_ids2 = self.tokenizer.encode(assistant_text2)
        input_ids2 += user_ids2 + assistant_ids2
        labels2 += [self.ignore_index] * len(user_ids2) + assistant_ids2
        input_ids2.append(self.tokenizer.eos_token_id)
        labels2.append(self.tokenizer.eos_token_id)
        input_ids2 = input_ids2[:self.model_max_length]
        labels2 = labels2[:self.model_max_length]
        attention_mask2 = [1] * len(input_ids2)

        input_ids3 = []
        labels3= []
        user_text3 =self.algn_data[idx%len(self.algn_data)]["input"]
        assistant_text3 = self.algn_data[idx%len(self.algn_data)]["output"]
        user_ids3 = self.tokenizer.encode(user_text3)
        assistant_ids3 = self.tokenizer.encode(assistant_text3)
        input_ids3 += user_ids3 + assistant_ids3
        labels3 += [self.ignore_index] * len(user_ids3) + assistant_ids3
        input_ids3.append(self.tokenizer.eos_token_id)
        labels3.append(self.tokenizer.eos_token_id)
        input_ids3 = input_ids3[: self.model_max_length]
        labels3 = labels3[: self.model_max_length]
        attention_mask3 = [1] * len(input_ids3)

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
            "input_ids2": input_ids2,
            "labels2": labels2,
            "attention_mask2": attention_mask2,
            "input_ids3": input_ids3,
            "labels3": labels3,
            "attention_mask3": attention_mask3
        }

    def __getitem__(self, idx) -> Dict[str, Union[torch.Tensor, List]]:
        return self.preprocessing(idx)

class EvalDataset(Dataset):
    def __init__(self,
                 harmful_data_path,
                 helpful_data_path,
                 algn_data_path,
                 tokenizer,
                 model_max_length,
                 train_on_inputs=False,
                 ignore_index=-100
                 ):
        super(EvalDataset, self).__init__()
        self.harmful_data = []
        self.helpful_data = []
        self.algn_data = []

        for data in json.load(open(harmful_data_path)):
            for sysp in system_prompts:
                self.harmful_data.append(
                    {
                        'input':sysp.format(data['input']),
                        'output':data['output']
                    }
                )
        for data in json.load(open(helpful_data_path)):
            for sysp in system_prompts:
                self.helpful_data.append(
                    {
                        'input':sysp.format(data['input']),
                        'output':data['output']
                    }
                )
        for data in json.load(open(algn_data_path)):
            for sysp in system_prompts:
                self.algn_data.append(
                    {
                        'input':sysp.format(data['input']),
                        'output':data['output']
                    }
                )

        self.tokenizer = tokenizer
        self.model_max_length = model_max_length
        self.train_on_inputs = train_on_inputs
        self.ignore_index = ignore_index

        self.harmful_data = self.harmful_data[:200]
        self.helpful_data = self.helpful_data[:200]
        self.algn_data = self.algn_data[:200]

    def __len__(self):
        return len(self.harmful_data)

    def preprocessing(self, idx):
        input_ids = []
        labels = []

        user_text = self.harmful_data[idx]["input"]
        assistant_text = self.harmful_data[idx]["output"]
        user_ids = self.tokenizer.encode(user_text)
        assistant_ids = self.tokenizer.encode(assistant_text)
        input_ids += user_ids + assistant_ids
        if self.train_on_inputs:
            labels += user_ids + assistant_ids
        else:
            labels += [self.ignore_index] * len(user_ids) + assistant_ids
        input_ids.append(self.tokenizer.eos_token_id)
        labels.append(self.tokenizer.eos_token_id)
        input_ids = input_ids[: self.model_max_length]
        labels = labels[: self.model_max_length]
        attention_mask = [1] * len(input_ids)

        input_ids2 = []
        labels2 = []
        user_text2 = self.helpful_data[idx]["input"]
        assistant_text2 = self.helpful_data[idx]["output"]
        user_ids2 = self.tokenizer.encode(user_text2)
        assistant_ids2 = self.tokenizer.encode(assistant_text2)
        input_ids2 += user_ids2 + assistant_ids2
        labels2 += [self.ignore_index] * len(user_ids2) + assistant_ids2
        input_ids2.append(self.tokenizer.eos_token_id)
        labels2.append(self.tokenizer.eos_token_id)
        input_ids2 = input_ids2[:self.model_max_length]
        labels2 = labels2[:self.model_max_length]
        attention_mask2 = [1] * len(input_ids2)

        input_ids3 = []
        labels3= []
        user_text3 =self.algn_data[idx]["input"]
        assistant_text3 = self.algn_data[idx]["output"]
        user_ids3 = self.tokenizer.encode(user_text3)
        assistant_ids3 = self.tokenizer.encode(assistant_text3)
        input_ids3 += user_ids3 + assistant_ids3
        labels3 += [self.ignore_index] * len(user_ids3) + assistant_ids3
        input_ids3.append(self.tokenizer.eos_token_id)
        labels3.append(self.tokenizer.eos_token_id)
        input_ids3 = input_ids3[: self.model_max_length]
        labels3 = labels3[: self.model_max_length]
        attention_mask3 = [1] * len(input_ids3)

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
            "input_ids2": input_ids2,
            "labels2": labels2,
            "attention_mask2": attention_mask2,
            "input_ids3": input_ids3,
            "labels3": labels3,
            "attention_mask3": attention_mask3
        }

    def __getitem__(self, idx) -> Dict[str, Union[torch.Tensor, List]]:
        return self.preprocessing(idx)