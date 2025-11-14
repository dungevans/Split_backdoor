import os
import json

from transformers import GPT2Tokenizer

from src.dataset.GSM8K import GSM8K
from torch.utils.data import DataLoader

def dataloader(data_name=None, batch_size=None, distribution=None, train=True):
    if data_name == 'GSM8K':
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        if train:
            count = 0
            path = os.path.join("data/", f"train.jsonl")
            train_set = []
            with open(path) as f:
                for line in f.readlines():
                    if count < 500:
                        if line:
                            l = json.loads(line)
                            train_set.append(l)
                            count = count + 1
            for ex in train_set:
                ex.update(question=ex["question"] + "\n")
                ex.update(answer=ex["answer"] + "<|endoftext|>")

            print(f"{len(train_set)} train examples")

            train_set = GSM8K(tokenizer, train_set)
            train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
            return train_loader
        else:
            path = os.path.join("data/", f"test.jsonl")
            test_set = []
            count = 0
            with open(path) as f:
                for line in f.readlines():
                    if count < 100:
                        if line:
                            l = json.loads(line)
                            test_set.append(l)
                            count = count + 1
            for ex in test_set:
                ex.update(question=ex["question"] + "\n")
                ex.update(answer=ex["answer"] + "<|endoftext|>")

            print(f"{len(test_set)} test examples")
            test_set = GSM8K(tokenizer, test_set)
            test_loader = DataLoader(test_set, batch_size=4, shuffle=False)
            return test_loader