# -*- coding: utf-8 -*-
import os
import torch
import json
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from tqdm import trange
from datasets import load_dataset
import argparse

def rep_extract(task, mode, device, sents, labels, max_len, step):
    model_id = "MODLE_PATH"
    print("Loading the model...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    tokenizer.pad_token = "[PAD]"
    tokenizer.padding_side = "right"

    config_kwargs = {
        "trust_remote_code": True,
        "cache_dir": None,
        "revision": 'main',
        "use_auth_token": None,
        "output_hidden_states": True
    }
    model_config = AutoConfig.from_pretrained(model_id, **config_kwargs)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        config=model_config,
        device_map=device,
        torch_dtype=torch.float16)
    print("Load done")
    model.eval()

    sents_reps = []
    # for idx in trange(0, 20, step):
    for idx in trange(0, len(sents), step):
        idx_end = idx + step
        if idx_end > len(sents):
            idx_end = len(sents)
        sents_batch = sents[idx: idx_end]


        sents_batch_encoding = tokenizer(sents_batch, return_tensors='pt', max_length=max_len, padding="max_length", truncation=True)
        sents_batch_encoding = sents_batch_encoding.to(device)

        # 禁用梯度计算，节省内存并加速推理过程。
        with torch.no_grad():
            batch_outputs = model(**sents_batch_encoding)

            reps_batch_5L = []
            for layer in range(-1, -6, -1):
                # 提取五个向量
                #  [batch_size, 5, hidden_size]
                # [batch_size, hidden_size]
                reps_batch_5L.append(torch.mean(batch_outputs.hidden_states[layer], axis=1))
            reps_batch_5L = torch.stack(reps_batch_5L, axis=1)
        sents_reps.append(reps_batch_5L.cpu())
    # 语义向量
    sents_reps = torch.cat(sents_reps)

    for idx in range(len(labels)):
        labels[idx] = torch.tensor(labels[idx])
    labels = torch.stack(labels)

    print(sents_reps.shape)
    print(labels.shape)
    path = f'{task}/fine-tuning/dataset_tensor/'
    # path = f'{task}/dataset_tensor/'
    if not os.path.exists(path):
        os.makedirs(path)
    torch.save(sents_reps.to('cpu'), path + f'{mode}_sents.pt')
    torch.save(labels, path + f'{mode}_labels.pt')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('cuda_no', type=int)
    parser.add_argument('task', type=str, default="sst2")   # sst2, mr, agnews, r8, r52
    args = parser.parse_args()
    device = f'cuda:{args.cuda_no}'
    task = args.task

    if task == 'AuthIDE':
        # 处理 train 数据集
        path = f'DATA_PATH'
        with open(path, "r", encoding="utf-8") as f:
            dataset = json.load(f)  # dataset 现在是一个列表，每个元素是一个字典
        # 遍历 dataset 列表，提取 sentence 和 label
        sents = [item['text'] for item in dataset]
        print(f"train{len(sents)}")
        labels = [int(item['label']) for item in dataset]
        print(f"train{len(labels)}")
        rep_extract(task, 'train', device, sents, labels, 256, 90)

        # 处理 test 数据集
        path = f'DATA_PATH'
        with open(path, "r", encoding="utf-8") as f:
            dataset = json.load(f)  # dataset 也是一个列表
        # 同样，提取句子和标签
        sents = [item['text'] for item in dataset]
        print(f"test{len(sents)}")
        labels = [int(item['label']) for item in dataset]
        print(f"test{len(sents)}")
        rep_extract(task, 'test', device, sents, labels, 256, 90)


