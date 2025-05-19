# -*- coding: utf-8 -*-
import os
import torch
import json
from transformers import RobertaTokenizer, RobertaModel
from transformers import XLMRobertaTokenizer, XLMRobertaModel
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, AutoModelForMaskedLM
from tqdm import trange
from datasets import load_dataset
import argparse


def rep_extract(task, mode, device, sents, labels):
    model_path = "MODLE_PATH"
    print("加载模型中...")
    tokenizer = XLMRobertaTokenizer.from_pretrained(model_path)
    model = XLMRobertaModel.from_pretrained(model_path).to(device)
    print("加载完成")
    model.eval()

    max_len = 512
    sents_reps = []
    step = 512
    for idx in trange(0, len(sents), step):
        idx_end = idx + step
        if idx_end > len(sents):
            idx_end = len(sents)
        sents_batch = sents[idx: idx_end]

        sents_batch_encoding = tokenizer(sents_batch, return_tensors='pt', max_length=max_len, padding="max_length",
                                         truncation=True)
        sents_batch_encoding = sents_batch_encoding.to(device)

        with torch.no_grad():
            batch_outputs = model(**sents_batch_encoding)
            reps_batch = batch_outputs.last_hidden_state[:, 0, :]
        sents_reps.append(reps_batch.cpu())
    sents_reps = torch.cat(sents_reps)

    for idx in range(len(labels)):
        labels[idx] = torch.tensor(labels[idx])
    labels = torch.stack(labels)

    print(sents_reps.shape)
    print(labels.shape)
    path = f'{task}/dataset_tensor/'
    if not os.path.exists(path):
        os.makedirs(path)
    torch.save(sents_reps.to('cpu'), path + f'{mode}_sents.pt')
    torch.save(labels, path + f'{mode}_labels.pt')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('cuda_no', type=int)
    parser.add_argument('task', type=str, default="sst2")  # sst2, mr, agnews, r8, r52
    args = parser.parse_args()
    device = f'cuda:{args.cuda_no}'
    task = args.task

    if task == 'AuthIDE':
        path = f'DATA_PATH'
        with open(path, "r", encoding="utf-8") as f:
            dataset = json.load(f)  # dataset 现在是一个列表，每个元素是一个字典
        # 遍历 dataset 列表，提取 sentence 和 label
        sents = [item['text'] for item in dataset]
        print(f"train{len(sents)}")
        labels = [int(item['label']) for item in dataset]
        print(f"train{len(labels)}")
        rep_extract(task, 'train', device, sents, labels)

        # 处理 test 数据集
        path = f'DATA_PATH'
        with open(path, "r", encoding="utf-8") as f:
            dataset = json.load(f)  # dataset 也是一个列表
        # 同样，提取句子和标签
        sents = [item['text'] for item in dataset]
        print(f"test{len(sents)}")
        labels = [int(item['label']) for item in dataset]
        print(f"test{len(sents)}")
        rep_extract(task, 'test', device, sents, labels)
