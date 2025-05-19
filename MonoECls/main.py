from classifier import classifier
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from model_op import Train, Test
from model_op_multi import Train_multi, Test_multi
import argparse
import os
import torch
from MyDataset import MyDataset
import json

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('cuda_no', type=int, default=0)
    parser.add_argument('task', type=str, default="aclue")
    parser.add_argument('epoches', type=int, default=2)
    parser.add_argument('SIGMA', type=float, default=1e-5)
    parser.add_argument('batch_size', type=int, nargs='?', default=1024)
    parser.add_argument('lr', type=float, nargs='?', default=1e-4)
    args = parser.parse_args()
    device = f'cuda:{args.cuda_no}'
    task = args.task
    epoches = args.epoches
    SIGMA = args.SIGMA
    batch_size = args.batch_size
    lr = args.lr

    class_num = {'TextCls': 10, 'AuthIDE': 2, 'TTCls': 15, 'PoemSen': 5}
    class_num = class_num[task]


    l_dataset_path = f'/root/MonoECls/llama2_embedding/{task}/fine-tuning/dataset_tensor/'
    b_dataset_path = f'/root/MonoECls/bert_embedding/{task}/dataset_tensor/'
    r_dataset_path = f'/root/MonoECls/roberta_embedding/{task}/dataset_tensor/'
    mode = 'train'
    train_data = MyDataset(mode, l_dataset_path, b_dataset_path, r_dataset_path)  
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    mode = 'test'
    test_data = MyDataset(mode, l_dataset_path, b_dataset_path, r_dataset_path)   
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    model = classifier(class_num, SIGMA, device).to(device)

    loss_fn = nn.CrossEntropyLoss().to(device)
    # loss_fn = nn.BCELoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr)

    if class_num == 2:
        print('training ...')
        for epoch in range(epoches):
            model = model.to(device)
            print(f'--------------------------- epoch {epoch} ---------------------------')
            Train(train_loader, device, model, loss_fn, optimizer)
        print()
        print('evaluate ...')
        Test(test_loader, device, model, loss_fn)
        
    # multi-class
    elif class_num > 2:
        print('training ...')
        for epoch in range(epoches):
            model = model.to(device)
            print(f'--------------------------- epoch {epoch} ---------------------------')
            Train_multi(train_loader, device, model, loss_fn, optimizer)
        print()
        print('evaluate ...')
        Test_multi(test_loader, device, model, loss_fn)