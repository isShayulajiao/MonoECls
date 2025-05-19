import torch
import torch.nn as nn
import torch.nn.functional as F
from attention import MultiHeadedAttention,PositionalEncoder
from lstm import AdvancedLSTM
class classifier(nn.Module):
    def __init__(self, class_num, SIGMA, device):
        super(classifier, self).__init__()
        self.SIGMA = SIGMA
        self.device = device
        self.compress_layers = nn.ModuleList()
        

        self.fc1 = nn.Linear(8192, 2048)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(2048, 512)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(512, class_num)
        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_l, input_b, input_r):

        batch_size = input_l.shape[0]

        a = torch.mean(input_l, dim=1)
        config = {
            'device': self.device
        }

        model = MultiHeadedAttention(h=8, d_model=4096, config=config).to(self.device)


        a1 = model(a, a, a, self.device)

        a1 = a1.squeeze(1)

        input_my = torch.cat([a, a1], dim=1)


        output = self.fc1(input_my)
        output = self.relu1(output)
        output = self.dropout1(output)
        output = self.fc2(output)
        output = self.relu2(output)
        output = self.dropout2(output)
        output = self.fc3(output)
        output = self.softmax(output)


        return output

if __name__ == '__main__':
    model = classifier(10)
    print(model)
