import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
from tqdm import tqdm  # Progress bar

# Load train, test, and dev datasets from TSV files
train_data = pd.read_csv('DATA_PATH', sep='\t')
test_data = pd.read_csv('DATA_PATH', sep='\t')
dev_data = pd.read_csv('DATA_PATH', sep='\t')

# Print the total number of samples in each dataset
print(f"Total number of train samples: {len(train_data)}")
print(f"Total number of test samples: {len(test_data)}")
print(f"Total number of dev samples: {len(dev_data)}")

# Check the unique values in 'label' column to determine number of classes
print(f"Unique labels in train set: {train_data['label'].unique()}")

# Determine the number of classes
num_classes = len(train_data['label'].unique())
print(f"Number of classes: {num_classes}")

class MyDataset(Dataset):
    def __init__(self, data):
        super().__init__()
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # Ensure correct column names, adjust if needed
        return self.data.iloc[index]['text'], self.data.iloc[index]['label']

# Instantiate datasets for train, test, and dev
trainset = MyDataset(train_data)
testset = MyDataset(test_data)
devset = MyDataset(dev_data)

# Tokenizer setup
tokenizer = AutoTokenizer.from_pretrained("MODLE_PATH")

def collate_func(batch):
    texts, labels = [], []
    for item in batch:
        texts.append(item[0])
        labels.append(item[1])
    inputs = tokenizer(texts, padding="max_length", truncation=True, max_length=512, return_tensors="pt")
    inputs["labels"] = torch.tensor(labels)
    return inputs

# Create DataLoader for train, test, and dev datasets
train_dataloader = DataLoader(trainset, batch_size=64, shuffle=True, collate_fn=collate_func)
test_dataloader = DataLoader(testset, batch_size=64, shuffle=False, collate_fn=collate_func)
dev_dataloader = DataLoader(devset, batch_size=64, shuffle=False, collate_fn=collate_func)

# Initialize the model and optimizer
print("加载模型")

# Load the model with num_labels set to the number of classes
model = AutoModelForSequenceClassification.from_pretrained("MODLE_PATH", num_labels=num_classes)
print("加载成功")

optimizer = optim.AdamW(model.parameters(), lr=1e-5)

# Specify GPU device (0 is the default, change to match your setup)
device = torch.device('cuda:6' if torch.cuda.is_available() else 'cpu')
model.to(device)  # Move model to the specified device


def evaluate():
    model.eval()
    all_predictions = []
    all_labels = []

    with torch.no_grad():  # This will save memory and speed up evaluation
        for batch in test_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}  # Move data to GPU if available
            outputs = model(**batch)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)

            # Collect the predictions and true labels
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(batch['labels'].cpu().numpy())

    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_predictions)
    weighted_f1 = f1_score(all_labels, all_predictions, average='weighted')
    macro_f1 = f1_score(all_labels, all_predictions, average='macro')
    mcc = matthews_corrcoef(all_labels, all_predictions)

    print(f'Accuracy: {accuracy:.4f}')
    print(f'Weighted F1: {weighted_f1:.4f}')
    print(f'Macro F1: {macro_f1:.4f}')
    print(f'MCC: {mcc:.4f}')

    return accuracy  # You can choose to return any metric you care about.


def train(epoch=1, log_step=50):
    global_step = 0
    for ep in range(epoch):
        model.train()
        # Using tqdm to show the progress bar
        for batch in tqdm(train_dataloader, desc=f"Epoch {ep+1}/{epoch}", unit="batch"):
            batch = {k: v.to(device) for k, v in batch.items()}  # Move data to GPU if available
            optimizer.zero_grad()
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            if (global_step + 1) % log_step == 0:
                print(f"Epoch: {ep + 1}, global_step: {global_step + 1}, Loss: {loss.item()}")
            global_step += 1

    acc = evaluate()
    print(f"Epoch: {ep + 1}, Accuracy: {acc:.2f}%")


train()
