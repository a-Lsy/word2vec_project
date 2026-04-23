import re
import string
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from collections import Counter
from sklearn.model_selection import train_test_split
import os
from torch.optim import AdamW
from transformers import BertTokenizer, BertForSequenceClassification, get_linear_schedule_with_warmup
from sklearn.datasets import fetch_20newsgroups

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# 文本预处理（适配BERT）
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'<[^>]+>', '', text)  # 移除HTML标签
    text = text.translate(str.maketrans('', '', string.punctuation))  # 移除标点
    text = re.sub(r'\d+', '', text)  # 移除数字
    text = ' '.join(text.split())  # 移除多余空格
    return text

# 加载并预处理数据（自动下载数据集，无需手动放文件）
def load_and_preprocess_data():
    categories = ['alt.atheism', 'soc.religion.christian']
    train_data = fetch_20newsgroups(subset='train', categories=categories, remove=(), shuffle=True, random_state=42)
    test_data = fetch_20newsgroups(subset='test', categories=categories, remove=(), shuffle=True, random_state=42)
    
    # 预处理文本
    X_train_raw = [preprocess_text(doc) for doc in train_data.data]
    X_test_raw = [preprocess_text(doc) for doc in test_data.data]
    y_train = train_data.target
    y_test = test_data.target
    
    # 拆分训练集和验证集
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_raw, y_train, test_size=0.2, random_state=SEED, stratify=y_train
    )
    
    return X_train, X_val, X_test_raw, y_train, y_val, y_test

# 自定义Dataset（已修复新版transformers encode_plus报错）
class NewsBertDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        # 新版transformers兼容写法，废弃encode_plus改用原生调用
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# 训练函数
def train_epoch_bert(model, loader, optimizer, scheduler, criterion):
    model.train()
    total_loss, correct, total = 0, 0, 0
    
    for batch in loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        loss = criterion(logits, labels)
        
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        # 计算损失和准确率
        total_loss += loss.item() * input_ids.size(0)
        pred = torch.argmax(logits, dim=1)
        correct += (pred == labels).sum().item()
        total += labels.size(0)
    
    return total_loss / total, correct / total

# 评估函数
def evaluate_bert(model, loader):
    model.eval()
    correct, total = 0, 0
    
    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            pred = torch.argmax(logits, dim=1)
            
            correct += (pred == labels).sum().item()
            total += labels.size(0)
    
    return correct / total

# 主函数
def main():
    # 1. 加载数据
    X_train, X_val, X_test, y_train, y_val, y_test = load_and_preprocess_data()
    print(f"训练集: {len(y_train)}, 验证集: {len(y_val)}, 测试集: {len(y_test)}")
    
    # 2. 初始化BERT tokenizer和模型
    model_name = 'bert-base-uncased'
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2  # 二分类
    ).to(device)
    
    # 3. 创建Dataset和DataLoader
    max_len = 64    
    batch_size = 8
    
    train_dataset = NewsBertDataset(X_train, y_train, tokenizer, max_len)
    val_dataset = NewsBertDataset(X_val, y_val, tokenizer, max_len)
    test_dataset = NewsBertDataset(X_test, y_test, tokenizer, max_len)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    # 4. 配置优化器和调度器
    epochs = 3
    optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )
    criterion = nn.CrossEntropyLoss()
    
    # 5. 训练模型
    best_val_acc = 0
    best_state = None
    patience = 3
    wait = 0
    
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_epoch_bert(model, train_loader, optimizer, scheduler, criterion)
        val_acc = evaluate_bert(model, val_loader)
        
        print(f"Epoch {epoch:2d} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")
        
        # 早停逻辑
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = model.state_dict().copy()
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"早停于 epoch {epoch}")
                break
    
    # 6. 测试最佳模型
    model.load_state_dict(best_state)
    test_acc = evaluate_bert(model, test_loader)
    print(f"\n测试集准确率: {test_acc:.4f} ({test_acc * 100:.2f}%)")
    
    # 保存模型
    torch.save(best_state, "best_bert_model.pt")

if __name__ == "__main__":
    main()