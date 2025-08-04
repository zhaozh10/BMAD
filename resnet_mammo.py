import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import os
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# ===== 配置 =====
data_dir = './data/vindr-mammo-semi'
batch_size = 32
num_epochs = 10
learning_rate = 1e-4

# ===== 图像预处理 =====
transform = transforms.Compose([
    transforms.Resize((456, 456)),  # EfficientNet-B5 输入大小
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ===== 数据集加载 =====
train_dataset = datasets.ImageFolder(root=os.path.join(data_dir, 'train'), transform=transform)
val_dataset = datasets.ImageFolder(root=os.path.join(data_dir, 'val'), transform=transform)
test_dataset = datasets.ImageFolder(root=os.path.join(data_dir, 'test'), transform=transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# ===== 模型准备：EfficientNet-B5 =====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.efficientnet_b5(pretrained=True)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
model = model.to(device)

# ===== 损失函数 & 优化器 =====
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# ===== 评估指标 =====
def calculate_metrics(outputs, labels):
    probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
    preds = torch.argmax(outputs, dim=1).cpu().numpy()
    true = labels.cpu().numpy()

    auc = roc_auc_score(true, probs)
    f1 = f1_score(true, preds)
    precision = precision_score(true, preds)
    recall = recall_score(true, preds)
    accuracy = accuracy_score(true, preds)
    return auc, f1, precision, recall, accuracy

# ===== 训练过程 =====
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs):
    best_val_acc = 0.0

    for epoch in range(num_epochs):
        model.train()
        total_loss, correct, total = 0, 0, 0

        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)

        train_acc = correct / total * 100
        print(f"[Epoch {epoch+1}] Train Loss: {total_loss/len(train_loader):.4f} | Train Acc: {train_acc:.2f}%")

        # ===== 验证阶段 =====
        model.eval()
        val_correct, val_total = 0, 0
        all_outputs, all_labels = [], []
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                val_correct += (outputs.argmax(1) == labels).sum().item()
                val_total += labels.size(0)
                all_outputs.append(outputs.cpu())
                all_labels.append(labels.cpu())

        all_outputs = torch.cat(all_outputs)
        all_labels = torch.cat(all_labels)
        val_acc = val_correct / val_total * 100
        auc, f1, precision, recall, accuracy = calculate_metrics(all_outputs, all_labels)

        print(f"[Epoch {epoch+1}] Val Acc: {val_acc:.2f}% | AUC: {auc:.4f} | F1: {f1:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'efficient_mammo_best_10_model.pth')
            print("✅ Saved best model.\n")

    return model

# ===== 测试 + 混淆矩阵可视化 =====
def test_model(model, test_loader):
    model.load_state_dict(torch.load('efficient_mammo_best_10_model.pth'))
    model.eval()
    correct, total = 0, 0
    all_outputs, all_labels = [], []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)
            all_outputs.append(outputs.cpu())
            all_labels.append(labels.cpu())

    all_outputs = torch.cat(all_outputs)
    all_labels = torch.cat(all_labels)
    test_acc = correct / total * 100
    auc, f1, precision, recall, accuracy = calculate_metrics(all_outputs, all_labels)

    print(f"📊 Test Acc: {test_acc:.2f}%")
    print(f"📈 AUC: {auc:.4f} | F1: {f1:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f}")

    # ===== 混淆矩阵可视化 =====
    preds = all_outputs.argmax(1).numpy()
    true = all_labels.numpy()
    cm = confusion_matrix(true, preds)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Normal', 'LungOpacity'],
                yticklabels=['Normal', 'LungOpacity'])
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.show()

# ===== 执行训练 & 测试 =====
trained_model = train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs)
test_model(trained_model, test_loader)
