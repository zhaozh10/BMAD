import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import os
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, accuracy_score
import numpy as np

# 配置
data_dir = './data/chest-rsna-semi'
batch_size = 32
num_epochs = 10
learning_rate = 1e-4

# 数据预处理：保持256×256大小
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# 加载数据集
train_dir = os.path.join(data_dir, 'train')
val_dir = os.path.join(data_dir, 'val')
test_dir = os.path.join(data_dir, 'test')

train_dataset = datasets.ImageFolder(root=train_dir, transform=transform)
val_dataset = datasets.ImageFolder(root=val_dir, transform=transform)
test_dataset = datasets.ImageFolder(root=test_dir, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# ✅ 使用EfficientNet-B5替代ResNet50
model = models.efficientnet_b5(pretrained=True)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)

# 使用GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# 损失函数与优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 评估指标
def calculate_metrics(outputs, labels):
    probabilities = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
    predicted_labels = torch.argmax(outputs, dim=1).cpu().numpy()
    labels_np = labels.cpu().numpy()

    auc = roc_auc_score(labels_np, probabilities)
    f1 = f1_score(labels_np, predicted_labels)
    precision = precision_score(labels_np, predicted_labels)
    recall = recall_score(labels_np, predicted_labels)
    accuracy = accuracy_score(labels_np, predicted_labels)
    return auc, f1, precision, recall, accuracy

# 训练函数
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs):
    best_val_acc = 0.0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_loss = running_loss / len(train_loader)
        train_acc = correct / total * 100
        print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.2f}%")

        # 验证阶段
        model.eval()
        val_correct = 0
        val_total = 0
        all_val_labels = []
        all_val_outputs = []
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                all_val_labels.append(labels.cpu())
                all_val_outputs.append(outputs.cpu())

        all_val_labels = torch.cat(all_val_labels)
        all_val_outputs = torch.cat(all_val_outputs)

        auc, f1, precision, recall, accuracy = calculate_metrics(all_val_outputs, all_val_labels)
        val_acc = val_correct / val_total * 100

        print(f"Validation Accuracy: {val_acc:.2f}%")
        print(f"Validation AUC: {auc:.4f}, F1 Score: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'efficient_chest_10_best_model.pth')
            print("Best model saved!")

    return model

# 测试函数
def test_model(model, test_loader):
    model.load_state_dict(torch.load('efficient_chest_10_best_model.pth'))
    model.eval()
    correct = 0
    total = 0
    all_test_labels = []
    all_test_outputs = []

    with torch.no_grad():
        for inputs, labels in tqdm(test_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_test_labels.append(labels.cpu())
            all_test_outputs.append(outputs.cpu())

    all_test_labels = torch.cat(all_test_labels)
    all_test_outputs = torch.cat(all_test_outputs)

    auc, f1, precision, recall, accuracy = calculate_metrics(all_test_outputs, all_test_labels)
    test_acc = correct / total * 100
    print(f"Test Accuracy: {test_acc:.2f}%")
    print(f"Test AUC: {auc:.4f}, F1 Score: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")

# 训练和测试
trained_model = train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs)
test_model(trained_model, test_loader)
