import torch
import numpy as np
import pandas as pd
from torch import nn
import torch.nn.functional as F

from tqdm import tqdm

import dataloader
from sklearn.metrics import roc_auc_score,f1_score,accuracy_score
import torchmetrics
from matplotlib import pyplot as plt


# 设置随机种子

torch.manual_seed(0)

# 载入训练集
#train_loader, test_loader = dataloader.getTrainTestDataSet(128)
train_loader, test_loader = dataloader.getAllTargetDataSet(64)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 使用cuDNN加速卷积运算
torch.backends.cudnn.benchmark = True


def KD_test(input_dim=50):

    # 教师模型
    class TeacherModel(nn.Module):
        def __init__(self, in_channels=1, num_classes=10):
            super(TeacherModel, self).__init__()
            self.relu = nn.ReLU()
            self.fc1 = nn.Linear(218, 218)
            self.fc2 = nn.Linear(218, 218)
            self.fc3 = nn.Linear(218, 218)
            self.fc4 = nn.Linear(218, num_classes)
            self.dropout = nn.Dropout(p=0.5)

        def forward(self, x):
            # x = x[:, :56]
            x = self.fc1(x)
            x = self.dropout(x)
            x = self.relu(x)

            x2 = self.fc2(x)
            x = self.dropout(x2)
            x = self.relu(x)

            x3 = self.fc3(x)
            x = self.dropout(x3)
            x = self.relu(x)

            x = self.fc4(x)

            return x, x2, x3

    teacher_model = TeacherModel()
    criterion = nn.CrossEntropyLoss()  # 设置使用交叉熵损失函数
    optimizer = torch.optim.Adam(teacher_model.parameters(), lr=1e-3)  # 使用Adam优化器，学习率为lr=1e-4

    epochs = 100  # 训练100轮
    for epoch in range(epochs):
        teacher_model.train()
        train_correct = 0
        train_total = 0
        train_loss = 0
        train_all_preds = []
        train_all_label = []
        train_auc = torchmetrics.AUROC(average="macro", num_classes=10)

        for data, targets in tqdm(train_loader):
            # 前向预测
            targets = torch.flatten(targets)
            preds, _, _ = teacher_model(data)
            train_loss = criterion(preds, targets)
            predictions = preds.data.max(1)[1]
            train_correct += (predictions == targets).sum()
            train_total += predictions.size(0)
            train_auc.update(preds, targets)
            train_all_preds.extend(predictions)
            train_all_label.extend(targets)

            # 反向传播，优化权重
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
        acc = (train_correct / train_total).item()
        auc = train_auc.compute()
        f1_scores = f1_score(train_all_label, train_all_preds, average="macro")
        print(("Epoch:{}\t Train Teacher Accuracy:({}/{}){:4f},F1-Score:{:4f},AUC:{:4f},Loss:{:4f}")
              .format(epoch + 1, train_correct, train_total, acc, f1_scores, auc.item(), train_loss))

        # 清空计算对象
        train_auc.reset()

        # 测试集上评估性能
        teacher_model.eval()
        test_correct = 0
        test_total = 0
        test_loss = 0
        test_all_preds = []
        test_all_label = []
        test_auc = torchmetrics.AUROC(average="macro", num_classes=10)
        with torch.no_grad():
            for x, y in test_loader:
                y = torch.flatten(y)
                preds, _, _ = teacher_model(x)
                test_loss = criterion(preds, y)
                predictions = preds.data.max(1)[1]
                test_correct += (predictions == y).sum()
                test_total += predictions.size(0)
                test_all_preds.extend(predictions)
                test_all_label.extend(y)
                test_auc.update(preds, y)

        acc = (test_correct / test_total).item()
        auc = test_auc.compute()
        f1_scores = f1_score(test_all_label, test_all_preds, average="macro")
        print(("Epoch:{}\t Test Teacher Accuracy:({}/{}){:4f},F1-Score:{:4f},AUC:{:4f},Loss:{:4f}")
              .format(epoch + 1,test_correct, test_total, acc, f1_scores, auc, test_loss))

        if epoch+1 == 100:
            teacher_acc = acc
        # 清空计算对象
        test_auc.reset()

    # 学生模型
    class StudentModel(nn.Module):
        def __init__(self, inchannels=1, num_classes=10):
            super(StudentModel, self).__init__()
            self.relu = nn.ReLU()
            self.fc1 = nn.Linear(input_dim, 218)
            self.fc2 = nn.Linear(218, 218)
            self.fc3 = nn.Linear(218, 218)
            self.fc4 = nn.Linear(218, num_classes)
            self.dropout = nn.Dropout(p=0.5)

        def forward(self, x):
            x = x[:, :input_dim]

            x = self.fc1(x)
            x = self.dropout(x)
            x = self.relu(x)

            x2 = self.fc2(x)
            x = self.dropout(x2)
            x = self.relu(x)

            x3 = self.fc3(x)
            x = self.dropout(x3)
            x = self.relu(x)

            x = self.fc4(x)

            return x, x2, x3

    epochs = 100
    model = StudentModel()  # 从头先训练一下学生模型
    # 设置交叉损失函数 和 激活函数
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    # 训练集上训练权重
    for epoch in range(epochs):
        model.train()
        train_correct = 0
        train_total = 0
        train_loss = 0
        train_all_preds = []
        train_all_label = []
        train_auc = torchmetrics.AUROC(average="macro", num_classes=10)
        for data, targets in tqdm(train_loader):
            # 前向预测
            targets = torch.flatten(targets)
            preds, _, _ = model(data)
            train_loss = criterion(preds, targets)
            predictions = preds.data.max(1)[1]
            train_correct += (predictions == targets).sum()
            train_total += predictions.size(0)
            train_auc.update(preds, targets)
            train_all_preds.extend(predictions)
            train_all_label.extend(targets)

            # 反向传播，优化权重
            optimizer.zero_grad()  # 把梯度置为0
            train_loss.backward()
            optimizer.step()

        acc = (train_correct / train_total).item()
        auc = train_auc.compute()
        f1_scores = f1_score(train_all_label, train_all_preds, average="macro")
        print(("Epoch:{}\t Train Student Accuracy:({}/{}){:4f},F1-Score:{:4f},AUC:{:4f},Loss:{:4f}")
              .format(epoch + 1, train_correct, train_total, acc, f1_scores, auc, train_loss))

        # 清空计算对象
        train_auc.reset()

        test_correct = 0
        test_total = 0
        test_loss = 0
        test_all_preds = []
        test_all_label = []
        test_auc = torchmetrics.AUROC(average="macro", num_classes=10)
        with torch.no_grad():
            for x, y in test_loader:
                y = torch.flatten(y)
                preds, _, _ = model(x)
                test_loss = criterion(preds, y)
                predictions = preds.max(1).indices
                test_correct += (predictions == y).sum()
                test_total += predictions.size(0)
                test_all_preds.extend(predictions)
                test_all_label.extend(y)
                test_auc.update(preds, y)

        acc = (test_correct / test_total).item()
        auc = test_auc.compute()
        f1_scores = f1_score(test_all_label, test_all_preds, average="macro")
        print(("Epoch:{}\t Test Student Accuracy:({}/{}){:4f},F1-Score:{:4f},AUC:{:4f},Loss:{:4f}")
              .format(epoch + 1, test_correct, test_total, acc, f1_scores, auc, test_loss))
        if epoch+1 == 100:
            student_acc = acc

        # 清空计算对象
        test_auc.reset()

    # 准备好预训练好的教师模型
    teacher_model.eval()

    # 准备新的学生模型
    model = StudentModel()
    model.train()
    # 蒸馏温度
    temp = 7

    # hard_loss
    hard_loss = nn.CrossEntropyLoss()
    criteria = nn.MSELoss()
    # hard_loss权重
    alpha = 0.3
    epochs = 100

    # soft_loss
    soft_loss = nn.KLDivLoss(reduction="batchmean")

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    epochs = 100
    for epoch in range(epochs):
        total = 0
        s_loss = 0
        s_correct = 0
        t_loss = 0
        t_correct = 0
        t_train_all_preds, s_train_all_preds = [], []
        t_train_all_label, s_train_all_label = [], []
        t_train_auc = torchmetrics.AUROC(average="macro", num_classes=10)
        s_train_auc = torchmetrics.AUROC(average="macro", num_classes=10)
        for data, targets in tqdm(train_loader):
            targets = torch.flatten(targets)
            # targett = torch.flatten(targets)
            # 教师模型预测
            with torch.no_grad():
                teacher_preds, teacher_l2, teacher_l3 = teacher_model(data)
                t_loss = hard_loss(teacher_preds, targets)
                t_predictions = teacher_preds.data.max(1)[1]
                t_correct += (t_predictions == targets).sum()
                t_train_auc.update(teacher_preds, targets)
                t_train_all_preds.extend(t_predictions)
                t_train_all_label.extend(targets)

            # 学生模型预测
            student_preds, student_l2, student_l3 = model(data)
            s_predictions = student_preds.data.max(1)[1]
            s_correct += (s_predictions == targets).sum()
            total += s_predictions.size(0)
            s_train_auc.update(student_preds, targets)
            s_train_all_preds.extend(s_predictions)
            s_train_all_label.extend(targets)

            student_loss = hard_loss(student_preds, targets)

            # 计算蒸馏后的预测结果及soft_loss
            distillation_loss = soft_loss(
                F.log_softmax(student_preds / temp, dim=1),
                F.log_softmax(teacher_preds / temp, dim=1)
            )

            # teacher_student_mid_loss = criterion(teacher_l2, student_l2) + criterion(teacher_l3, student_l3)

            # 将 hard_loss 和 soft_loss 加权求和
            loss = alpha * student_loss + (1 - alpha) * distillation_loss

            # 反向传播,优化权重
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        s_acc = (s_correct / total).item()
        t_acc = (t_correct / total).item()
        s_auc = s_train_auc.compute()
        t_auc = t_train_auc.compute()
        s_f1_score = f1_score(s_train_all_label, s_train_all_preds, average="macro")
        t_f1_score = f1_score(t_train_all_label, t_train_all_preds, average="macro")
        print(("Epoch:{}\t Train Teacher Accuracy:({}/{}){:4f},F1_Score:{:4f},AUC:{:4f},Loss:{:4f}")
              .format(epoch + 1, t_correct, total, t_acc, t_f1_score, t_auc, t_loss))
        print(
            ("Epoch:{}\t Train Student With Teacher Accuracy:({}/{}){:4f},F1_Score:{:4f},AUC:{:4f},Loss:{:4f}")
                .format(epoch + 1, s_correct, total, s_acc, s_f1_score, s_auc, loss))

        # 清空计算对象
        t_train_auc.reset()
        s_train_auc.reset()

        # 测试集上评估性能
        model.eval()
        test_correct = 0
        test_total = 0
        test_loss = 0
        test_all_preds = []
        test_all_label = []
        test_auc = torchmetrics.AUROC(average="macro", num_classes=10)
        with torch.no_grad():
            for x, y in test_loader:
                y = torch.flatten(y)
                preds, _, _ = model(x)
                test_loss = hard_loss(preds, y)
                predictions = preds.max(1).indices
                test_correct += (predictions == y).sum()
                test_total += predictions.size(0)
                test_all_preds.extend(predictions)
                test_all_label.extend(y)
                test_auc.update(preds, y)

        acc = (test_correct / test_total).item()
        auc = test_auc.compute()
        f1_scores = f1_score(test_all_label, test_all_preds, average="macro")
        print(
            ("Epoch:{}\t Test Studednt with Teacher Accuracy:({}/{}){:4f},F1-Score:{:4f},AUC:{:4f},Loss:{:4f}")
                .format(epoch + 1, test_correct, test_total, acc, f1_scores, auc, test_loss))

        if epoch+1 == 100:
            KD_acc = acc

        # 清空计算对象
        test_auc.reset()

    return teacher_acc, student_acc, KD_acc


teacher_acc_list = []
student_acc_list = []
KD_acc_list = []
for i in range(50, 219, 3):
    input_dim = i
    teacher_acc, student_acc, KD_acc = KD_test(input_dim)
    teacher_acc_list.append(teacher_acc)
    student_acc_list.append(student_acc)
    KD_acc_list.append(KD_acc)


y1 = teacher_acc_list
y2 = student_acc_list
y3 = KD_acc_list
plt.plot(y1, label="teacher_accuracy")
plt.plot(y2, label="student_accuracy")
plt.plot(y3, label="kd_accuracy")
plt.xticks(range(50, 219, 3))
plt.ylim([0, 1])
plt.legend()
plt.show()
