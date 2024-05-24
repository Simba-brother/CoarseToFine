import os
import copy
import torch
import torchvision
from torch import nn
from torch.utils.data import Dataset,DataLoader
from torchvision.models import resnet18,vgg19,resnet50
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import pickle
import time
import pickle
import config
import setproctitle
from log import Logger
#公开数据集的预处理库,格式转换
import torchvision.transforms as transforms  
import logging
from torch.optim.lr_scheduler import StepLR
from collections import defaultdict

def unpickle(file_path):
        with open(file_path, "rb") as f:
            file_dict = pickle.load(f, encoding="latin1")
        return file_dict
    
def get_coarseToFine(file_path,):
    file_dict = unpickle(file_path)
    fine_labels = file_dict["fine_labels"]
    coarse_labels = file_dict["coarse_labels"]
    coarseToFine = defaultdict(set)
    for fine_label, coarse_label in zip(fine_labels, coarse_labels):
        coarseToFine[coarse_label].add(fine_label)
    for coarse_label in coarseToFine.keys():
        fine_label_list = list(coarseToFine[coarse_label])
        sorted_fine_label_list = sorted(fine_label_list)
        coarseToFine[coarse_label] = sorted_fine_label_list
    return coarseToFine


def fineToCoarse(file_path, labels):
    file_dict = unpickle(file_path)
    fine_labels = file_dict["fine_labels"]
    coarse_labels = file_dict["coarse_labels"]
    assert len(fine_labels) == len(coarse_labels), "数量不对"
    mapping = {}
    for fine_label, coarse_label in zip(fine_labels, coarse_labels):
         mapping[fine_label] = coarse_label
    fine_keys = mapping.keys()
    coarse_values = mapping.values()
    assert len(fine_keys) == 100, "数量不对"
    assert len(set(coarse_values)) == 20, "数量不对"
    new_labels = []
    for label in labels:
        new_labels.append(mapping[label])
    return new_labels


def load_dataset():
    '''
    加载CIFAR-100数据集
    '''
    # 准备数据集
    # 数据集格式转换
    transform_train = transforms.Compose(
        [transforms.Resize(256),           #transforms.Scale(256)
        transforms.CenterCrop(224), 
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    
    transform_test = transforms.Compose(
        [transforms.Resize(256),         #transforms.Scale(256)
        transforms.CenterCrop(224), 
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    # 加载数据集
    dataset_root_dir = "/data/mml/CoarseToFine/dataset/CIFAR100/cifar-100-python"
    train_data_fine=torchvision.datasets.CIFAR100(dataset_root_dir,train=True,transform=transform_train,download=True)
    test_data_fine=torchvision.datasets.CIFAR100(dataset_root_dir,train=False,transform=transform_test,download=True)

    train_data_coarse = copy.deepcopy(train_data_fine)
    test_data_coarse = copy.deepcopy(test_data_fine)

    train_data_coarse.targets = fineToCoarse(os.path.join(dataset_root_dir, "train"), train_data_coarse.targets)
    test_data_coarse.targets = fineToCoarse(os.path.join(dataset_root_dir, "test"), test_data_coarse.targets)

    return train_data_fine, test_data_fine, train_data_coarse, test_data_coarse

    # 试验输出
    # print(train_data.targets[0]) #输出第一个标签值，为19，对应牛的标签
    # print(type(train_data.targets)) # <class 'list'>,数据集标签类型是列表
    # print(train_data.data[0].shape) #(32, 32, 3) 原始数据集图像的维度
    # plt.imshow(train_data.data[0]) #输出了牛的图片
    # plt.show()

def load_model(num_classes, model_name):
    if model_name == "ResNet18":
        model = resnet18(pretrained = True)
        # 冻结预训练模型中所有参数的梯度
        # for param in model.parameters():
        #     param.requires_grad = False
        # 修改最后一个全连接层的输出类别数量
        fc_features = model.fc.in_features
        model.fc = nn.Linear(fc_features, num_classes)
        return model
    elif model_name == "VGG19":
        model = vgg19(pretrained = True)
        # 冻结预训练模型中所有参数的梯度
        # for param in model.parameters():
        #     param.requires_grad = False
        # 修改最后一个全连接层的输出类别数量
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
        return model
    elif model_name == "ResNet50":
        model = resnet50(pretrained = True)
        # 冻结预训练模型中所有参数的梯度
        # for param in model.parameters():
        #     param.requires_grad = False
        # 修改最后一个全连接层的输出类别数量
        fc_features = model.fc.in_features
        model.fc = nn.Linear(fc_features, num_classes)
        return model

def train_main(logger):
    # 加载数据集
    train_data_fine, test_data_fine, train_data_coarse, test_data_coarse = load_dataset()
    train_loader = DataLoader(
        train_data_coarse,
        batch_size = 128,
        shuffle=True,
        num_workers=4
    )

    test_loader = DataLoader(
        test_data_coarse,
        batch_size = 128,
        shuffle=False,
        num_workers=4
    )

    # 加载模型
    model = load_model(num_classes=20, model_name=config.model_name)
    # 指定训练设备
    device = torch.device("cuda:0")
    model.to(device)
    # 指定损失函数,对于cross_entropy会进行log_softmax操作
    loss_fn=nn.CrossEntropyLoss()
    loss_fn.to(device)
    # 指定学习率
    learning_rate=0.01
    # 指定优化器
    optimizer=torch.optim.SGD(params=model.parameters(),lr=learning_rate, momentum=0.9,weight_decay=0.0001)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

    train_acc_list = []
    train_loss_list = []
    test_acc_list = []
    test_loss_list=[]
    epochs=20
    best_test_acc = 0
    for epoch in range(epochs):
        logger.debug("-----第{}轮训练开始------".format(epoch))
        # 统计该轮次训练集损失
        train_loss=0.0
        # 统计该轮次测试集损失
        test_loss=0.0
        # 统计训练集数量
        train_sum = 0.0
        # 统计训练集中分类正确数量
        train_cor = 0.0
        # 统计测试集数量
        test_sum = 0.0
        # 统计测试集中分类正确数量
        test_cor=0.0
        #训练步骤开始
        model.train()
        for batch_idx,(data,target) in enumerate(train_loader):
            data,target=data.to(device),target.to(device)
            # 要将梯度清零，因为如果梯度不清零，pytorch中会将上次计算的梯度和本次计算的梯度累加
            optimizer.zero_grad()  
            output = model(data)
            # 该批次损失
            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()  # 更新所有的参数
            # 累加该批次训练集的Loss
            train_loss += loss.item()
            # 选择最大的（概率）值所在的列数就是他所对应的类别数，
            _, predicted = torch.max(output.data, 1)  
            # train_cor += (predicted == target).sum().item()  # 正确分类个数
            train_cor += (predicted == target).sum().item()  # 正确分类个数
            train_sum += target.size(0)  
        scheduler.step()
        model.eval()
        with torch.no_grad():
            for batch_idx1,(data,target) in enumerate(test_loader):
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = loss_fn(output, target)
                test_loss+=loss.item()
                _, predicted = torch.max(output.data, 1)
                test_cor += (predicted == target).sum().item()
                test_sum += target.size(0)
        
        train_loss = round(train_loss/batch_idx,4)
        train_acc = round(100*train_cor/train_sum,4)
        test_loss = round(test_loss/batch_idx1,4)
        test_acc = round(100*test_cor/test_sum,4)
        msg = f"Train loss:{train_loss}|Train accuracy:{train_acc}%|Test loss:{test_loss}|Test accuracy:{test_acc}%"
        logger.debug(msg)
        
        train_loss_list.append(train_loss)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_loss)
        test_loss_list.append(test_acc)
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            save_dir = os.path.join(config.exp_dir, f"{config.model_name}")
            os.makedirs(save_dir, exist_ok=True)
            save_file_name = "best_coarse.pth"
            save_file_path = os.path.join(save_dir,save_file_name)
            torch.save(model,save_file_path)
            logger.debug(f"best coarse model is saved in {save_file_path}")


def eval(model,dataset,device):
    model.eval()
    test_loader = DataLoader(
        dataset,
        batch_size = 128,
        shuffle=False,
        num_workers=4
    )
    test_loss = 0
    test_correct = 0
    test_total = 0
    loss_fn=nn.CrossEntropyLoss()
    loss_fn.to(device)
    with torch.no_grad():
        for batch_idx1,(data,target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = loss_fn(output, target)
            test_loss+=loss.item()*data.shape[0]
            _, predicted = torch.max(output.data, 1)
            test_correct += (predicted == target).sum().item()
            test_total += target.size(0)
    acc = round(test_correct/test_total, 4)
    loss = round(test_loss/test_total, 4)
    print(f"acc:{acc}|loss:{loss}")

def eval_app():
    train_data_fine, test_data_fine, train_data_coarse, test_data_coarse = load_dataset()
    device = torch.device("cuda:0")
    coarse_model = torch.load(os.path.join(config.exp_dir, config.model_name, "best_coarse.pth"))
    acc = eval(coarse_model,test_data_coarse,device)
    print(acc)
if __name__ == "__main__":
    # proctitle = f"{config.dataset_name}_{config.model_name}_coarse-train"
    # setproctitle.setproctitle(proctitle)
    # logger = Logger(path=f'./Log/{proctitle}.log', clevel=logging.ERROR, Flevel=logging.DEBUG)
    # train_main(logger)
    eval_app()

    