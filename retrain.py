import os
import math
import random
import copy
import joblib
from collections import defaultdict
from collections import Counter

import torch
import torch.nn as nn
from torchvision.models.feature_extraction import create_feature_extractor
from torch.utils.data import Dataset,DataLoader

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score,silhouette_samples,classification_report
from sklearn.preprocessing import MinMaxScaler
from sklearn.manifold import TSNE


from train import load_dataset
from train import get_coarseToFine
from sampling import sampling_based_cluster,clustering,random_sampling,DeepGini_sampling,MCP_sampling
import config

import setproctitle


def gen_new_output_layer(old_output_layer, new_num_classes):
    w = old_output_layer.weight
    b = old_output_layer.bias
    in_features = old_output_layer.in_features
    new_output_layer = nn.Linear(in_features, new_num_classes)
    with torch.no_grad():
        for i in range(19):
            new_output_layer.weight[i] = w[i]
            new_output_layer.bias[i] = b[i]
    return new_output_layer

def reconstruct_model(model, num_classes = 24, model_name = "VGG19", freeze_feature = True):
    '''
    功能:修改model的分类层到指定分类数
    参数：
        model:original model
        num_classes:指定分类数
    '''

    new_model = copy.deepcopy(model)
    if freeze_feature == True:
        for param in new_model.parameters():
            param.requires_grad = False
    if model_name in ["ResNet18","ResNet50"]:
        old_output_layer = model.fc
        new_output_layer = gen_new_output_layer(old_output_layer, new_num_classes=num_classes)
        new_model.fc = new_output_layer
        return new_model
    elif model_name == "VGG19":
       old_output_layer = model.classifier[-1]
       new_output_layer = gen_new_output_layer(old_output_layer, new_num_classes=num_classes)
       new_model.classifier[-1] = new_output_layer
       return new_model
    
def get_create_feature_extractor(model, model_name):
    if model_name in ["ResNet18","ResNet50"]:
        return_node = "avgpool"
        feature_size = model.fc.in_features
        feature_extractor = create_feature_extractor(model, return_nodes=[return_node])
        return feature_size, feature_extractor, return_node
    elif model_name == "VGG19":
        return_node = "classifier.4"
        feature_size = model.classifier[-1].in_features
        feature_extractor = create_feature_extractor(model, return_nodes=[return_node])
        return feature_size, feature_extractor, return_node
    
def get_features(model,dataset,device):
    ptr = 0
    dataset_loader = DataLoader(
        dataset,
        batch_size = 128,
        shuffle=False,
        num_workers=4
    )
    model.eval()
    dataset_size = len(dataset)
    model.to(device)
    feature_size,feature_extractor,return_node = get_create_feature_extractor(model, config.model_name)
    features = torch.zeros((dataset_size,feature_size))
    for batch_idx, batch in enumerate(dataset_loader):
        X = batch[0]
        X = X.to(device)
        batch_size = X.shape[0]
        feature_dic = feature_extractor(X)
        feature = feature_dic[return_node]
        feature_squeezed = feature.squeeze()
        features[ptr:ptr+batch_size] = feature_squeezed
        ptr = ptr+batch_size
    return features

def kmeans_clustering(data_array,n_clusters):
    # K-Means 聚类
    kmeans = KMeans(n_clusters=n_clusters,max_iter=1000,n_init="auto")
    model =kmeans.fit(data_array)
    # 获得聚类中心
    centers = model.cluster_centers_
    # 获得聚类标签
    cluster_labels=model.labels_
    return centers, cluster_labels, model 



def center_sampling(record, sampling_rate = 0.1):
    # 根据聚类标签组内根据样本轮廓系数排序
    for cluster_label in record.keys():
        silhoue_list = record[cluster_label]
        # 计算所有轮廓系数值与0的绝对值
        silhoue_list = [(sample_idx, s) for sample_idx, s in silhoue_list]
        record[cluster_label] = sorted(silhoue_list, key=lambda item: item[-1])
    # 抽取每个聚类标签组内前1%
    sampled_dict= defaultdict(list)
    for cluster_label in record.keys():
        temp_list = record[cluster_label]
        cutoff = math.ceil(len(temp_list)*sampling_rate)
        sampled_dict[cluster_label] = temp_list[:cutoff]
    return sampled_dict

def reducting_dim(data_array, n_components, method="PCA"):
    if method == "PCA":
        pca = PCA(n_components=n_components)
        data_array_reduced = pca.fit_transform(data_array)
        return data_array_reduced
    elif method == "TSNE":
        tsne = TSNE(n_components=n_components)
        data_array_reduced = tsne.fit_transform(data_array)
        return data_array_reduced
    else:
        print("请指定正确的降维方法")
        

class ExtractTargetClassDataset(Dataset):
    '''
    从数据集中抽取出某个类别(target_class_idx)的数据集
    '''
    def __init__(self, dataset_coarse, dataset_fine, splited_class_idx):
        '''
        dataset_coarse: 带有粗标签的数据集
        dataset_fine: 带有细标签的数据集
        splited_class_idx: 要分裂的class_idx
        '''
        self.dataset_coarse = dataset_coarse
        self.dataset_fine = dataset_fine
        self.splited_class_idx = splited_class_idx
        self.splitedClassDataset = self._getSplitedClassDataset()

    def _getSplitedClassDataset(self):
        '''
        过滤出要分裂的class_idx的数据集,并带有粗标签和细标签
        '''
        splitedClassDataset = []
        for id in range(len(self.dataset_coarse)):
            sample, label_coarse = self.dataset_coarse[id][0], self.dataset_coarse[id][1]
            _, label_fine = self.dataset_fine[id][0], self.dataset_fine[id][1]
            if label_coarse == self.splited_class_idx:
                splitedClassDataset.append((sample, label_coarse, label_fine))
        return splitedClassDataset
    
    def __len__(self):
        return len(self.splitedClassDataset)
    
    def __getitem__(self, index):
        sample,label_coarse,label_fine =self.splitedClassDataset[index]
        return sample,label_coarse, label_fine

class NewSceneDataset(Dataset):
    '''
    从数据集中抽取出某个类别(target_class_idx)的数据集
    '''
    def __init__(self, dataset_coarse,dataset_fine,splited_class_idx, coarseTofine):
        self.dataset_coarse = dataset_coarse
        self.dataset_fine = dataset_fine
        self.splited_class_idx = splited_class_idx
        self.coarseTofine = coarseTofine
        self.newSceneDataset = self._newSceneDataset()

    def _newSceneDataset(self):
        newSceneDataset = []
        self.labels = []
        for id in range(len(self.dataset_coarse)):
            sample, label_coarse = self.dataset_coarse[id][0], self.dataset_coarse[id][1]
            if label_coarse == self.splited_class_idx:
                _, label_fine = self.dataset_fine[id][0], self.dataset_fine[id][1]
                new_label = self.coarseTofine[label_coarse].index(label_fine)+self.splited_class_idx
                newSceneDataset.append((sample,new_label))
                self.labels.append(new_label)
            else:
                newSceneDataset.append((sample,label_coarse))
                self.labels.append(label_coarse)
        return newSceneDataset
    
    def __len__(self):
        return len(self.newSceneDataset)
    
    def __getitem__(self, index):
        sample,label =self.newSceneDataset[index]
        return sample,label

class SampledDataset(Dataset):
    '''
    从数据集中根据sampled_indices采样出sampled dataset并修改标签到新分类场景标签
    '''
    def __init__(self, dataset, sampled_indices, coarseTofine, offset):
        self.dataset = dataset # 带有粗标签的数据集
        self.sampled_indices = sampled_indices
        self.coarseTofine = coarseTofine # {coarse_label:fine_label_list(sorted)}
        self.offset = offset # 新场景标签偏移量,一般是值等于splited_class_idx
        self.sampled_dataset = self._get_sampled_dataset()

    def _get_sampled_dataset(self):
        sampled_dataset = []
        self.labels = []
        for id in range(len(self.dataset)):
            if id in self.sampled_indices:
                sample, label_coarse, label_fine = self.dataset[id][0], self.dataset[id][1], self.dataset[id][2]
                label = self.coarseTofine[label_coarse].index(label_fine)+self.offset
                sampled_dataset.append((sample,label))
                self.labels.append(label)
        return sampled_dataset
    
    def __len__(self):
        return len(self.sampled_dataset)
    
    def __getitem__(self, index):
        sample,label =self.sampled_dataset[index]
        return sample,label


class MixDataset(Dataset):
    def __init__(self, train_dataset_coarse, splited_class_idx, sampled_dataset):
        self.train_dataset_coarse = train_dataset_coarse
        self.splited_class_idx = splited_class_idx
        self.sampled_dataset = sampled_dataset
        self.mix_dataset = self._create_mix_dataset()

    def _create_mix_dataset(self):
        mix_dataset = []
        self.labels = []
        candidate_train_id_list = []
        for id in range(len(self.train_dataset_coarse)):
            sample, label_coarse = self.train_dataset_coarse[id][0], self.train_dataset_coarse[id][1]
            if label_coarse != self.splited_class_idx:
                candidate_train_id_list.append(id)
        # 对于train_coarse_dataset，每个coarse label样本量是2500
        # 对于test_coarse_dataset，每个coarse label样本量是500
        # 对于新场景下，每个新标签的样本量是500/5=100，因此新场景下新标签的样本量是老粗标签的0.2倍
        # 因此我们希望重训练时，老粗标签样本量也是新标签的5倍。老粗的基数是新标签的25倍（2500/100=25），因此先给老的乘以个0.2系数。
        sample_size = int(0.2*len(candidate_train_id_list)*0.1) 
        sampled_train_id_list = random.sample(candidate_train_id_list, sample_size)
        for id in sampled_train_id_list:
            sample, label_coarse = self.train_dataset_coarse[id][0], self.train_dataset_coarse[id][1]
            mix_dataset.append((sample,label_coarse))
            self.labels.append(label_coarse)
        for id in range(len(self.sampled_dataset)):
            sample, label = self.sampled_dataset[id][0], self.sampled_dataset[id][1]
            mix_dataset.append((sample,label))
            self.labels.append(label)
        return mix_dataset
    
    def __len__(self):
        return len(self.mix_dataset)
    
    def __getitem__(self, index):
        sample,label =self.mix_dataset[index]
        return sample,label

def eval_model(model, dataset, device):
    y_pred = []
    y_true = []
    test_correct = 0
    test_sum = 0
    test_loss = 0.0
    test_loader = DataLoader(
        dataset,
        batch_size = 128,
        shuffle=False,
        num_workers=4
    )
    loss_fn=nn.CrossEntropyLoss()
    loss_fn.to(device)
    model.eval()
    model.to(device)
    with torch.no_grad():
        for batch_idx1,(data,target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            y_true.extend(target.tolist())
            output = model(data)
            loss = loss_fn(output, target)
            test_loss+=loss.item()*data.shape[0]
            _, predicted = torch.max(output.data, 1)
            y_pred.extend(predicted.tolist())
            test_correct += (predicted == target).sum().item()
            test_sum += target.size(0)
    test_loss = round(test_loss/test_sum,4)
    test_acc = round(100*test_correct/test_sum,4)
    msg = f"Test loss:{test_loss}|Test accuracy:{test_acc}%"
    print(msg)
    cla_report = classification_report(y_true, y_pred, output_dict=True)
    print(cla_report)
    return test_acc

def retrain_fine_model(model, dataset, device, epochs, lr):
    model.train()
    model.to(device)
    train_loader = DataLoader(
        dataset,
        batch_size = 128,
        shuffle=False,
        num_workers=4
    )
    optimizer=torch.optim.SGD(params=model.parameters(),lr=lr, momentum=0.9,weight_decay=0.0001)
    loss_fn=nn.CrossEntropyLoss()
    loss_fn.to(device)
    for epoch in range(epochs):
        print("-----第{}轮重训练开始------".format(epoch))
        # 统计该轮次训练集损失
        train_loss=0.0
        # 统计训练集数量
        train_sum = 0.0
        # 统计训练集中分类正确数量
        train_cor = 0.0
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
            train_loss += loss.item()*data.shape[0]
            # 选择最大的（概率）值所在的列数就是他所对应的类别数，
            _, predicted = torch.max(output.data, 1)  
            train_cor += (predicted == target).sum().item()  # 正确分类个数
            train_sum += target.size(0)
        train_loss = round(train_loss/train_sum,4)
        train_acc = round(100*train_cor/train_sum,4)
        msg = f"Train loss:{train_loss}|Train accuracy:{train_acc}%"
        print(msg)
    return model


def retrain_app(train_data_coarse,splited_class_idx,sampled_dataset,coarse_model,device, newSceneDataset):
    # 1:构建出其他训练集和采样集的混合集
    print("(1)构建出其他训练集和边界样本集混合集")
    mixDataset = MixDataset(
        train_data_coarse, 
        splited_class_idx=splited_class_idx, 
        sampled_dataset=sampled_dataset)
    print(f"retrain样本数量:{len(mixDataset)}")
    labels_counts = Counter(mixDataset.labels)
    print(f"retrain样本标签统计:{labels_counts}")
    # 2:构建出fine model
    print("(2) 构建出fine model")
    fine_model = reconstruct_model(coarse_model, num_classes=24, model_name=config.model_name,freeze_feature = True)
    # 3:开始对initial_fine_model retrain
    print("(3) 开始对initial_fine_model retrain")
    epochs = 10
    lr = 0.01
    print(f"超参数:epochs:{epochs}|lr:{lr}")
    retrained_fine_model = retrain_fine_model(
        fine_model, 
        mixDataset, 
        device, 
        epochs=epochs, 
        lr = lr)
    # 3:对retrained_fine_model eval
    print("(4) 开始对retrained_fine_model eval")
    retrained_test_acc = eval_model(retrained_fine_model, newSceneDataset, device)
    return retrained_test_acc



def retrain_exp(exp_id):
    print(f"retrain_exp:{exp_id}")
    # 加载数据集
    print("加载数据集")
    train_data_fine, test_data_fine, train_data_coarse, test_data_coarse = load_dataset()
    # 加载retrained coarse model
    print("加载retrained coarse model")
    coarse_model = torch.load(os.path.join(config.exp_dir, config.model_name, "best_coarse.pth"))
    # 指定分裂的类别,eg最后粗粒度的最后一个类别
    splited_class_idx = 19
    print(f"指定分裂的类别:{splited_class_idx}")
    splitedClassDataset =  ExtractTargetClassDataset(
        dataset_coarse = test_data_coarse, # 粗粒度全部测试集
        dataset_fine=test_data_fine,
        splited_class_idx = splited_class_idx # 分裂的类别id
        )
    print(f"分裂的数据集的数量:{len(splitedClassDataset)}")
    device = torch.device("cuda:0")
    n_clusters = 5
    print(f"开始聚类｜簇数:{n_clusters}")
    cluster_ans = clustering(splitedClassDataset, coarse_model, device, n_clusters)
    silhouette =cluster_ans["silhouette"]
    cluster_labels = cluster_ans["cluster_labels"]
    # 聚类中心采样
    print("聚类中心采样")
    _, center_sampled_indices = sampling_based_cluster(silhouette, cluster_labels, sampling_rate = 0.1, strategy="center")
    # 聚类边界采样
    print("聚类边界采样")
    _, boundary_sampled_indices = sampling_based_cluster(silhouette, cluster_labels, sampling_rate = 0.1, strategy="boundary")
    # 随机采样
    print("随机采样")
    random_sampled_indices = random_sampling(id_list = list(range(len(splitedClassDataset))), sampled_num=len(center_sampled_indices))
    # DeepGini采样
    print("DeepGini采样")
    deepGini_sampled_indices = DeepGini_sampling(coarse_model,splitedClassDataset,device,sampled_num=len(center_sampled_indices))
    # MCP采样
    print("MCP采样")
    mcp_sampled_indices = MCP_sampling(coarse_model, splitedClassDataset, device, sampled_num=len(center_sampled_indices))

    # 构建出采样数据集并修改标签为新分类场景中的标签
    coarseTofine = get_coarseToFine(file_path=os.path.join(config.dataset_root_dir, "train"))

    # 1:构建聚类边界样本集
    print("1:构建聚类边界样本集")
    boundary_sampled_dataset = SampledDataset(splitedClassDataset,boundary_sampled_indices,coarseTofine,splited_class_idx)
    print(f"采样的边界样本数量:{len(boundary_sampled_dataset)}")
    labels_counts = Counter(boundary_sampled_dataset.labels)
    print(f"采样的边界标签数量统计:{labels_counts}")

    # 2:构建聚类中心采样集
    print("2:构建聚类中心采样集")
    center_sampled_dataset = SampledDataset(splitedClassDataset,center_sampled_indices,coarseTofine,splited_class_idx)
    print(f"采样的中心样本数量:{len(center_sampled_dataset)}")
    labels_counts = Counter(center_sampled_dataset.labels)
    print(f"采样的中心标签数量统计:{labels_counts}")

    # 3:构建随机采样集
    print("3:构建随机采样集")
    random_sampled_dataset = SampledDataset(splitedClassDataset,random_sampled_indices,coarseTofine,splited_class_idx)
    print(f"样本数量:{len(random_sampled_dataset)}")
    labels_counts = Counter(random_sampled_dataset.labels)
    print(f"随机采样标签数量统计:{labels_counts}")
    
    # 4:构建DeepGini采样集
    print("4:构建DeepGini采样集")
    deepGini_sampled_dataset = SampledDataset(splitedClassDataset,deepGini_sampled_indices,coarseTofine,splited_class_idx)
    print(f"DeepGini采样样本数量:{len(deepGini_sampled_dataset)}")
    labels_counts = Counter(deepGini_sampled_dataset.labels)
    print(f"DeepGini采样标签数量统计:{labels_counts}")
    
    # 5:构建MCP采样集
    mcp_sampled_dataset = SampledDataset(splitedClassDataset,mcp_sampled_indices,coarseTofine,splited_class_idx)
    print(f"MCP采样样本数量:{len(mcp_sampled_dataset)}")
    labels_counts = Counter(mcp_sampled_dataset.labels)
    print(f"MCP采样标签数量统计:{labels_counts}")

    # 构建新场景数据集
    print("构建新场景数据集")
    newSceneDataset = NewSceneDataset(
        test_data_coarse,
        test_data_fine,
        splited_class_idx=splited_class_idx,
        coarseTofine=coarseTofine)
    print(f"新场景样本数量:{len(newSceneDataset)}")
    labels_counts = Counter(newSceneDataset.labels)
    print(f"新场景样本标签数量统计:{labels_counts}")

    # 构建出fine model
    print("构建出fine model")
    fine_model = reconstruct_model(coarse_model, num_classes=24, model_name=config.model_name,freeze_feature = True)
    # 用新场景数据集对initial_fine_model进行评估
    print("用新场景数据集对initial_fine_model进行评估")
    initial_test_acc = eval_model(fine_model, newSceneDataset, device)

    res = {}
    # 开始retrain阶段
    print("开始retrain阶段")
    print("1:聚类边界采样retrain")
    retrain_test_acc = retrain_app(
        train_data_coarse,
        splited_class_idx,
        boundary_sampled_dataset,
        coarse_model,
        device, 
        newSceneDataset)
    improve_acc = round(retrain_test_acc - initial_test_acc,4)
    print(f"boundary_retrain|improve_acc:{improve_acc}%")
    res["boundary"] = improve_acc

    print("2:聚类中心采样retrain")
    retrain_test_acc = retrain_app(
        train_data_coarse,
        splited_class_idx,
        center_sampled_dataset,
        coarse_model,
        device, 
        newSceneDataset)
    improve_acc = round(retrain_test_acc - initial_test_acc,4)
    print(f"center_retrain|improve_acc:{improve_acc}%")
    res["center"] = improve_acc

    print("3:随机采样retrain")
    retrain_test_acc = retrain_app(
        train_data_coarse,
        splited_class_idx,
        random_sampled_dataset,
        coarse_model,
        device, 
        newSceneDataset)
    improve_acc = round(retrain_test_acc - initial_test_acc,4)
    print(f"random_retrain|improve_acc:{improve_acc}%")
    res["random"] = improve_acc

    print("4:DeepGini采样retrain")
    retrain_test_acc = retrain_app(
        train_data_coarse,
        splited_class_idx,
        deepGini_sampled_dataset,
        coarse_model,
        device, 
        newSceneDataset)
    improve_acc = round(retrain_test_acc - initial_test_acc,4)
    print(f"DeepGini_retrain|improve_acc:{improve_acc}%")
    res["DeepGini"] = improve_acc

    print("5:MCP采样retrain")
    retrain_test_acc = retrain_app(
        train_data_coarse,
        splited_class_idx,
        mcp_sampled_dataset,
        coarse_model,
        device, 
        newSceneDataset)
    improve_acc = round(retrain_test_acc - initial_test_acc,4)
    print(f"MCP_retrain|improve_acc:{improve_acc}%")
    res["MCP"] = improve_acc
    return res

def retrain_exp_main():
    repeat_num = 5
    retrain_ans = {}
    for exp_id in range(repeat_num):
        retrain_ans[exp_id] = retrain_exp(exp_id)
    save_dir = os.path.join("exp_ans_data",config.dataset_name,config.model_name)
    os.makedirs(save_dir,exist_ok=True)
    save_file_name = "retrain_ans.data"
    save_path = os.path.join(save_dir,save_file_name)
    joblib.dump(retrain_ans,save_path)
    print("实验结果保存在:",save_path)
    
if __name__ == "__main__":
    proctitle = f"{config.dataset_name}_{config.model_name}_retrain"
    setproctitle.setproctitle(proctitle)
    retrain_exp_main()