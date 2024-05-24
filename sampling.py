import random
import queue
import torch
from torch.utils.data import Dataset,DataLoader
from torchvision.models.feature_extraction import create_feature_extractor
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import silhouette_score,silhouette_samples,classification_report
from sklearn.cluster import KMeans
from collections import defaultdict
import math
import config

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

def kmeans_clustering(data_array,n_clusters):
    # K-Means 聚类
    kmeans = KMeans(n_clusters=n_clusters,max_iter=1000,n_init="auto")
    model =kmeans.fit(data_array)
    # 获得聚类中心
    centers = model.cluster_centers_
    # 获得聚类标签
    cluster_labels=model.labels_
    return centers, cluster_labels, model 

def clustering(dataset,model,device,n_clusters):
    '''
    对数据集进行聚类然后进行聚类边界进行采样
    return:
        sampled_indices:dataset中采样的indices
    '''
    # 1:获得数据集中数据在model上的features
    features = get_features(model,dataset,device)
    features_array = features.detach().numpy()
    # 2:基于特征进行聚类和聚类边界采样
    #   (1):先对特征进行降维处理,例如使用TSNE降维算法
    features_array_reduced = reducting_dim(features_array,n_components=2,method="TSNE")
    #   (2):对降维后的特征进行归一化
    scaler = MinMaxScaler()
    features_array_reduced = scaler.fit_transform(features_array_reduced)
    #   (3):对降维且归一化的特征进行聚类,例如使用Kmeans聚类算法
    centers, cluster_labels, kmeans_model = kmeans_clustering(features_array_reduced,n_clusters=n_clusters)
    #   (4):评估聚类效果并采样聚类边界sample_id_list
    # 计算轮廓系数
    silhouette_avg = silhouette_score(features_array_reduced, cluster_labels, metric='euclidean')
    print(f"For n_clusters = {n_clusters}, The average silhouette_score is : {silhouette_avg}")
    # 得到每个样本的轮廓系数
    silhouette = silhouette_samples(features_array_reduced, cluster_labels, metric='euclidean')
    # 返回计算结果
    ans = {
        "cluster_labels":cluster_labels,
        "silhouette":silhouette,
        "kmeans_model":kmeans_model,
        "centers":centers,
        "features_array_reduced":features_array_reduced
    }
    return ans

def priorityQueue_2_list(q:queue.PriorityQueue):
    qsize = q.qsize()
    res = []
    while not q.empty():
        res.append(q.get())
    assert len(res) == qsize, "队列数量不对"
    return res

def random_sampling(id_list, sampled_num):
    sampled_id_list = random.sample(id_list, sampled_num)    
    return sampled_id_list

def DeepGini_sampling(model,dataset,device,sampled_num):
    def caculate_deepGini(prob_list):
        csum = 0
        for p in prob_list:
            csum += p*p
        deepgini= 1-csum
        return deepgini
    model.eval()
    test_loader = DataLoader(
        dataset,
        batch_size = 128,
        shuffle=False,
        num_workers=4
    )
    gini_list = []    
    with torch.no_grad():
        for batch_idx1,(data,_,_) in enumerate(test_loader):
            data= data.to(device)
            output = model(data)
            for i in range(output.shape[0]):
                gini_list.append(caculate_deepGini(output[i].tolist()))
    q = queue.PriorityQueue()
    for idx, deepGini in enumerate(gini_list):
        item = (-deepGini,idx)
        q.put(item)
    
    priority_list = priorityQueue_2_list(q)
    sampled_id_list = [idx for _, idx in priority_list[0:sampled_num]]
    return sampled_id_list

def MCP_sampling(model, dataset, device, sampled_num):
    def get_mcp_matrix(prob_outputs, class_num):
        matrix = [[None for i in range(class_num)] for j in range(class_num)]
        for i in range(class_num):
            for j in range(class_num):
                q = queue.PriorityQueue()
                matrix[i][j] = q
        for i in range(len(prob_outputs)):
            prob_list = prob_outputs[i]
            sorted_prob_list = sorted(prob_list)
            max_p = sorted_prob_list[-1]
            second_p = sorted_prob_list[-2]
            priority = second_p/max_p # 值越大优先级越高
            label_one = prob_list.index(max_p)
            label_two = prob_list.index(second_p)
            item = (-priority, label_one, label_two, i)
            matrix[label_one][label_two].put(item)
        return matrix
    model.eval()
    test_loader = DataLoader(
        dataset,
        batch_size = 128,
        shuffle=False,
        num_workers=4
    )
    output_list = []
    with torch.no_grad():
        for batch_idx1,(data,_,_) in enumerate(test_loader):
            data = data.to(device)
            outputs = model(data)
            for output in outputs:
                output_list.append(output.tolist())
    class_num = 20
    mcp_matrix = get_mcp_matrix(output_list, class_num = class_num)
    
    priority_list = []
    while sampled_num > 0:
        for i in range(class_num):
            if sampled_num <= 0:
                    break
            for j in range(class_num):
                if mcp_matrix[i][j].qsize() == 0:
                    continue
                priority_list.append(mcp_matrix[i][j].get()) 
                sampled_num -= 1
                if sampled_num <= 0:
                    break
    sampled_id_list = [idx for _,_,_, idx in priority_list]
    return sampled_id_list
    
def sampling_based_cluster(silhouette, cluster_labels, sampling_rate = 0.1, strategy="center"):
    # 根据聚类标签分组每个样本
    record = defaultdict(list)
    for sample_idx in range(len(cluster_labels)):
        cluster_label = cluster_labels[sample_idx]
        silhouet = silhouette[sample_idx]
        record[cluster_label].append((sample_idx,silhouet))
    assert(len(record.keys()) == 5), "数量不对"
    # 根据聚类标签组内根据样本轮廓系数排序
    for cluster_label in record.keys():
        silhoue_list = record[cluster_label]
        if strategy == "boundary":
        # 计算所有轮廓系数值与0的绝对值
            silhoue_list = [(sample_idx, abs(s-0)) for sample_idx, s in silhoue_list]
            record[cluster_label] = sorted(silhoue_list, key=lambda item: item[-1])
        elif strategy == "center":
            silhoue_list = [(sample_idx, s) for sample_idx, s in silhoue_list]
            record[cluster_label] = sorted(silhoue_list, key=lambda item: item[-1], reverse=True) # 从大到小
        else:
            print("请指定基于距离的采样策略")
    # 抽取每个聚类标签组内前1%
    sampled_dict= defaultdict(list)
    for cluster_label in record.keys():
        temp_list = record[cluster_label]
        cutoff = math.ceil(len(temp_list)*sampling_rate)
        sampled_dict[cluster_label] = temp_list[:cutoff]
    # 获得采样的indices
    sampled_indices = []
    for cluster_label in sampled_dict.keys():
        sampled_indices.extend([sampled_idx for sampled_idx, sil in sampled_dict[cluster_label]])
    return sampled_dict,sampled_indices