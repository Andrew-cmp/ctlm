import pandas as pd 
import numpy 
import sklearn
import argparse
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.manifold import MDS
from sklearn.decomposition import PCA
#plt.rcParams['font.sans-serif'] = ['serif']
#plt.rcParams['axes.unicode_minus'] = False  
def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--choice", required=True,type=str, choices = ["plt_png","calculate_cos","calculate_euc","calculate_corr","kmeans","pca","dbscan"],help="Pick out the top k"
    )
    return parser.parse_args()
script_args = _parse_args()
def plt_png():
    info_path = "dataset/measured/statistics_each_target/total_inversions.csv"
    df = pd.read_csv(info_path,index_col=0)
    subgraphs_name = df.columns
    targets_name = df.index
    subgraphs_name = [str(i) for i in subgraphs_name]
    targets_name = [str(i) for i in targets_name]
    # print(subgraphs_name)
    # print(targets_name)

    # print(df)
    df.describe()
    data = df.to_numpy()
    # print(type(data[0][0]))
    data = data.T
    subgraph_num= len(data)
    target_num= len(data[0])

    n_samples = subgraph_num
    n_features = target_num
    # print(data)
    # print(f"n_samples:{n_samples}")
    # print(f"n_features:{n_features}")

    plt.figure(figsize=(10, 6))
    plt.figure(figsize=(10, 6))
    subgraph_var = []
    for i,subgraph_name in enumerate(subgraphs_name):
        plt.figure(figsize=(8, 4))
        # x轴为特征索引，0, 1, 2, ..., n_feature-1
        x = np.arange(n_features)
        y = data[i, :]  # 当前样本的所有特征值
        
        sample_variance = np.var(y , ddof=1)
        subgraph_var.append([subgraph_name,sample_variance])
        plt.xticks(x, targets_name)
        # 绘制散点图
        plt.scatter(x, y, color='blue')
        # 可选：使用折线连接各点（如有需要）
        #plt.plot(x, y, color='gray', alpha=0.5)
        # 添加标题和坐标轴标签
        plt.title(f'Sample {subgraph_name} Feature Distribution')
        plt.xlabel('Feature Index')
        plt.ylabel('Value')
        # 调整布局以防止标签被截断
        plt.tight_layout()
        # 保存当前图到文件，例如保存为 "sample_0.png", "sample_1.png", ...
        plt.savefig(f'dataset/measured/plt/{subgraph_name}.png')
        # 关闭当前图，防止内存占用过多
        plt.close()
    subgraph_var.sort(key=lambda x: x[1])
    df = pd.DataFrame(subgraph_var,columns=["subgraph_name","var"])
    df.to_csv(f'dataset/measured/plt/0subgraph_var.csv')
def calculate_cos():
    info_path = "dataset/measured/statistics_each_target/total_inversions.csv"
    df = pd.read_csv(info_path,index_col =0)
    data = df.to_numpy()
    targets_name = df.index
    targets_name = [str(i) for i in targets_name]
    targets_name_T = targets_name
    similarity_matrix = cosine_similarity(data)
    print(similarity_matrix)
    df = pd.DataFrame(similarity_matrix,index = targets_name,columns = targets_name)
    print(df)
def calculate_euc():
    info_path = "dataset/measured/statistics_each_target/total_inversions.csv"
    df = pd.read_csv(info_path,index_col =0)
    data = df.to_numpy()
    targets_name = df.index
    targets_name = [str(i) for i in targets_name]
    targets_name_T = targets_name
    distance_matrix  = euclidean_distances(data)
    similarity_matrix_euclid = 1 / (1 + distance_matrix)
    print(similarity_matrix_euclid)
    df = pd.DataFrame(similarity_matrix_euclid,index = targets_name,columns = targets_name)
    print(df)
def calculate_corr():
    info_path = "dataset/measured/statistics_each_target/total_inversions.csv"
    df = pd.read_csv(info_path,index_col=0)
    data = df.to_numpy()
    targets_name = df.index
    targets_name = [str(i) for i in targets_name]
    targets_name_T = targets_name
    correlation_matrix = np.corrcoef(data)
    softmax_correlation_matrix = soft_max(correlation_matrix)
    df = pd.DataFrame(softmax_correlation_matrix,index = targets_name,columns = targets_name)
    print(df)
def soft_max(data):
    data[data == 1] = 0
    # 2. 按行计算 softmax
    # 先计算每一行的最大值（用于数值稳定性）
    row_max = np.max(data, axis=1, keepdims=True)
    # 计算每个元素的指数
    exp_data = np.exp(data - row_max)
    # 计算每一行指数的和
    sum_exp = np.sum(exp_data, axis=1, keepdims=True)
    # 计算 softmax
    softmax_data = exp_data / sum_exp
    return softmax_data

def dim_reduction_pca():
    info_path = "dataset/measured/statistics_each_target/total_inversions.csv"
    df = pd.read_csv(info_path,index_col =0)
    data = df.to_numpy()
    targets_name = df.index
    targets_name = [str(i) for i in targets_name]
    scaler = StandardScaler()
    data_scaled =  scaler.fit_transform(data)
    df = pd.DataFrame(data_scaled,index = targets_name)
    #print(df)
    pca = PCA(n_components=0.99)
    data_pca = pca.fit_transform(data_scaled)
    df = pd.DataFrame(data_pca,index = targets_name)
    print(df)
    n_clusters = 3  # 可以自己设置类别数量
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(data_pca)

    # 打印每个样本的类别
    print("每个样本的聚类标签：")
    print(targets_name)
    print(labels)

    # -----------------------------
    # 4. 使用 PCA 降维用于可视化
    # -----------------------------
    pca_visible = PCA(n_components=2)
    data_pac_visible = pca_visible.fit_transform(data_pca)

    # -----------------------------
    # 5. 可视化聚类结果
    # -----------------------------
    plt.figure(figsize=(8, 6))
    for i in range(n_clusters):
        plt.scatter(data_pac_visible[labels == i, 0], data_pac_visible[labels == i, 1], label=f"Cluster {i}")

    for i, name in enumerate(targets_name):
        plt.text(data_pac_visible[i, 0], data_pac_visible[i, 1], name, fontsize=9, ha='right')
    plt.title("KMeans Clustering (PCA Reduced)")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    plt.savefig("kmeans_clustering_result.png", dpi=300)  # dpi=300 表示高清输出
    plt.close()  # 关闭图像，避免不必要的显示（如在循环中）
def DBSCAN_cluster():
    
    
    info_path = "dataset/measured/statistics_per_target_each_target/distance.csv"
    df = pd.read_csv(info_path,index_col =0)
    distance = 1 - df.to_numpy()
    targets_name = df.index
    targets_name = [str(i) for i in targets_name]
    
    # eps 是最大距离阈值
    model = DBSCAN(metric='precomputed', eps=0.05, min_samples=1)
    labels = model.fit_predict(distance)
    print("target name：", targets_name)
    print("DBSCAN label：", labels)
    
    
    embedding = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
    X_2d = embedding.fit_transform(distance)

    # ------------------------------
    # 5. 可视化聚类结果
    # ------------------------------
    plt.figure(figsize=(8, 6))
    unique_labels = set(labels)
    colors = plt.colormaps["tab10"]

    for k in unique_labels:
        idx = (labels == k)
        if k == -1:
            # 噪声点
            plt.scatter(X_2d[idx, 0], X_2d[idx, 1], c='k', marker='x', label='Noise')
        else:
            plt.scatter(X_2d[idx, 0], X_2d[idx, 1], label=f'Cluster {k}')

    # 加上文字标签
    for i, name in enumerate(targets_name):
        plt.text(X_2d[i, 0], X_2d[i, 1], name, fontsize=9, ha='right')

    plt.title("DBSCAN visible(MDS)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    plt.savefig("dbscan_clustering_result.png", dpi=300)  # dpi=300 表示高清输出
    plt.close()  # 关闭图像，避免不必要的显示（如在循环中）
def main():
    if script_args.choice == "plt_png":
        plt_png()
    elif script_args.choice == "calculate_cos":
        calculate_cos()
    elif script_args.choice == "calculate_euc":
        calculate_euc()
    elif script_args.choice == "calculate_corr":
        calculate_corr()
    elif script_args.choice == "kmeans":
        cluster_kmeans()
    elif script_args.choice == "pca":
        dim_reduction_pca()
    elif script_args.choice == "dbscan":
        DBSCAN_cluster()
if __name__ == "__main__":
    main()