### RAPTOR 算法：对文档进行聚类后，对每个聚类进行总结，并递归进行，直到指定级别
# 这个算法的主要步骤：
# 1. 将文档转换为嵌入向量
# 2. 使用 UMAP 进行降维
# 3. 使用高斯混合模型进行聚类
# 4. 对每个聚类生成摘要
# 5. 递归重复上述步骤，直到达到指定层级
# 应用场景：处理大量文档时，通过层次化的方式组织文档，并生成摘要

from typing import List, Dict, Optional, Tuple
import numpy as np
import pandas as pd
from umap.umap_ import UMAP
from sklearn.mixture import GaussianMixture
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

RANDOM_SEED = 42

# 全局降维考虑所有数据点，局部降维考虑局部邻居
def local_cluster_embeddings(
    embeddings: np.ndarray,
    dim: int,
    n_neighbors: int=10,
    metric: str="cosine",
) -> np.ndarray:
    """
    使用 UMAP 执行局部维度降维

    Args:
        embeddings: 嵌入向量
        dim: 目标维度
        n_neighbors: 每个点的邻居数
        metric: 距离计算方式

    Returns:
        np.ndarray: 降维后的向量
    """
    return UMAP(
        n_neighbors=n_neighbors,
        n_components=dim,
        metric=metric,
        random_state=RANDOM_SEED
    ).fit_transform(embeddings)
    
def global_cluster_embeddings(
    embeddings: np.ndarray,
    dim: int,
    n_neighbors: Optional[int] = None,
    metric: str = "cosine",
) -> np.ndarray:
    """
    使用 UMAP 执行全局维度到降维
    UMAP (Uniform Manifold Approximation and Projection) 是一种非线性降维算法，
    它能够保持数据的局部和全局结构

    Args:
        embeddings: 嵌入向量
        dim: 目标维度
        n_neighbors: 对于每个点，要考虑到邻居数，如未提供，默认是维度的平方根
        metric: 距离计算方式

    Returns:
        np.ndarray: 降维后的向量
    """
    if n_neighbors is None:
        n_neighbors = int((len(embeddings)-1) ** 0.5)
    return local_cluster_embeddings(
        embeddings=embeddings,
        dim=dim,
        n_neighbors=n_neighbors,
        metric=metric
    )

def get_optimal_clusters(
    embeddings: np.ndarray,
    max_clusters: int=50,
    random_state: int=RANDOM_SEED,
) -> int:
    """
    获取最佳聚类数
    使用贝叶斯信息准则(BIC)来确定最优的聚类数量
    BIC 值越小表示模型越好，这里选择 BIC 最小的聚类数

    Args:
        embeddings: 嵌入向量
        max_clusters: 最大聚类数
        random_state: 随机种子

    Returns:
        int: 最佳聚类数
    """
    # 确保最大聚类数不会超过数据点的数量
    max_clusters = min(max_clusters, len(embeddings))
    # 生成从1到最大聚类数的整数数组（代表我们要尝试的不同聚类数量）
    n_clusters = np.arange(1, max_clusters)
    # 存储每个聚类数的 BIC 值（BIC 值越小表示模型越好）
    distortions = []
    # 遍历每个聚类数，计算 BIC 值
    for n in n_clusters:
        # 创建高斯混合模型，并拟合数据
        clusters = GaussianMixture(n_components=n, random_state=random_state).fit(embeddings)
        # 计算 BIC 值，并存储到列表中
        distortions.append(clusters.bic(embeddings))
    # 返回 BIC 值最小的聚类数   
    return n_clusters[np.argmin(distortions)]

def GMM_cluster(
    embeddings: np.ndarray, 
    threshold: float,
    random_state: int=0,
):
    """
    使用基于概率的高斯混合模型进行聚类

    Args:
        embeddings: 嵌入向量
        threshold: 将一个点分配到某个聚类的概率阈值
        random_state: 随机种子

    Returns:
        聚类结果
    """
    # 获取最佳聚类数
    n_clusters = get_optimal_clusters(embeddings)
    # 创建高斯混合模型，并拟合数据
    gm = GaussianMixture(n_components=n_clusters, random_state=random_state).fit(embeddings)
    # 计算每个数据点属于每个聚类的概率，结果每一行代表一个数据点，每一列代表一个聚类（第二列代表第二类）
    probs = gm.predict_proba(embeddings)
    # 根据阈值，将每个数据点分配到大于阈值的聚类中，概率最高的类；如果所有类都小于阈值，则分配到概率最高的类
    labels = []
    for p in probs:
        valid_labels = np.where(p > threshold)[0]
        if len(valid_labels) > 0:
            max_prob_label = valid_labels[np.argmax(p[valid_labels])]
        else:
            max_prob_label = np.argmax(p)
        labels.append(np.array([max_prob_label]))
    return labels, n_clusters

def perform_clustering(
    embeddings: np.ndarray,
    threshold: float,
    dim: int,
) -> List[np.ndarray]:
    """
    执行层次化聚类
    1. 首先进行全局降维
    2. 使用高斯混合模型进行全局聚类
    3. 在每个全局聚类内进行局部聚类
    4. 返回每个文档所属的所有聚类标签

    Args:
        embeddings: 嵌入向量
        threshold: 将一个点分配到某个聚类的概率阈值
        dim: 目标维度

    Returns:
        List[np.ndarray]: 数组列表，每个数组包含每个嵌入的集群 ID
    """
    # 如果数据点太少，直接返回所有点属于同一个类
    if len(embeddings) <= dim + 1:
        return [np.array([0]) for _ in range(len(embeddings))]
    
    # 全局聚类，将所有数据点聚类到几个全局聚类中
    reduced_embeddings_global = global_cluster_embeddings(embeddings, dim)
    global_clusters, n_global_clusters = GMM_cluster(reduced_embeddings_global, threshold)
    # 局部聚类，对每个全局聚类进行局部聚类（按照全局聚类同样的聚类数）
    all_local_clusters = [np.array([]) for _ in range(len(embeddings))]
    total_clusters = 0
    for i in range(n_global_clusters):
        # 获取属于当前类别下的所有数据点（如：第一类，则i=0）
        global_cluster_embeddings_ = embeddings[
            np.array([i in gc for gc in global_clusters])
        ]
        # 如果当前类别下没有数据点，则跳过
        if len(global_cluster_embeddings_) == 0:
            continue
        # 如果当前类别下数据点太少，直接返回所有点属于同一个类
        if len(global_cluster_embeddings_) <= dim + 1:
            local_clusters = [np.array([0]) for _ in global_cluster_embeddings_]
            n_local_clusters = 1
        else:
            # 对当前类别下的数据点进行局部降维
            reduced_embeddings_local = local_cluster_embeddings(
                global_cluster_embeddings_,
                dim
            )
            # 对局部聚类后的数据点进行高斯混合模型聚类
            local_clusters, n_local_clusters = GMM_cluster(
                reduced_embeddings_local,
                threshold
            )
        
        # 对于每个局部聚类
        for j in range(n_local_clusters):
            # 获取属于当前局部类别下的所有数据点（如：第一类，则j=0）
            local_cluster_embeddings_ = global_cluster_embeddings_[
                np.array([j in lc for lc in local_clusters])
            ]
            # 找到这些数据点在原始数据中的索引
            indices = np.where(
                (embeddings == local_cluster_embeddings_[:, None]).all(-1)
            )[1]
            # 将局部聚类标签（加上偏移量）添加到对应数据点点标签列表中
            for idx in indices:
                all_local_clusters[idx] = np.append(
                    all_local_clusters[idx], j + total_clusters
                )
        total_clusters += n_local_clusters
    
    return all_local_clusters

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from dotenv import load_dotenv
import os

load_dotenv()

embd = OpenAIEmbeddings(
    model="text-embedding-3-small",
    api_key=os.getenv("API_KEY"),
    base_url=os.getenv("BASE_URL")
)
model = ChatOpenAI(
    model="gpt-4o",
    api_key=os.getenv("API_KEY"),
    base_url=os.getenv("BASE_URL")
)

def embed(texts):
    """
    给文档列表生成嵌入向量
    """
    text_embeddings = embd.embed_documents(texts)
    text_embeddings_np = np.array(text_embeddings)
    return text_embeddings_np

def embed_cluster_texts(texts):
    """
    给文档列表生成嵌入向量，返回数据表（文本，他们的嵌入向量，聚类标签）
    """
    text_embeddings_np = embed(texts)
    cluster_labels = perform_clustering(
        embeddings=text_embeddings_np,
        threshold=0.1,
        dim=10
    )
    df = pd.DataFrame(
        {
            "text": texts,
            "embd": list(text_embeddings_np),
            "cluster_label": cluster_labels
        }
    )
    return df

def fmt_txt(df: pd.DataFrame) -> str:
    """
    将数据表格式化为文本
    """
    unique_txt = df['text'].tolist()
    return "--- --- \n --- --- ".join(unique_txt)

def embed_cluster_summarize_texts(
    texts: List[str],
    level: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    对文本列表进行嵌入、聚类和汇总
    这是 RAPTOR 算法的核心函数，它：
    1. 为文本生成嵌入向量
    2. 根据相似性进行聚类
    3. 对每个聚类内的文本进行摘要
    4. 返回原始聚类结果和摘要结果

    Args:
        texts: 文本列表
        level: 当前聚类层级

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: 
        数据表1（原始文本，他们的嵌入向量，聚类标签），
        数据表2（每个类的摘要，层级信息）
    """
    df_clusters = embed_cluster_texts(texts)
    expanded_list = []
    for index, row in df_clusters.iterrows():
        for cluster in row['cluster_label']:
            expanded_list.append(
                {
                    "text": row['text'],
                    "embd": row['embd'],
                    "cluster_label": cluster,
                }
            )
            
    expanded_df = pd.DataFrame(expanded_list)
    all_clusters = expanded_df['cluster_label'].unique()
    print(f"-- Generated{len(all_clusters)} clusters --")
    
    template = """Here is a sub-set of LangChain Expression Language doc. 
    
    LangChain Expression Language provides a way to compose chain in LangChain.
    
    Give a detailed summary of the documentation provided.
    
    Documentation:
    {context}
    """
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | model | StrOutputParser()
    
    summaries = []
    for i in all_clusters:
        df_clusters = expanded_df[expanded_df['cluster_label'] == i]
        formatted_txt = fmt_txt(df_clusters)
        summaries.append(chain.invoke({"context": formatted_txt}))
        
    df_summary = pd.DataFrame(
        {
            "summaries": summaries,
            "level": [level] * len(summaries),
            "cluster_label": list(all_clusters)
        }
    )
    return df_clusters, df_summary

def recursive_embed_cluster_summarize(
    texts: List[str],
    level: int=1,
    n_levels: int=3,
) -> Dict[int, Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    递归嵌入，聚类和汇总文本，直到指定级别
    这是 RAPTOR 算法的顶层函数，它：
    1. 对当前层级的文本进行聚类和摘要
    2. 将摘要作为下一层级的输入
    3. 递归重复直到达到最大层级或只剩一个聚类
    4. 返回所有层级的聚类和摘要结果

    Args:
        texts: 文本列表
        level: 当前级别（由1开始）
        n_levels: 最大级别

    Returns:
        Dict[int, Tuple[pd.DataFrame, pd.DataFrame]]: 
        每个级别的聚类结果和摘要
    """
    results = {}
    
    df_clusters, df_summary = embed_cluster_summarize_texts(texts, level)
    
    results[level] = (df_clusters, df_summary)
    
    unique_clusters = df_summary['cluster_label'].nunique()
    if level < n_levels and unique_clusters > 1:
        new_texts = df_summary['summaries'].tolist()
        next_level_results = recursive_embed_cluster_summarize(
            new_texts,
            level=level+1,
            n_levels=n_levels
        )
        results.update(next_level_results)
        
    return results