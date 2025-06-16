import tiktoken, requests
import numpy as np
from langchain.load import dumps, loads

def cosine_similarity(vec1, vec2):
    """
    计算两个向量的余弦相似度
    
    Args:
        vec1: 第一个向量
        vec2: 第二个向量
        
    Returns:
        余弦相似度值
    """
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    return dot_product / (norm_vec1 * norm_vec2)

def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """
    计算文本的 token 数量
    
    Args:
        string: 输入文本
        encoding_name: 编码器名称，例如 'cl100k_base' 是 GPT-4 使用的编码器
        
    Returns:
        token 数量
    """
    # 获取指定编码方式的编码器，例如 'cl100k_base' 是 GPT-4 使用的编码器
    encoding = tiktoken.get_encoding(encoding_name)
    # 使用编码器将文本转换为 token 序列，并返回 token 的数量
    return len(encoding.encode(string))

def format_docs(docs):
    """
    将文档列表格式化为单个字符串
    
    Args:
        docs: 文档列表
        
    Returns:
        格式化后的字符串
    """
    return "\n\n".join(doc.page_content for doc in docs)

def get_unique_union(documents: list[list]):
    """
    获取检索到的文档的唯一并集
    
    Args:
        documents: 检索到的文档列表的列表
        （前一步的结果，每个query都会检索得到一个文档列表，多个query即会有多个文档列表）
        
    Returns:
        唯一并集的文档列表
    """
    # 将文档列表的列表展平，并转换为字符串
    flattened_docs = [dumps(doc) for sublist in documents for doc in sublist]
    # 去重
    unique_docs = list(set(flattened_docs))
    # 将字符串转换为 Document 对象
    return [loads(doc) for doc in unique_docs]

def format_qa_pair(question, answer):
    """
    格式化问题和答案
    
    Args:
        question: 问题
        answer: 答案
        
    Returns:
        格式化后的字符串
    """
    formatted_string = ""
    formatted_string += f"Question: {question}\nAnswer: {answer}\n\n"
    return formatted_string.strip()

from langchain.load import dumps, loads
def reciprocal_rank_fusion(results: list[list], k=60):
    """
    多路召回结果融合，将多个召回结果融合成一个结果，并进行重排序
    
    Args:
        results: 多个召回结果
        k: 重排序的参数
        
    Returns:
        reranked_results: 重排序后的结果
    """
    # 初始化一个字典，用于存储每个文档的融合得分
    fused_scores = {}
    
    # 遍历每个召回结果（多个query同时提问，会得到多个文档列表）
    for docs in results:
        # 遍历每个召回结果中的文档，并获取其排名（每个文档列表中的文档排名是独立的）
        for rank, doc in enumerate(docs):
            # 将文档转换为字符串格式，作为字典的键
            doc_str = dumps(doc)
            # 如果文档不在融合得分字典中，则初始化得分为0（即先前的 query 没有检索到该文档）
            if doc_str not in fused_scores:
                fused_scores[doc_str] = 0
            # 获取该文档的当前得分（RRF得分）
            previous_score = fused_scores[doc_str]
            # 更新文档的得分，使用RRF公式：1 / (rank + k)（rank 是文档在当前文档列表中的排名，k 是重排序的参数）
            fused_scores[doc_str] += 1 / (rank + k)
    # 将文档按照他们的得分排序        
    reranked_results = [
        (loads(doc), score)
        for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    ]
    # 返回重排序后的结果
    return reranked_results

def get_wikipedia_page(title: str):
    """
    从 Wikipedia 获取页面内容
    
    Args:
        title: 页面标题
        
    Returns:
        页面内容
    """
    url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",      # 指定操作为查询
        "format": "json",       # 指定返回格式为 JSON
        "titles": title,        # 要查询的页面标题
        "prop": "extracts",     # 获取页面内容
        "explaintext": True,    # 返回纯文本格式
    }
    headers = {
        "User-Agent": "RAGatouille_tutorial/0.0.1 (ben@clavie.eu)"
    }
    response = requests.get(url, params=params, headers=headers)
    data = response.json()
    # data["query"]["pages"] 是一个字典，我们取第一个值
    page = next(iter(data["query"]["pages"].values()))
    
    # 如果页面存在，返回其内容，否则返回 None
    return page["extract"] if "extract" in page else None