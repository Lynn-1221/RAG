### helper_utils.py
# 辅助函数

import numpy as np
import chromadb
import pandas as pd
from pypdf import PdfReader
import numpy as np

def project_embeddings(embeddings, umap_transform):
    """
    对高维向量进行降维度处理，以便于可视化分析（理解文档之间相似性&聚类情况，知识库优化：发现冗余&识别盲点等）
    使用的是 UMAP 算法，是一种无监督的降维算法，可以保留数据的局部结构，同时将高维数据映射到低维空间。
    
    Args:
    embeddings (numpy.ndarray): The embeddings to project.
    umap_transform (umap.UMAP): The trained UMAP transformer.

    Returns:
    numpy.ndarray: The projected embeddings.
    """
    projected_embeddings = umap_transform.transform(embeddings)
    return projected_embeddings


def word_wrap(text, width=87):
    """
    将文本按指定宽度换行，以便于阅读。

    Args:
    text (str): The text to wrap.
    width (int): The width to wrap the text to.

    Returns:
    str: The wrapped text.
    """
    return "\n".join([text[i : i + width] for i in range(0, len(text), width)])


def extract_text_from_pdf(file_path):
    """
    从PDF文件中提取文本。

    Args:
    file_path (str): The path to the PDF file.

    Returns:
    str: The extracted text.
    """
    text = []
    with open(file_path, "rb") as f:
        pdf = PdfReader(f)
        for page_num in range(pdf.get_num_pages()):
            page = pdf.get_page(page_num)
            text.append(page.extract_text())
    return "\n".join(text)


def load_chroma(filename, collection_name, embedding_function):
    """
    从PDF文件中加载文档，提取文本，生成向量，并存储在Chroma集合中
    
    Args:
    filename (str): The path to the PDF file.
    collection_name (str): The name of the Chroma collection.
    embedding_function (callable): A function to generate embeddings.

    Returns:
    chroma.Collection: The Chroma collection with the document embeddings.
    """
    # 从PDF文件中提取文本
    text = extract_text_from_pdf(filename)

    # 将文本按段落分割
    paragraphs = text.split("\n\n")

    # 为每个段落生成向量
    embeddings = [embedding_function(paragraph) for paragraph in paragraphs]

    # 创建一个DataFrame来存储文本和向量
    data = {"text": paragraphs, "embeddings": embeddings}
    df = pd.DataFrame(data)

    # 加载或创建向量

    collection = chromadb.Client().create_collection(collection_name)

    # 逐一将数据存储进向量数据表中
    for ids, row in df.iterrows():
        collection.add(ids=ids, documents=row["text"], embeddings=row["embeddings"])

    return collection
