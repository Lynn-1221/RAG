# 扩展答案：
# 1. 将原始问题和假设的答案拼接起来
# 2. 查询ChromaDB，获取最相关的文档
# 3. 使用UMAP算法将拼接后的向量降维
# 4. 将拼接后的向量和原始问题的向量进行可视化
# 5. 通过可视化结果，调整问题，以获得更好的答案

# Q: 为什么要将原始问题和假设的答案拼接起来？
# A: 1. 可以获得更全面的信息，从而获得更准确的答案；

# Q: UMAP算法是什么？
# A: UMAP（Uniform Manifold Approximation and Projection）是一种无监督的降维算法，可以保留数据的局部结构，同时将高维数据映射到低维空间。

from helper_utils import project_embeddings, word_wrap
from pypdf import PdfReader
import os
from openai import OpenAI
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from chromadb.utils.embedding_functions import EmbeddingFunction
from typing import List
import chromadb
from pypdf import PdfReader
from umap.umap_ import UMAP  # 修改 UMAP 的导入方式
import warnings
warnings.filterwarnings("ignore")

# 设置环境变量以禁用 tokenizers 警告
os.environ["TOKENIZERS_PARALLELISM"] = "false"

load_dotenv()

# 获取环境变量
api_key = os.getenv("API_KEY")
base_url = os.getenv("BASE_URL")

if not api_key:
    raise ValueError("API_KEY not found in environment variables")

# 创建自定义的 Embedding 函数
class CustomEmbeddingFunction(EmbeddingFunction):
    def __init__(self, api_key: str, base_url: str):
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            base_url=base_url,
            api_key=api_key
        )
    
    def __call__(self, input: List[str]) -> List[List[float]]:
        return self.embeddings.embed_documents(input)

# 初始化 embedding function
embedding_function = CustomEmbeddingFunction(api_key=api_key, base_url=base_url)

# 创建 ChromaDB 客户端和集合
chroma_client = chromadb.Client()
chroma_collection = chroma_client.create_collection(
    "microsoft-collection", 
    embedding_function=embedding_function
)

reader = PdfReader("data/microsoft-annual-report.pdf")
pdf_texts = [p.extract_text().strip() for p in reader.pages]

# 过滤空白文本
pdf_texts = [text for text in pdf_texts if text]

# 将文本分成小块
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    SentenceTransformersTokenTextSplitter,
)

# 按照标点符号分割文本
character_splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", ". ", " ", "", "!", "?"], chunk_size=1000, chunk_overlap=20
)
character_split_texts = character_splitter.split_text("\n\n".join(pdf_texts))

print(word_wrap(character_split_texts[10]))
print(f"\nTotal chunks(by character): {len(character_split_texts)}")

# 按照句子分割文本
token_splitter = SentenceTransformersTokenTextSplitter(
    chunk_overlap=20, tokens_per_chunk=256
)
token_split_texts = []
for text in character_split_texts:
    token_split_texts += token_splitter.split_text(text)

print(word_wrap(token_split_texts[10]))
print(f"\nTotal chunks(by token): {len(token_split_texts)}")


import chromadb

# 将文本添加到ChromaDB中
ids = [str(i) for i in range(len(token_split_texts))]
chroma_collection.add(ids=ids, documents=token_split_texts)
chroma_collection.count()

query = "What was the total revenue for the year?"


results = chroma_collection.query(query_texts=[query], n_results=5)
retrieved_documents = results["documents"][0]
print("Retrieved documents:")
for document in retrieved_documents:
    print(word_wrap(document))
    print("\n")

client = OpenAI(api_key=api_key, base_url=base_url)

def augment_query_generated(query, model="gpt-4o"):
    prompt = """You are a helpful expert financial research assistant. 
   Provide an example answer to the given question, that might be found in a document like an annual report."""
    messages = [
        {
            "role": "system",
            "content": prompt,
        },
        {"role": "user", "content": query},
    ]

    response = client.chat.completions.create(
        model=model,
        messages=messages,
    )
    content = response.choices[0].message.content
    return content


original_query = "What was the total profit for the year, and how does it compare to the previous year?"
# 生成一个假设的答案
hypothetical_answer = augment_query_generated(original_query)

# 将原始问题和假设的答案拼接起来
joint_query = f"{original_query} {hypothetical_answer}"
print(word_wrap(joint_query))

# 查询ChromaDB，获取最相关的文档
results = chroma_collection.query(
    query_texts=joint_query, n_results=5, include=["documents", "embeddings"]
)
retrieved_documents = results["documents"][0]


# for doc in retrieved_documents:
#     print(word_wrap(doc))
#     print("")
# 获取所有文档的向量
embeddings = chroma_collection.get(include=["embeddings"])["embeddings"]
# 构建UMAP模型
umap_transform = UMAP(random_state=0, transform_seed=0).fit(embeddings)
# 将向量降维
projected_dataset_embeddings = project_embeddings(embeddings, umap_transform)

# 获取检索到的文档的向量
retrieved_embeddings = results["embeddings"][0]
# 获取原始问题的向量
original_query_embedding = embedding_function([original_query])
# 获取拼接后的问题的向量
augmented_query_embedding = embedding_function([joint_query])
# 将各类向量降维
projected_original_query_embedding = project_embeddings(
    original_query_embedding, umap_transform
)
projected_augmented_query_embedding = project_embeddings(
    augmented_query_embedding, umap_transform
)
projected_retrieved_embeddings = project_embeddings(
    retrieved_embeddings, umap_transform
)

import matplotlib.pyplot as plt

# Plot the projected query and retrieved documents in the embedding space
plt.figure()

plt.scatter(
    projected_dataset_embeddings[:, 0],
    projected_dataset_embeddings[:, 1],
    s=10,
    color="gray",
)
plt.scatter(
    projected_retrieved_embeddings[:, 0],
    projected_retrieved_embeddings[:, 1],
    s=100,
    facecolors="none",
    edgecolors="g",
)
plt.scatter(
    projected_original_query_embedding[:, 0],
    projected_original_query_embedding[:, 1],
    s=150,
    marker="X",
    color="r",
)
plt.scatter(
    projected_augmented_query_embedding[:, 0],
    projected_augmented_query_embedding[:, 1],
    s=150,
    marker="X",
    color="orange",
)

plt.gca().set_aspect("equal", "datalim")
plt.title(f"{original_query}")
plt.axis("off")
plt.show()  # display the plot
