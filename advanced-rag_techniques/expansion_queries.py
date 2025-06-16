# 扩展问题
# 1. 将原始问题和扩展后的问题拼接起来
# 2. 查询ChromaDB，获取最相关的文档
# 3. 使用UMAP算法将拼接后的向量降维
# 4. 将拼接后的向量和原始问题的向量进行可视化
# 5. 通过可视化结果，调整问题，以获得更好的答案

from helper_utils import project_embeddings, word_wrap
from pypdf import PdfReader
import os
from openai import OpenAI
from dotenv import load_dotenv
from pypdf import PdfReader
import numpy as np
from umap.umap_ import UMAP
from langchain_openai import OpenAIEmbeddings
from chromadb.utils.embedding_functions import EmbeddingFunction
import chromadb
from typing import List
import warnings
warnings.filterwarnings('ignore')

# 设置环境变量以禁用 tokenizers 警告
os.environ["TOKENIZERS_PARALLELISM"] = "false"

load_dotenv()

api_key = os.getenv("API_KEY")
base_url = os.getenv("BASE_URL")
client = OpenAI(api_key=api_key, base_url=base_url)

reader = PdfReader("data/microsoft-annual-report.pdf")
pdf_texts = [p.extract_text().strip() for p in reader.pages]

# 过滤空白文本
pdf_texts = [text for text in pdf_texts if text]


from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    SentenceTransformersTokenTextSplitter,
)

character_splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", ". ", " ", "", "!", "?"], chunk_size=1000, chunk_overlap=20
)
character_split_texts = character_splitter.split_text("\n\n".join(pdf_texts))


token_splitter = SentenceTransformersTokenTextSplitter(
    chunk_overlap=20, tokens_per_chunk=256
)

token_split_texts = []
for text in character_split_texts:
    token_split_texts += token_splitter.split_text(text)

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

# 将文本添加到ChromaDB中
chroma_client = chromadb.Client()
chroma_collection = chroma_client.create_collection(
    "microsoft-collection", embedding_function=embedding_function
)

ids = [str(i) for i in range(len(token_split_texts))]

chroma_collection.add(ids=ids, documents=token_split_texts)
chroma_collection.count()

query = "What was the total revenue for the year?"

results = chroma_collection.query(query_texts=[query], n_results=5)
retrieved_documents = results["documents"][0]

for document in retrieved_documents:
    print(word_wrap(document))
    print("\n")


def generate_multi_query(query, model="gpt-4o"):

    prompt = """
    You are a knowledgeable financial research assistant. 
    Your users are inquiring about an annual report. 
    For the given question, propose up to five related questions to assist them in finding the information they need. 
    Provide concise, single-topic questions (withouth compounding sentences) that cover various aspects of the topic. 
    Ensure each question is complete and directly related to the original inquiry. 
    List each question on a separate line without numbering.
                """

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
    content = content.split("\n")
    return content


original_query = (
    "What details can you provide about the factors that led to revenue growth?"
)
aug_queries = generate_multi_query(original_query)

# 1. 显示扩展后的问题
for query in aug_queries:
    print("\n", query)

# 2. 将原始问题和扩展后的问题拼接起来
joint_query = [
    original_query
] + aug_queries  # original query is in a list because chroma can actually handle multiple queries, so we add it in a list

print("======> \n\n", joint_query)

results = chroma_collection.query(
    query_texts=joint_query, n_results=5, include=["documents", "embeddings"]
)
retrieved_documents = results["documents"]

# 去重
unique_documents = set()
for documents in retrieved_documents:
    for document in documents:
        unique_documents.add(document)


# 显示结果
for i, documents in enumerate(retrieved_documents):
    print(f"Query: {joint_query[i]}")
    print("")
    print("Results:")
    for doc in documents:
        print(word_wrap(doc))
        print("")
    print("-" * 100)

# 获取所有文档的向量
embeddings = chroma_collection.get(include=["embeddings"])["embeddings"]
umap_transform = UMAP(random_state=0, transform_seed=0).fit(embeddings)
projected_dataset_embeddings = project_embeddings(embeddings, umap_transform)

# 4. 获取原始问题和扩展后的问题的向量
original_query_embedding = embedding_function([original_query])
augmented_query_embeddings = embedding_function(joint_query)

# 将各类向量降维
project_original_query = project_embeddings(original_query_embedding, umap_transform)
project_augmented_queries = project_embeddings(
    augmented_query_embeddings, umap_transform
)

retrieved_embeddings = results["embeddings"]
result_embeddings = [item for sublist in retrieved_embeddings for item in sublist]

projected_result_embeddings = project_embeddings(result_embeddings, umap_transform)

import matplotlib.pyplot as plt


# 绘制各类向量在空间中的投影
plt.figure()
plt.scatter(
    projected_dataset_embeddings[:, 0],
    projected_dataset_embeddings[:, 1],
    s=10,
    color="gray",
)
plt.scatter(
    project_augmented_queries[:, 0],
    project_augmented_queries[:, 1],
    s=150,
    marker="X",
    color="orange",
)
plt.scatter(
    projected_result_embeddings[:, 0],
    projected_result_embeddings[:, 1],
    s=100,
    facecolors="none",
    edgecolors="g",
)
plt.scatter(
    project_original_query[:, 0],
    project_original_query[:, 1],
    s=150,
    marker="X",
    color="r",
)

plt.gca().set_aspect("equal", "datalim")
plt.title(f"{original_query}")
plt.axis("off")
plt.show()  # display the plot
