from helper_utils import word_wrap, load_chroma
from pypdf import PdfReader
import os
from openai import OpenAI
from dotenv import load_dotenv


from pypdf import PdfReader
import numpy as np

from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from chromadb.utils.embedding_functions import EmbeddingFunction
import chromadb
from typing import List

import warnings
warnings.filterwarnings('ignore')

# 设置环境变量以禁用 tokenizers 警告
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Load environment variables from .env file
load_dotenv()

api_key = os.getenv("API_KEY")
base_url = os.getenv("BASE_URL")
client = OpenAI(api_key=api_key, base_url=base_url)

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

reader = PdfReader("data/microsoft-annual-report.pdf")
pdf_texts = [p.extract_text().strip() for p in reader.pages]

# Filter the empty strings
pdf_texts = [text for text in pdf_texts if text]

from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    SentenceTransformersTokenTextSplitter,
)

character_splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", ". ", " ", ""], chunk_size=1000, chunk_overlap=0
)
character_split_texts = character_splitter.split_text("\n\n".join(pdf_texts))

token_splitter = SentenceTransformersTokenTextSplitter(
    chunk_overlap=0, tokens_per_chunk=256
)

token_split_texts = []
for text in character_split_texts:
    token_split_texts += token_splitter.split_text(text)

chroma_client = chromadb.Client()
chroma_collection = chroma_client.get_or_create_collection(
    "microsoft-collect", embedding_function=embedding_function
)

# extract the embeddings of the token_split_texts
ids = [str(i) for i in range(len(token_split_texts))]

chroma_collection.add(ids=ids, documents=token_split_texts)

count = chroma_collection.count()

query = "What has been the investment in research and development?"

results = chroma_collection.query(
    query_texts=query, n_results=10, include=["documents", "embeddings"]
)

retrieved_documents = results["documents"][0]

for document in results["documents"][0]:
    print(word_wrap(document))
    print("")

from sentence_transformers import CrossEncoder

# 使用CrossEncoder进行重排序
# Q: CrossEncoder是什么？
# A: CrossEncoder是一种用于重排序的模型，它可以将两个文本作为输入，并输出一个分数，表示两个文本的相似度。
# 
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

pairs = [[query, doc] for doc in retrieved_documents]
scores = cross_encoder.predict(pairs)

print("Scores:")
for score in scores:
    print(score)

print("New Ordering:")
for o in np.argsort(scores)[::-1]:
    print(o + 1)

original_query = (
    "What were the most important factors that contributed to increases in revenue?"
)

generated_queries = [
    "What were the major drivers of revenue growth?",
    "Were there any new product launches that contributed to the increase in revenue?",
    "Did any changes in pricing or promotions impact the revenue growth?",
    "What were the key market trends that facilitated the increase in revenue?",
    "Did any acquisitions or partnerships contribute to the revenue growth?",
]

# concatenate the original query with the generated queries
queries = [original_query] + generated_queries


results = chroma_collection.query(
    query_texts=queries, n_results=10, include=["documents", "embeddings"]
)
retrieved_documents = results["documents"]

# 去重
unique_documents = set()
for documents in retrieved_documents:
    for document in documents:
        unique_documents.add(document)

unique_documents = list(unique_documents)

pairs = []
for doc in unique_documents:
    pairs.append([original_query, doc])

scores = cross_encoder.predict(pairs)

print("Scores:")
for score in scores:
    print(score)

print("New Ordering:")
for o in np.argsort(scores)[::-1]:
    print(o)
# ====
top_indices = np.argsort(scores)[::-1][:5]
top_documents = [unique_documents[i] for i in top_indices]

# 将top文档拼接成一个context
context = "\n\n".join(top_documents)


# 使用OpenAI模型生成最终答案
def generate_multi_query(query, context, model="gpt-4o"):

    prompt = f"""
    You are a knowledgeable financial research assistant. 
    Your users are inquiring about an annual report. 
    """

    messages = [
        {
            "role": "system",
            "content": prompt,
        },
        {
            "role": "user",
            "content": f"based on the following context:\n\n{context}\n\nAnswer the query: '{query}'",
        },
    ]

    response = client.chat.completions.create(
        model=model,
        messages=messages,
    )
    content = response.choices[0].message.content
    content = content.split("\n")
    return content


res = generate_multi_query(query=original_query, context=context)
print("Final Answer:")
print(res)
