### 模块4：索引相关方法：
# 1. 多表示索引：将文档的多个表示形式存储在向量数据库中，以便在检索时使用
# 2. RAPTOR：递归聚类和摘要文档，构建语义树状结构
# 3. ColBERT：优化版的密集段落检索，将文档和 Query 都转换成 token embeddings，然后取最大相似度之和作为文档的分数
import os, warnings
warnings.filterwarnings("ignore")
from dotenv import load_dotenv
load_dotenv()

api_key = os.getenv("API_KEY")
base_url = os.getenv("BASE_URL")

from langchain_openai import ChatOpenAI
llm = ChatOpenAI(
    model="gpt-4o",
    api_key=api_key,
    base_url=base_url
)

### Part 12: 多表示索引（添加摘要，假设性问题，图文说明等能优化检索效果）
# 12.1 加载文档
from langchain_community.document_loaders import WebBaseLoader
loader = WebBaseLoader("https://lilianweng.github.io/posts/2023-06-23-agent/")
docs = loader.load()
loader = WebBaseLoader("https://lilianweng.github.io/posts/2024-02-05-human-data-quality/")
docs.extend(loader.load())
print("A. Total docs:")
print(len(docs))

# 12.2 摘要生成
import uuid
from langchain_core.documents import Document # 用于创建文档对象
from langchain_core.output_parsers import StrOutputParser # 用于将模型输出转换为字符串
from langchain_core.prompts import ChatPromptTemplate # 用于创建提示词模板
chain = (
    {"doc": lambda x: x.page_content} # 提取文档内容
    | ChatPromptTemplate.from_template(
        """
        You are a helpful assistant that summarizes documents.
        Here is the document:
        {doc}
        """
    ) # 使用提示模板格式化文档内容
    | llm
    | StrOutputParser()
)
# 批量处理所有文档，生成摘要
summaries = chain.batch(docs, {"max_concurrency": 5}) 
print("B. Total summaries:")
print(len(summaries))
print(summaries[0])
print("--------------------------------")

# 12.3 向量存储
from langchain.storage import InMemoryByteStore # 用于存储文档的二进制数据
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.retrievers.multi_vector import MultiVectorRetriever

embedding_function = OpenAIEmbeddings(
    model="text-embedding-3-small",
    api_key=api_key,
    base_url=base_url
)
vectorstore = Chroma(
    collection_name="summaries",
    embedding_function=embedding_function
)
# store = InMemoryByteStore()  # 创建内存字节存储实例
# id_key = "doc_id"
# # 创建多向量检索器，能够同时处理原始文档和文档的摘要表示
# retriever = MultiVectorRetriever(
#     vectorstore=vectorstore,
#     byte_store=store,
#     id_key=id_key
# )
# doc_ids = [str(uuid.uuid4()) for _ in docs] # 为每个文档生成一个唯一的标识符
# # 创建摘要文档列表，每个摘要文档都包含对应的文档 ID
# summary_docs = [
#     Document(page_content=s, metadata={id_key: doc_ids[i]})
#     for i, s in enumerate(summaries)
# ]
# # 将摘要文档添加到向量存储中
# retriever.vectorstore.add_documents(summary_docs)  
# # 将原始文档添加到文档存储中，与摘要文档通过 doc_id 关联
# retriever.docstore.mset(list(zip(doc_ids, docs)))

# # 12.4 相似度搜索
# # 12.4.1 使用向量存储执行相似度搜索，检索与查询最相关的文档
# query = "Memory in agents"
# sub_docs = vectorstore.similarity_search(query, k=1)
# print("C. Similarity search:")
# print(sub_docs[0])
# print("--------------------------------")

# # 12.4.2 使用多向量检索获取相关文档
# retrieved_docs = retriever.invoke(query, n_results=1)
# # retrieved_docs = retriever.get_relevant_documents(query, n_results=1)
# print("D. Multi-vector retrieval:")
# print(retrieved_docs[0].page_content[:500])
# print("--------------------------------")


# ### Part 13: RAPTOR
# # 核心思想：通过构建语义树状结构，对文档进行递归式抽象和组织，使 LLM 能够高效地检索和理解文档
# # 步骤：
# # 1. 将文档组织成一棵树：树叶节点：原始文档片段；中间节点：每层递归由 LLM 对子节点进行抽象总结；根节点：文档或文集的总体总结；
# # 2. 在检索时按需扩展：根据问题，从树顶开始，向下展开最相关路径；最终选择多个相关片段作为上下文，送入 LLM 生成答案
# # 算法构建流程：
# # 1. 文档构建阶段（离线）
# # 1.1 将文档分成基本快（如段落）
# # 1.2 以固定粒度构建树（如3个叶子组成一个中间节点）
# # 1.3 使用 LLM 对每组子节点生成摘要，作为父节点内容；
# # 1.4 递归执行，直到树根；（直到摘要数量为1）
# # 2. 检索阶段（在线）
# # 2.1 输入查询
# # 2.2 对树顶节点做语义匹配，选择最相关的路径向下展开；
# # 2.3 展开到叶子节点或某一深度后，取出相关节点文本；
# # 2.4 将这些片段作为上下文，联合用户问题，输入到 LLM 生成答案；

# # 13.1 加载文档
# from langchain_community.document_loaders.recursive_url_loader import RecursiveUrlLoader
# from bs4 import BeautifulSoup as Soup

# url = "https://python.langchain.com/docs/expression_language/"
# loader = RecursiveUrlLoader(
#     url=url, max_depth=20, extractor=lambda x: Soup(x, "html.parser").text
# )
# docs = loader.load()
# url = "https://python.langchain.com/docs/how_to/output_parser_structured/"
# loader = RecursiveUrlLoader(
#     url=url,max_depth=1, extractor=lambda x: Soup(x, "html.parser").text
# )
# docs_pydantic = loader.load()
# url = "https://python.langchain.com/docs/how_to/self_query/"
# loader = RecursiveUrlLoader(
#     url=url, max_depth=1, extractor=lambda x: Soup(x, "html.parser").text
# )
# docs_sq = loader.load()
# docs.extend([*docs_pydantic, *docs_sq])
# docs_texts = [d.page_content for d in docs]
# print("E. Total docs:")
# print(len(docs))
# print("--------------------------------")

# # 13.2 可视化文档长度分布
# import matplotlib.pyplot as plt
# from bs4 import BeautifulSoup as Soup
# from helper_utils import num_tokens_from_string

# counts = [num_tokens_from_string(d, "cl100k_base") for d in docs_texts]
# plt.figure(figsize=(10, 6))
# plt.hist(counts, bins=30, color="skyblue", edgecolor="black", alpha=0.7)
# plt.title("Token Count Distribution")
# plt.xlabel("Token Count")
# plt.ylabel("Frequency")
# plt.grid(axis="y", alpha=0.7)
# plt.show()
# print(len(docs))

# # 13.3 将文档连接成一个字符串
# d_sorted = sorted(docs, key=lambda x: x.metadata['source'])
# d_reversed = list(reversed(d_sorted))
# concatenated_content = "\n\n\n --- \n\n\n".join(
#     [d.page_content for d in d_reversed]
# )
# print("F. Num tokens in all context: %s" % num_tokens_from_string(concatenated_content, "cl100k_base"))

# # 13.4 文本分割
# from langchain_text_splitters import RecursiveCharacterTextSplitter

# text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
#     chunk_size=2000,
#     chunk_overlap=0
# )
# texts_split = text_splitter.split_text(concatenated_content)

# # 13.5 使用 RAPTOR 聚类和摘要文档
# from clusters import recursive_embed_cluster_summarize

# leaf_texts = docs_texts
# results = recursive_embed_cluster_summarize(
#     leaf_texts,
#     level=1,
#     n_levels=3
# )

# from langchain.schema import Document

# all_texts = leaf_texts.copy()
# for level in sorted(results.keys()):
#     summaries = results[level][1]['summaries'].tolist()
#     all_texts.extend(summaries)
# documents = [Document(page_content=t) for t in all_texts]
    
# vectorstore = Chroma.from_documents(
#     documents=documents,
#     embedding=embedding_function,
#     collection_name="raptor_summaries"
# )
# retriever = vectorstore.as_retriever()

# from langchain import hub
# from langchain_core.runnables import RunnablePassthrough
# from helper_utils import format_docs

# # 从 langchain hub 中获取一个预定义的 rag 提示模板
# prompt = hub.pull("rlm/rag-prompt")

# raptor_rag_chain = (
#     {"context": retriever | format_docs, "question": RunnablePassthrough()}
#     | prompt
#     | llm
#     | StrOutputParser()
# )
# question = "How to define a RAG chain? Give me a specific code example."
# response = raptor_rag_chain.invoke(question)
# print("I. RAG chain:")
# print(response)
# print("--------------------------------")

### Part 14: ColBERT
# 背景：DPR检索效果极大受限于嵌入模型（嵌入模型需要识别所有相关的搜索词），而对于不常见的属于，可能效果不佳，若效果不佳，可能需要重建索引
# 说明：DPR: 密集段落检索，将查询和段落转换为向量，然后计算余弦相似度，返回相似度最高的段落
# 核心思想：为段落中的每个 token 生成一个受上下文影响的向量，同样，也为查询中的每个 token 生成向量
# 然后，每个文档的分数是每个查询嵌入于任意文档嵌入的最大相似度之和
# 步骤：
# 1. 构建离线索引
# 1.1 将文档分词并编码（BERT)，每个 token 生成一个向量，构建一组 token_embeddings
# 1.2 压缩和量化（为节省内存，可用 Product Quantization，FAISS-compatible等对每个 token_embeddings 进行压缩） 
# 1.3 构建索引，将所有文档的 token_embeddings 存储到向量数据库中（如 FAISS，HNSW等）
# 2. 在线查询于检索阶段
# 2.1 将查询 Q 用 BERT 编码，得到 token_embeddings 序列；
# 2.2 计算 maxsim 相似度，得到每个文档的分数；（每个文档所有 token 向量的最大相似度）
# 2.3 得分排序，返回 top-k 个文档；

def maxsim(qv, documnet_embeddings):
    return max(qv @ dv for dv in documnet_embeddings)

def score(query_embeddings, document_embeddings):
    return sum(maxsim(qv, document_embeddings) for qv in query_embeddings)

from ragatouille import RAGPretrainedModel
from helper_utils import get_wikipedia_page

# 14.1 加载与训练的 colbert 2.0 模型
RAG = RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0")
full_document = get_wikipedia_page("Hayao_Miyazaki")
# 14.2 对文档进行索引
RAG.index(
    collection=[full_document], # 要索引的文档列表
    index_name="Miyazaki-123",  # 索引名称
    max_document_length=300,    # 每个文档片段的最大长度
    split_documents=True,       # 是否将文档拆分成多个段落
)
# 14.3 对查询进行检索
results = RAG.search(
    query="What animation studio did Miyazaki found?",
    index_name="Miyazaki-123",
    k=3,
)
print("G. ColBERT search:")
print(results)
print("--------------------------------")

# 14.4 将 RAG 模型转换成 LangChain 兼容的检索器
retriever = RAG.as_langchain_retriever(k=3)
results = retriever.invoke("What animation studio did Miyazaki found?")
print("H. ColBERT retriever:")
print(results)
print("--------------------------------")
