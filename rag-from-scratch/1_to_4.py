### 模块1：NAIVE RAG 的基础介绍
# 背景：传统的 RAG 模型，将文档转换为向量，然后计算余弦相似度，返回相似度最高的文档
import os
from dotenv import load_dotenv
import bs4
from langchain import hub
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough # 用于 langchain 中的数据传递
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import numpy as np
import tiktoken # 用于计算文本 token 数量的工具库
from langchain.prompts import ChatPromptTemplate
from helper_utils import cosine_similarity, num_tokens_from_string, format_docs

### 配置环境变量
load_dotenv()

# 获取 API 配置
api_key = os.getenv("API_KEY")
base_url = os.getenv("BASE_URL")

if not api_key:
    raise ValueError("API_KEY not found in environment variables")

### Part 1: 实例化所需模型
# 初始化嵌入模型 & 问答模型
embd = OpenAIEmbeddings(
    model='text-embedding-3-small',
    base_url=base_url,
    api_key=api_key
)
llm = ChatOpenAI(
    model_name="gpt-4o",
    base_url=base_url,
    api_key=api_key,
    temperature=0
)
# 测试相似度计算
test_question = "What kinds of pets do I like?"
test_document = "My favorite pet is a cat."

# 计算 token 数量
token_count = num_tokens_from_string(test_question, "cl100k_base")
print(f"\nToken count for question: {token_count}")

# 计算相似度
query_embedding = embd.embed_query(test_question)
document_embedding = embd.embed_query(test_document)
similarity = cosine_similarity(query_embedding, document_embedding)
print(f"Cosine Similarity: {similarity}")

### Part 2: 将文档索引化
# 创建网页加载器，用于从指定URL加载文档内容
loader = WebBaseLoader(
    # 指定要加载的网页URL，这里是一个关于AI Agent的博客文章
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    # 配置BeautifulSoup解析器，只提取特定class的内容
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            # 只提取包含以下class的HTML元素：
            # post-content: 文章主要内容
            # post-title: 文章标题
            # post-header: 文章头部信息
            class_=("post-content", "post-title", "post-header")
        )
    ),
)
# 加载文档内容
docs = loader.load()
print(docs)

# 文本分割：初始化文本分割器
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
splits = text_splitter.split_documents(docs)

# 创建向量存储
vectorstore = Chroma.from_documents(
    documents=splits,
    embedding=embd
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 1})

### Part 3: 检索
docs = retriever.get_relevant_documents("What is Task Decomposition?")
print(docs)

# Part 4: 生成答案
template = """Answer the question based only on the following context:
{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

# 构建 RAG 链
rag_chain = (
    # retriever | format_docs 是 retriever 的输出作为 format_docs 的输入：将检索出的文本列表转成字符串，该字符串即 ”context"
    # 然后 prompt 的输出作为 llm 的输入
    # 然后 llm 的输出作为 StrOutputParser 的输入
    # 然后 StrOutputParser 的输出作为 rag_chain 的输出
    # RunnablePassthrough() 用于传递用户的问题（不对数据进行任何操作）
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# 示例查询
question = "What is Task Decomposition?"
# 数据流向
# 用户问题 -> RunnablePassthrough() -> prompt的question参数
# 检索文档 -> retriever -> format_docs -> prompt的context参数
result = rag_chain.invoke(question)
print(f"Question: {question}")
print(f"Answer: {result}")