### 模块5: 检索
# 方法1：Re-ranking
# 方法2: CRAG
# 方法3: Self-RAG
# 方法4: Impact of long context
import os
from dotenv import load_dotenv
load_dotenv()

api_key = os.getenv("API_KEY")
base_url = os.getenv("BASE_URL")

# Part 15: Re-ranking
import bs4
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 15.1 加载文档
loader = WebBaseLoader(
    web_path=("https://lilianweng.github.io/posts/2023-06-23-agent/"),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header")
        )
    ))
blog_docs = loader.load()

# 15.2 切分文档
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=300,
    chunk_overlap=50
)
splits = text_splitter.split_documents(blog_docs)

# 15.3 文档向量化
embedding_function = OpenAIEmbeddings(
    model="text-embedding-3-small",
    api_key=api_key,
    base_url=base_url
)
vectorstore = Chroma.from_documents(
    documents=splits,
    embedding=embedding_function
)
retriever = vectorstore.as_retriever()

# 15.4 查询扩展
template = """You are a helpful assistant that generates multiple search queries based on a single input query. \n
Generate multiple search queries related to: {question} \n
Output (4 queries):"""
prompt_rag_fusion = ChatPromptTemplate.from_template(template)

llm = ChatOpenAI(
    model="gpt-4o",
    api_key=api_key,
    base_url=base_url
)
generate_queries_chain = (
    prompt_rag_fusion |
    llm |
    StrOutputParser()|
    (lambda x: x.split("\n"))
)

# 15.5 多路召回（包含重排序）
# 方法1: 使用自建的 RRF 算法，对多个召回结果进行重排序
from helper_utils import reciprocal_rank_fusion
question = "What is task decomposition for LLM agents?"
rag_retrieval_chain = generate_queries_chain | retriever.map() | reciprocal_rank_fusion
docs = rag_retrieval_chain.invoke({"question": question})
print("B. RRF answer:")
print(docs)
print("--------------------------------")

# 方法2: 使用 LangChain 自带的 cohere 接口，对多个召回结果进行重排序
from langchain_community.llms import Cohere
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CohereRerank

retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
compressor = CohereRerank(
    model="rerank-english-v3.0",
    top_n=5,
    api_key=api_key,
    base_url=base_url
)
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=retriever
)
compressed_docs = compression_retriever.get_relevant_documents(question)
print("C. Cohere RRF answer:")
print(compressed_docs)
print("--------------------------------")

# 15.6 生成最终回答
template = """Answer the following question based on this context:

{context}

Question: {question}
"""
from operator import itemgetter
prompt_final_answer = ChatPromptTemplate.from_template(template)
final_rag_chain = (
    {"context": rag_retrieval_chain, "question": itemgetter("question")} |
    prompt_final_answer |
    llm |
    StrOutputParser()
)
final_answer = final_rag_chain.invoke({"question": question})
print("A. Re-ranking answer:")
print(final_answer)
print("--------------------------------")

# Part 16: CRAG
# 核心思路：将对比学习引入 RAG 框架，通过对检索文档进行质量建模，优化生成问答的相关性与准确性 -> 对检索结果进行评分
# 背景：在传统 RAG 框架中，信息检索模块与生成模块往往是独立训练的，这会存在2个问题：检索到的文档可能不适合用作生成上下文；检索系统未考虑文档对最终答案的影响；
# CRAG 核心目标：用对比学习的方法，让检索系统学会选择那些真正对生成答案有用的文档
# 主要步骤：
# 1. 训练一个对比学习模型，用于判断文档是否适合用作生成上下文
# 输入：Q；输出：一组候选文档：正文档与 Q距离更近，负文档与 Q距离更远，正负文档的判别基于答案重构能力；
# 2. 训练一个生成模型，用于生成答案
# 使用标准的 seq2seq 训练目标优化（如：cross-entropy)
# 3. 联合训练优化
# 训练使用 对比损失（正文档得分高；负文档得分低） + 生成损失


# 15.5 多路召回

