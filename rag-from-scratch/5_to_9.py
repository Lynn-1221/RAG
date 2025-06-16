### 模块2：查询语句转换：
# 方法1: 多 Query 检索；基于原始 Query，生成多个不同角度的 Query【查询间可能差异较大】，然后使用简单的驱虫合并；【优点：问题理解更全面，计算开销小】
# 方法2: RAG Fusion: 基于原始 Query，生成多个相关的 Query【查询之间更相似】，然后使用 RRF 算法，对文档进行重排序【优点：结果排序更精准】
# 方法3: Query 拆分：将原始 Query 拆分为多个子问题，通过迭代式问答的方式，回答所有子问题【todo：可在最后加上原始问题的回答】
# 方法4: Step-back prompting：将原始 Query 转换为更抽象的 Query，直到问题可以被回答为止
# 方法5: HyDE：先让大语言模型生成一个对问题的假设性回答，再用这个假设去检索相关文档，从而引导模型更准确地回答原始问题

# 从更少抽象到更多抽象：
# sub-question -> Re-written(RAG-Fusion, Multi-Query) -> Step-back question(Step-back prompting)

import bs4, os
from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser
from operator import itemgetter
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain import hub
import warnings
warnings.filterwarnings("ignore")

# 加载环境变量
load_dotenv()
api_key = os.getenv("API_KEY")
base_url = os.getenv("BASE_URL")

# 前置：读取文档，初始化切分器，嵌入模型，向量检索器
# 1. 读取文档
loader = WebBaseLoader(
    web_path=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header")
        )
    )
)
docs = loader.load()
print(len(docs))
# 2. 文本切分
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50
)
texts = text_splitter.split_documents(docs)
print(len(texts))
# 3. 初始化嵌入模型 & 问答模型
embd = OpenAIEmbeddings(
    model="text-embedding-3-small",
    base_url=base_url,
    api_key=api_key
)
llm = ChatOpenAI(
    model_name="gpt-4o",
    base_url=base_url,
    api_key=api_key
)
# 4. 初始化向量检索器
vectorstore = Chroma.from_documents(
    texts,
    embedding=embd
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

## Part 5: 多 Query 检索
# 5.1 生成多个不同的查询
template = """You are an AI language model assistant. Your task is to generate five 
different versions of the given user question to retrieve relevant documents from a vector 
database. By generating multiple perspectives on the user question, your goal is to help
the user overcome some of the limitations of the distance-based similarity search. 
Provide these alternative questions separated by newlines. Original question: {question}"""
prompt_perspectives = ChatPromptTemplate.from_template(template)
# query生成链：提示模板以指导LLM生成5个不同的query -> 将前一步生成的提示词发送给LLM -> 将 LLM 的输出转换为字符串格式（确保输出是纯文本，便于后续处理）
# -> 将字符串按换行符分割成一个列表（由于 LLM 生成的每个查询问题都是单独一行，需要将它们转换为列表）-> 一个包含5个不同查询问题的列表
generate_queries = (
    prompt_perspectives |
    llm |
    StrOutputParser() |
     (lambda x: x.split("\n"))
)

# 5.2 检索相关文档
from helper_utils import get_unique_union
global_question = "What are the main components of an LLM-powered autonomous agent system?"
# 检索链：生成5个不同的query -> 对每个生成的query进行检索（map()操作会将前一步生成的 query 作为输入，检索得到一个文档列表）-> 去重
retrieval_chain = (
    generate_queries |
    retriever.map() |
    get_unique_union
)
docs = retrieval_chain.invoke({"question": global_question})

# 5.3 生成答案
# RAG
template = """Answer the following question based on this context:

{context}

Question: {question}
"""
rag_prompt = ChatPromptTemplate.from_template(template)
# 答案生成链：输入映射步骤（用于准备传递给提示模板的数据）-> 提示模板（用于构建最终答案生成提示词） -> 问答模型 -> 输出解析器（将模型输出转换为字符串）
rag_chain = (
    {"context": retrieval_chain, "question": itemgetter("question")} |
    rag_prompt |
    llm |
    StrOutputParser()
)

result = rag_chain.invoke({"question": global_question})
print("--------------------------------", "Part 5: 多 Query 检索", "--------------------------------")
print(result)
print("------------------------------------------------------------------------------------------------")

### Part 6: RAG Fusion
# 6.1 RAG-Fusion prompt 生成
template = """You are a helpful assistant that generates multiple search queries based on a single input query. \n
Generate multiple search queries related to: {question} \n
Output (4 queries):"""
prompt_rag_fusion = ChatPromptTemplate.from_template(template)
# 检索链：提示模板以指导LLM生成4个不同的query -> 将前一步生成的提示词发送给LLM -> 将 LLM 的输出转换为字符串格式（确保输出是纯文本，便于后续处理）
# -> 将字符串按换行符分割成一个列表（由于 LLM 生成的每个查询问题都是单独一行，需要将它们转换为列表）-> 一个包含4个不同查询问题的列表
generate_queries = (
    prompt_rag_fusion 
    | llm
    | StrOutputParser() 
    | (lambda x: x.split("\n"))
)

# 6.2 多路召回结果融合
# RAG-Fusion 检索链：生成4个不同的query -> 对每个生成的query进行检索（map()操作会将前一步生成的 query 作为输入，检索得到一个文档列表）
# -> 多路召回结果融合 -> 返回重排序后的文档列表（作为后续的 context）
from helper_utils import reciprocal_rank_fusion
retrieval_chain_rag_fusion = generate_queries | retriever.map() | reciprocal_rank_fusion
docs = retrieval_chain_rag_fusion.invoke({"question": global_question})
print(len(docs))

# 6.3 生成答案
# RAG
template = """Answer the following question based on this context:

{context}

Question: {question}
"""
rag_prompt = ChatPromptTemplate.from_template(template)
final_rag_chain = (
    {"context": retrieval_chain_rag_fusion, "question": itemgetter("question")} |
    rag_prompt |
    llm |
    StrOutputParser()
)
result = final_rag_chain.invoke({"question": global_question})
print("--------------------------------", "Part 6: RAG Fusion", "--------------------------------")
print(result)
print("------------------------------------------------------------------------------------------------")

### Part 7: Decompose the Question
# 7.1 生成多个子问题
template = """You are a helpful assistant that generates multiple sub-questions related to an input question. \n
The goal is to break down the input into a set of sub-problems / sub-questions that can be answers in isolation. \n
Generate multiple search queries related to: {question} \n
Output (3 queries):"""
decompose_prompt = ChatPromptTemplate.from_template(template)
decompose_chain = (
    decompose_prompt |
    llm |
    StrOutputParser() |
    (lambda x: x.split("\n"))
)
questions = decompose_chain.invoke({"question": global_question})
print("--------------------------------", "Part 7: 多个子问题", "--------------------------------")
print(questions)
print("------------------------------------------------------------------------------------------------")

# 7.2 生成语境模板：3个输入：问题、问答历史、上下文
template = """Here is the question you need to answer:

\n --- \n {question} \n --- \n

Here is any available background question + answer pairs:

\n --- \n {q_a_pairs} \n --- \n

Here is additional context relevant to the question: 

\n --- \n {context} \n --- \n

Use the above context and any background question + answer pairs to answer the question: \n {question}
"""
context_template = ChatPromptTemplate.from_template(template)

from helper_utils import format_qa_pair
# 7.3 通过迭代式问答的方式，确保每个子问题都有相关问题的问答历史
q_a_pairs = ""
for q in questions:
    rag_chain = (
        {
            "question": itemgetter("question"), 
            "q_a_pairs": itemgetter("q_a_pairs"), 
            "context": itemgetter("question") | retriever
        } |
        context_template |
        llm |
        StrOutputParser()
    )
    answer = rag_chain.invoke({"question": q, "q_a_pairs": q_a_pairs})
    q_a_pair = format_qa_pair(q, answer)
    q_a_pairs = q_a_pairs + "\n---\n"+  q_a_pair
print("--------------------------------", "Part 7: Decompose the Question", "--------------------------------")
print(answer)
print("------------------------------------------------------------------------------------------------")

### Part 8: Step-back prompting
# 核心思想：将问题向上抽象，直到问题可以被回答为止
# 步骤：先让模型“退一步”，提出一个更高层次、更通用的问题 -> 然后利用这个抽象问题得到的信息作为背景信息指导原始问题的回答
# 8.1 构造 Few shot 示例
examples = [
    {
        "input": "Could the members of The Police perform lawful arrests?",
        "output": "what can the members of The Police do?"
    },
    {
        "input": "Jan Sindel’s was born in what country?",
        "output": "what is Jan Sindel’s personal history?"
    }
]
example_template = ChatPromptTemplate.from_messages(
    [
        ("human", "{input}"),
        ("ai", "{output}")
    ]
)
few_shot_prompt = FewShotChatMessagePromptTemplate(
    example_prompt=example_template,
    examples=examples
)

# 8.2 构建 Step-back 提示词
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system", 
            """You are an expert at world knowledge. Your task is to step back and paraphrase a question to a more generic step-back question, which is easier to answer. Here are a few examples:"""
        ),
        few_shot_prompt,
        ("user", "{question}")
    ]
)
step_back_prompt = prompt | llm | StrOutputParser()

# 8.3 生成抽象问题
step_back_question = step_back_prompt.invoke({"question": global_question})
print("--------------------------------", "Part 8:抽象问题", "--------------------------------")
print(step_back_question)
print("------------------------------------------------------------------------------------------------")

# 8.4 构建答案生成提示词
response_prompt_template = """
You are an expert of world knowledge. I am going to ask you a question. 
Your response should be comprehensive and not contradicted with the following context if they are relevant. 
Otherwise, ignore them if they are not relevant.
# {normal_context}
# {step_back_context}

# Original Question: {question}
# Answer:"""
response_prompt = ChatPromptTemplate.from_template(response_prompt_template)

# 8.5 生成答案
# 答案生成链：
# 1.输入映射部分：
# normal_context: 将原始问题通过 retriever 检索相关文档;
# step_back_context: 将抽象问题通过 retriever 检索相关文档;
# question: 原始问题
# 2.提示词模板部分：组合所有信息，并生成最终的提示词
# 3.问答模型部分：将构建好的提示词发送给 LLM
# 4.输出解析器部分：确保最终输出是纯文本
step_back_rag_chain = (
    {
        "normal_context": RunnableLambda(lambda x: retriever.invoke(x['question'])),
        "step_back_context": step_back_prompt | retriever,
        "question": lambda x: x["question"],
    } |
    response_prompt |
    llm |
    StrOutputParser()
)
result = step_back_rag_chain.invoke({"question": global_question})
print("--------------------------------", "Part 8: Step-back prompting", "--------------------------------")
print(result)
print("------------------------------------------------------------------------------------------------")

### Part 9: HyDE（Hypothetical Document Embeddings）
# 核心思想：先让大语言模型生成一个对问题的假设性回答，再用这个假设去检索相关文档，从而引导模型更准确地回答原始问题
# 9.1 生成假设性回答
template = """Please write a scientific paper passage to answer the question
Question: {question}
Passage:"""
hyde_prompt = ChatPromptTemplate.from_template(template)
generate_docs_for_retrieval = (
    hyde_prompt | llm | StrOutputParser()
)
passage = generate_docs_for_retrieval.invoke({"question": global_question})

# 9.2 用生成的假设性回答检索相关文档
hyde_retrieval_chain = (
    generate_docs_for_retrieval | retriever
)
retrieved_docs = hyde_retrieval_chain.invoke({"question": global_question})
print(len(retrieved_docs))

# 9.3 基于检索结果生成最终回答
template = """Answer the following question based on this context:

{context}

Question: {question}
"""
hyde_final_rag_prompt = ChatPromptTemplate.from_template(template)
hyde_final_rag_chain = (
    hyde_final_rag_prompt |
    llm | 
    StrOutputParser()
)
result = hyde_final_rag_chain.invoke({"question": global_question, "context": retrieved_docs})
print("--------------------------------", "Part 9: HyDE", "--------------------------------")
print(result)
print("------------------------------------------------------------------------------------------------")