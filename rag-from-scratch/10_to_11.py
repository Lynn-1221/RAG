### 模块3：Routing: 路由（选择合适的 DB）：基于逻辑 & 基于语义
# 方法1：逻辑及语义路由，通过系统提示词，让 LLM 基于逻辑或语义决定要用什么库
# 方法2：定义结构化查询的 Schema（基于 Pydantic 的 BaseModel），让 LLM 将自然语言转换成结构化查询语言（llm.with_structured_output）
import os, warnings
warnings.filterwarnings("ignore")
from dotenv import load_dotenv
load_dotenv()

api_key = os.getenv("API_KEY")
base_url = os.getenv("BASE_URL")

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
llm = ChatOpenAI(
    model="gpt-4o",
    api_key=api_key,
    base_url=base_url
)

### Part 10: 逻辑及语义路由（使用 function calling 用于分类）
from typing import Literal, Optional
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from pydantic import BaseModel, Field
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

# 10.1 构建逻辑路由器
class RouteQuery(BaseModel):
    """
    将用户问题根据编程语言类型路由到相应的文档数据库
    """
    # 使用 Literal 类型来限制 datasource 的值只能为 python_docs, js_docs, golang_docs
    datasource: Literal["python_docs", "js_docs", "golang_docs"] = Field(
        ...,
        description="Given a user question choose which datasource would be most relevant for answering their question"
    )
# 将 LLM 配置为输出符合 RouteQuery 模型的结构化数据
structured_llm = llm.with_structured_output(RouteQuery)

# 系统提示词定义 LLM 的角色和任务
system = """You are an expert at routing a user question to the appropriate data source.
Based on the programming language the question is referring to, route it to the relevant data source."""
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("user", "{question}")
    ]
)
# 用户查询处理链：接收用户问题 -> 通过 LLM 分析问题内容 -> 将查询路由到相应的文档数据库 -> 返回结构化的路由结果
router = prompt | structured_llm

# 使用示例：
question = """Why doesn't the following code work:

from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages(["human", "speak in {language}"])
prompt.invoke("french")
"""
result = router.invoke({"question": question})
print(result, result.datasource)

def choose_route(result):
    if "python_docs" in result.datasource.lower():
        return "chain for python_docs"
    elif "js_docs" in result.datasource.lower():
        return "chain for js_docs"
    else:
        return "golang_docs"

full_chain = router | RunnableLambda(choose_route)

result = full_chain.invoke({"question": question})
print(result)

# 10.2 构建语义路由器
from langchain.utils.math import cosine_similarity
from langchain_core.output_parsers import StrOutputParser

# 两种不同语义指令模板
physics_template = """You are a very smart physics professor. \
You are great at answering questions about physics in a concise and easy to understand manner. \
When you don't know the answer to a question you admit that you don't know.

Here is a question:
{query}"""

math_template = """You are a very good mathematician. You are great at answering math questions. \
You are so good because you are able to break down hard problems into their component parts, \
answer the component parts, and then put them together to answer the broader question.

Here is a question:
{query}"""

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    api_key=api_key,
    base_url=base_url
)
prompt_templates = [physics_template, math_template]
# 使用同步方法获取 embeddings
prompt_embeddings = embeddings.embed_documents(prompt_templates)

def prompt_router(input):
    # 将用户查询转换为向量
    query_embedding = embeddings.embed_query(input["query"])
    # 计算相似度
    similarity = cosine_similarity([query_embedding], prompt_embeddings)[0]
    most_similar = prompt_templates[similarity.argmax()]
    # 选择合适的指令模板
    print("Using MATH" if most_similar == math_template else "Using PHYSICS")
    return PromptTemplate.from_template(most_similar)

chain = (
    {"query": RunnablePassthrough()}
    | RunnableLambda(prompt_router)
    | llm
    | StrOutputParser()
)

# 使用同步方式调用
result = chain.invoke("What's a black hole？")
print(result)

### Part 11: 元数据过滤器的查询结构
from langchain_community.document_loaders import YoutubeLoader
docs = YoutubeLoader.from_youtube_url(
    "https://www.youtube.com/watch?v=pbAd8O1Lvm4", add_video_info=True
).load()
print(docs[0].metadata)

# 我们希望结合非结构化查询（内容 & 标题）和结构化查询（观看次数，发布日期，视频长度等）来检索视频
# 我们希望将自然语言转换成结构化查询语言，因此需要定义结构化查询的 Schema
# 11.1 定义结构化查询的 Schema
import datetime

class TutorialSearch(BaseModel):
    """
    从教学视频数据库中检索与软件库相关的
    """
    content_search: str = Field(
        ...,
        description="Similarity search query applied to video transcripts."
    )
    title_search: str = Field(
        ...,
        description=(
            "Alternate version of the content search query to apply to video titles. "
            "Should be succinct and only include key words that could be in a video "
            "title."
        ),
    )
    min_view_count: Optional[int] = Field(
        None,
        description="Minimum view count filter, inclusive. Only use if explicitly specified."
    )
    max_view_count: Optional[int] = Field(
        None,
        description="Maximum view count filter, exclusive. Only use if explicitly specified."
    )
    earliest_publish_date: Optional[datetime.date] = Field(
        None,
        description="Earliest publish date filter, inclusive. Only use if explicitly specified."
    )
    latest_publish_date: Optional[datetime.date] = Field(
        None,
        description="Latest publish date filter, exclusive. Only use if explicitly specified.",
    )
    min_length_sec: Optional[int] = Field(
        None,
        description="Minimum video length in seconds, inclusive. Only use if explicitly specified.",
    )
    max_length_sec: Optional[int] = Field(
        None,
        description="Maximum video length in seconds, exclusive. Only use if explicitly specified.",
    )
    
    def pretty_print(self) -> None:
        for field in self.__fields__:
            if getattr(self, field) is not None and getattr(self, field) != getattr(
                self.__fields__[field], "default", None
            ):
                print(f"{field}: {getattr(self, field)}")
                
# 11.2 使用 LLM 将自然语言转换成结构化查询语言
system = """You are an expert at converting user questions into database queries. \
You have access to a database of tutorial videos about a software library for building LLM-powered applications. \
Given a question, return a database query optimized to retrieve the most relevant results.

If there are acronyms or words you are not familiar with, do not try to rephrase them."""
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("user", "{question}")
    ]
)
structured_llm = llm.with_structured_output(TutorialSearch)
query_analyzer = prompt | structured_llm
query_analyzer.invoke({"question": "rag from scratch"}).pretty_print()
query_analyzer.invoke({"question": "videos on chat langchain published in 2023"}).pretty_print()
query_analyzer.invoke({"question": "videos that are focused on the topic of chat langchain that are published before 2024"}).pretty_print()