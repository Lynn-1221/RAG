from langchain_openai import OpenAIEmbeddings  # 导入OpenAI Embeddings用于文本向量化
from langchain_community.vectorstores import FAISS  # 导入FAISS用于向量数据库
from langchain.chains.combine_documents import create_stuff_documents_chain  # 导入文档链工具
from langchain_core.prompts import ChatPromptTemplate  # 导入提示模板
from langchain_openai import ChatOpenAI  # 导入OpenAI聊天模型
from langchain.chains import create_retrieval_chain  # 导入检索链工具
from langchain_community.document_loaders import PyPDFLoader  # 导入PDF加载器
import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
base_url = os.getenv("BASE_URL")

# 加载本地PDF文件并切分为页面
loader = PyPDFLoader("data/test_pdf.pdf")
pages = loader.load_and_split()
print(len(pages))  # 打印切分后的页面数量

# 初始化OpenAI聊天模型
model = ChatOpenAI(
    base_url=base_url,  # 自定义API服务地址
    api_key=api_key,  # OpenAI API密钥
    model_name="gpt-4o",  # 使用的模型名称
    temperature=0.5  # 温度参数，控制生成内容的多样性
)

# 初始化OpenAI Embeddings，用于将文本转为向量
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",  # 嵌入模型名称
    base_url=base_url,  # 自定义API服务地址
    api_key=api_key  # OpenAI API密钥
)

# 用文档和嵌入模型初始化FAISS向量数据库
vectorstore = FAISS.from_documents(pages, embeddings)

# 构建提示模板，{context} 会被检索到的内容填充，{input} 是用户问题
# prompt = ChatPromptTemplate.from_template(
#     "请基于给到的材料：{context} ; 回答以下问题：{input}"
# )
prompt = ChatPromptTemplate.from_template(
    """Answer the following question based on the provided context:
    <context>
    {context}
    </context>

    Question: {input}"""
)

# 创建文档链，将模型和提示模板结合
document_chain = create_stuff_documents_chain(model, prompt)
# 获取检索器
retriever = vectorstore.as_retriever()
# 创建检索链，将检索器和文档链结合
retrieval_chain = create_retrieval_chain(retriever, document_chain)

# 调用检索链，传入用户问题
# response = retrieval_chain.invoke({"input": "请按顺序列出以下关键信息：1. 案件编号 2. 案件名称 3. 案件类型 4. 案件状态"})
response = retrieval_chain.invoke({"context": "You are a lawyer who is creating an the gist for a given court order in the context.",
                                    "input": """Please list down the key points from the order.
                                    Please mention the case title, pwtitioner and defendant names at the top."""})

# 输出模型回答
print(response['answer'])