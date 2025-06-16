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

def create_docs(user_pdf_list):
    for pdf in user_pdf_list:
        print("Processing -- ", pdf.name)
        loader = PyPDFLoader(pdf)
        pages = loader.load_and_split()
        embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            base_url=base_url,
            api_key=api_key)
        
        vectorstore = FAISS.from_documents(pages, embeddings)

        template = ChatPromptTemplate.from_template(
            """
            Extract all the following values: invoice no., Description, Quantity, date, Unit price, Amount, Total, email, 
            phone number, and address from the following Invoice content
            (create a JSON output with the extracted fields only):
            {context}
            The fields and values in the above content may be jumbled up as they are extracted from a PDF. 
            Please extract the fields and values correctly based on the fields asked for in the question above.
            Expected JSON output format as follows:
            {{'Invoice no.': 'xxxxxxx', 'Description': 'xxxxxxx', 'Quantity': 'x', 'Date': 'dd/mm/yyyy', 'Unit price': 'xx.xx', 'Amount': 'xxx.xx', 'Total': 'xxx.xx', 'Email': 'xxx@xxx.xxx', 'Phone number': 'xxx-xxx-xxxx'}}
            Remove any dollar symbols or currency symbols from the extracted values.
            """
        )

        prompt = ChatPromptTemplate(template)

        llm = ChatOpenAI(
            model_name="gpt-4o",
            base_url=base_url,
            api_key=api_key)
        retriever = vectorstore.as_retriever()
        document_chain = create_stuff_documents_chain(llm, prompt)
        retrieval_chain = create_retrieval_chain(retriever, document_chain)
        response = retrieval_chain.invoke({"input": "Extract all the following values: invoice no., Description, Quantity, date, Unit price, Amount, Total, email, phone number, and address from the following Invoice content"})
        answer_content = response['answer']

        print("Extracted data: ")
        print(answer_content)
        print("-------------------------------- done --------------------------------")
        return answer_content