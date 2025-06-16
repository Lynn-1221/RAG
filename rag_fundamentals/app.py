import os
from dotenv import load_dotenv
import chromadb
from openai import OpenAI
from langchain_openai import OpenAIEmbeddings
from typing import List, Dict, Any
import logging
from pathlib import Path

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 加载环境变量
load_dotenv()

# 初始化配置
api_key = os.getenv("API_KEY")
base_url = os.getenv("BASE_URL")

if not api_key:
    raise ValueError("API_KEY not found in environment variables")

# 自定义Embedding函数，用于ChromaDB
class CustomEmbeddingFunction:
    def __init__(self, api_key: str, base_url: str):
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            base_url=base_url,
            api_key=api_key
        )
    
    def __call__(self, input: List[str]) -> List[List[float]]:
        return self.embeddings.embed_documents(input)

# 初始化OpenAI客户端和ChromaDB
client = OpenAI(api_key=api_key, base_url=base_url)
embedding_function = CustomEmbeddingFunction(api_key=api_key, base_url=base_url)
chroma_client = chromadb.PersistentClient(path="chroma_persistent_storage")
collection_name = "document_qa_collection"
collection = chroma_client.get_or_create_collection(
    name=collection_name,
    embedding_function=embedding_function
)

def load_documents_from_directory(directory_path: str) -> List[Dict[str, str]]:
    """
    从指定目录加载所有txt文件
    
    Args:
        directory_path: 文档目录路径
        
    Returns:
        包含文档ID和文本的字典列表
    """
    logger.info("Loading documents from directory: %s", directory_path)
    documents = []
    directory = Path(directory_path)
    
    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory_path}")
        
    for file_path in directory.glob("*.txt"):
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                documents.append({
                    "id": file_path.stem,
                    "text": file.read()
                })
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {str(e)}")
            
    return documents

def split_text(text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[str]:
    """
    将文本分割成重叠的块
    
    Args:
        text: 要分割的文本
        chunk_size: 每个块的最大字符数
        chunk_overlap: 块之间的重叠字符数
        
    Returns:
        文本块列表
    """
    chunks = []
    start = 0
    text_length = len(text)
    
    while start < text_length:
        # 确保不会超出文本长度
        end = min(start + chunk_size, text_length)
        # 如果不是最后一块，尝试在句子边界处分割
        if end < text_length:
            # 在chunk_size范围内找到最后一个句号或换行符
            last_period = text[start:end].rfind('.')
            last_newline = text[start:end].rfind('\n')
            split_point = max(last_period, last_newline)
            if split_point > 0:
                end = start + split_point + 1
        
        chunks.append(text[start:end].strip())
        start = end - chunk_overlap
        
    return chunks

def get_openai_embedding(text: str) -> List[float]:
    """
    获取文本的OpenAI嵌入向量
    
    Args:
        text: 要获取嵌入向量的文本
        
    Returns:
        文本的嵌入向量
    """
    try:
        response = client.embeddings.create(
            input=text,
            model='text-embedding-3-small'
        )
        return response.data[0].embedding
    except Exception as e:
        logger.error(f"Error getting embedding: {str(e)}")
        raise

def process_documents(directory_path: str) -> None:
    """
    处理文档：加载、分块、向量化并存储到ChromaDB
    
    Args:
        directory_path: 文档目录路径
    """
    # 加载文档
    documents = load_documents_from_directory(directory_path)
    
    # 分块处理
    chunked_documents = []
    for doc in documents:
        chunks = split_text(doc["text"])
        logger.info(f"Split document {doc['id']} into {len(chunks)} chunks")
        for i, chunk in enumerate(chunks):
            chunked_documents.append({
                "id": f"{doc['id']}_chunk{i+1}",
                "text": chunk
            })
    
    # 批量处理文档
    batch_size = 10
    for i in range(0, len(chunked_documents), batch_size):
        batch = chunked_documents[i:i + batch_size]
        try:
            # 获取嵌入向量
            embeddings = [get_openai_embedding(doc["text"]) for doc in batch]
            # 存储到ChromaDB
            collection.add(
                documents=[doc["text"] for doc in batch],
                embeddings=embeddings,
                ids=[doc["id"] for doc in batch]
            )
            logger.info(f"Processed batch {i//batch_size + 1}")
        except Exception as e:
            logger.error(f"Error processing batch {i//batch_size + 1}: {str(e)}")

def query_documents(question: str, n_results: int = 3) -> List[str]:
    """
    查询相关文档
    
    Args:
        question: 查询问题
        n_results: 返回结果数量
        
    Returns:
        相关文档块列表
    """
    try:
        results = collection.query(
            query_texts=[question],
            n_results=n_results
        )
        return [doc for sublist in results["documents"] for doc in sublist]
    except Exception as e:
        logger.error(f"Error querying documents: {str(e)}")
        return []

def generate_response(question: str, relevant_chunks: List[str]) -> str:
    """
    生成回答
    
    Args:
        question: 用户问题
        relevant_chunks: 相关文档块
        
    Returns:
        生成的回答
    """
    context = "\n\n".join(relevant_chunks)
    prompt = (
        "You are an assistant for question-answering tasks. Use the following pieces of "
        "retrieved context to answer the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the answer concise.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}"
    )
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": question}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"Error generating response: {str(e)}")
        return "Sorry, I encountered an error while generating the response."

def main():
    """主函数"""
    # 处理文档
    directory_path = "./news_articles"
    process_documents(directory_path)
    
    # 示例查询
    question = "tell me about databricks"
    relevant_chunks = query_documents(question)
    answer = generate_response(question, relevant_chunks)
    print(f"Question: {question}")
    print(f"Answer: {answer}")

if __name__ == "__main__":
    main()