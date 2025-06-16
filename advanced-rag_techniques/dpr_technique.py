# 使用DPR (Dense Passage Retriever) 技术，进行语义检索
# 1. 编码问题
# 2. 编码上下文信息，并拼接成一个张量
# 3. 计算相似度
# 4. 获取最相关的上下文信息

# Q:为什么用两个编码器？
# A: 
# 1. 问题和上下文的语言特点和语义结构不同；用两个编码器可以分别训练以更好捕捉各自特点
# 2. 可以预先计算所有上下文的向量，提高检索效率；
# 3. 可以分别优化问题和上下文的编码器，以提高检索质量；

from transformers import (
    DPRQuestionEncoder, # 问题编码器
    DPRContextEncoder, # 上下文编码器
    DPRQuestionEncoderTokenizer, # 问题编码器分词器
    DPRContextEncoderTokenizer, # 上下文编码器分词器
)
import torch
import numpy as np 
from sklearn.metrics.pairwise import cosine_similarity # 余弦相似度计算

# 加载预训练的DPR模型和分词器
question_encoder = DPRQuestionEncoder.from_pretrained(
    "facebook/dpr-question_encoder-single-nq-base"
)
context_encoder = DPRContextEncoder.from_pretrained(
    "facebook/dpr-ctx_encoder-single-nq-base"
)
question_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained(
    "facebook/dpr-question_encoder-single-nq-base"
)
context_tokenizer = DPRContextEncoderTokenizer.from_pretrained(
    "facebook/dpr-ctx_encoder-single-nq-base"
)

# 编码问题
query = "capital of africa?"
question_inputs = question_tokenizer(query, return_tensors="pt")
question_embedding = question_encoder(**question_inputs).pooler_output

# 将上下文信息进行编码
passages = [
    "Paris is the capital of France.",
    "Berlin is the capital of Germany.",
    "Madrid is the capital of Spain.",
    "Rome is the capital of Italy.",
    "Maputo is the capital of Mozambique.",
    "To be or not to be, that is the question.",
    "The quick brown fox jumps over the lazy dog.",
    "Grace Hopper was an American computer scientist and United States Navy rear admiral. who was a pioneer of computer programming, and one of the first programmers of the Harvard Mark I computer. inventor of the first compiler for a computer programming language.",
]

context_embeddings = []
for passage in passages:
    context_inputs = context_tokenizer(passage, return_tensors="pt")
    context_embedding = context_encoder(**context_inputs).pooler_output
    context_embeddings.append(context_embedding)

# 将上下文信息拼接成一个张量
context_embeddings = torch.cat(context_embeddings, dim=0)

# 计算相似度
similarities = cosine_similarity(
    question_embedding.detach().numpy(), context_embeddings.detach().numpy()
)
print("Similarities:", similarities)

# 获取最相关的上下文信息
most_relevant_idx = np.argmax(similarities)
print("Most relevant passage:", passages[most_relevant_idx])
