# -*- coding: utf-8 -*-
"""
面向小学教育的个性化学习问答系统 - RAG 增强版
修复：重排逻辑、Sigmoid归一化、历史记录修剪、向量得分映射
"""

import os
import gc
import math
import logging
from pathlib import Path
from typing import List, Any
from tenacity import retry, stop_after_attempt, wait_exponential

# ================= 1. 环境配置 =================
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from dotenv import load_dotenv
load_dotenv()

import torch
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from sentence_transformers import CrossEncoder

# ================= 1. 日志配置 =================
def setup_logging():
    log_dir = Path(__file__).resolve().parent.parent / "logs"
    log_dir.mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / "rag_service.log", encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger("CampusRAG")

logger = setup_logging()

# ================= 2. 资源管理 =================
class ResourceManager:
    @staticmethod
    def cleanup():
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    @staticmethod
    def get_device():
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"🎯 运行设备: {device}")
        return device

# ================= 3. RAG 核心引擎 =================
class CampusRAGEngine:
    def __init__(self):
        logger.info("⚙️ 初始化 RAG 引擎核心模块...")
        self.device = ResourceManager.get_device()
        self.base_dir = Path(__file__).resolve().parent.parent
        self.db_dir = self.base_dir / "chroma_db"

        # 参数配置（优先从环境变量读取）
        self.embedding_model_name = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
        self.reranker_model_name = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-base")
        self.min_relevance_score = float(os.getenv("MIN_RELEVANCE_SCORE", "0.45")) # Sigmoid 后建议 0.4-0.5
        self.retrieval_top_k = int(os.getenv("RETRIEVAL_TOP_K", "10"))
        self.rerank_top_n = int(os.getenv("RERANK_TOP_N", "3"))

        self._embeddings = None
        self._vectorstore = None
        self._reranker = None
        self._llm = None

    @property
    def embeddings(self):
        if self._embeddings is None:
            self._embeddings = HuggingFaceEmbeddings(
                model_name=self.embedding_model_name,
                model_kwargs={'device': self.device, 'trust_remote_code': True},
                encode_kwargs={'normalize_embeddings': True} # 必须开启以保证余弦计算准确
            )
        return self._embeddings

    @property
    def vectorstore(self):
        if self._vectorstore is None:
            # 指定使用 cosine 相似度算法
            self._vectorstore = Chroma(
                persist_directory=str(self.db_dir),
                embedding_function=self.embeddings,
                collection_metadata={"hnsw:space": "cosine"}
            )
        return self._vectorstore

    @property
    def reranker(self):
        if self._reranker is None:
            try:
                self._reranker = CrossEncoder(
                    self.reranker_model_name,
                    device=self.device
                )       
                logger.info("✅ 重排模型 CrossEncoder 加载完毕")
            except Exception as e:
                logger.error(f"⚠️ Reranker 加载失败: {e}")
                self._reranker = None
        return self._reranker

    @property
    def llm(self):
        if self._llm is None:
            self._llm = ChatOpenAI(
                api_key=os.getenv("KEY"),
                base_url=os.getenv("URL"),
                model=os.getenv("MODEL_NAME", "qwen-plus"),
                temperature=0.1
            )
        return self._llm

    # ================= 逻辑流 =================

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def query_rewrite(self, question: str, history: str) -> str:
        """基于历史对话重写问题，使其独立可检索"""
        if not history.strip():
            return question

        prompt = f"""你是一个学习助手。请将用户的最新问题改写为一个完整、独立的问题。

【对话历史】
{history}

【学生提出的问题】
{question}

【改写步骤】
1. 识别指代词：找出问题中的"它"、"这个"、"那个"等指代词
2. 定位指代对象：从历史中找到这些指代词的具体内容
3. 替换指代词：用具体内容替换指代词
4. 补充上下文：确保问题包含所有必要信息
5. 精简表达：去除冗余，保持简洁

【输出格式】
请严格按照以下格式输出：

【原始问题】
{question}

【改写后的问题】
（只输出改写后的问题，不要包含任何解释）

改写后的问题："""
        try:
            res = self.llm.invoke(prompt)
            rewritten = res.content.strip().replace('"', '')
            logger.info(f"🔍 查询改写: {question} -> {rewritten}")
            return rewritten
        except Exception as e:
            logger.warning(f"改写失败，使用原问题: {e}")
            return question

    def retrieve_and_rerank(self, query: str) -> List[Any]:
        """执行召回 + Cross-Encoder 重排"""
        try:
            # 1. 向量初步检索
            docs_scores = self.vectorstore.similarity_search_with_score(
                query, k=self.retrieval_top_k
            )
            if not docs_scores:
                return []

            docs = []
            for doc, dist in docs_scores:
                # Chroma 在使用 cosine 时返回的是 1-cosine，所以需要转换回来
                score = max(0.0, 1.0 - dist) 
                doc.metadata["base_score"] = score
                docs.append(doc)

            # 2. 精准重排
            if self.reranker:
                pairs = [(query, d.page_content) for d in docs]

                logits = self.reranker.predict(pairs, batch_size=32)

                doc_score_pairs = list(zip(docs, logits))
                doc_score_pairs.sort(key=lambda x: x[1], reverse=True)

                final_docs = []
                for doc, logit in doc_score_pairs[:self.rerank_top_n]:
                    # 使用 Sigmoid 归一化到 [0, 1]
                    norm_score = 1.0 / (1.0 + math.exp(-logit))
                    doc.metadata["relevance_score"] = norm_score
                    final_docs.append(doc)
            else:
                # 若无重排模型，则使用初筛分数降级
                docs.sort(key=lambda d: d.metadata["base_score"], reverse=True)
                final_docs = docs[:self.rerank_top_n]
                for d in final_docs:
                    d.metadata["relevance_score"] = d.metadata["base_score"]

            # 3. 相似度阈值过滤
            passed_docs = [
                d for d in final_docs 
                if d.metadata.get("relevance_score", 0) >= self.min_relevance_score
            ]
            logger.info(f"🎯 最终入选文档数: {len(passed_docs)}")
            return passed_docs

        except Exception as e:
            logger.error(f"检索重排链路故障: {e}", exc_info=True)
            return []

    def generate(self, question: str, context: str, history: str):
        """最终回答生成"""
        prompt_tpl = ChatPromptTemplate.from_template("""
你是一名经验丰富的小学教师，请根据【参考知识】严谨地回答【学生问题】。

【参考知识】
{context}

【对话历史】
{history}

【学生问题】
{question}

【任务判断】
请先判断学生提出的问题是否为新题目：
- 新题目特征：是否为小学题目（语文、数学、英语等），包含具体题目内容。属于计算题、应用题、选择题等明确的题目信息
- 非新题目特征：询问概念、讨论上一题的细节、询问解题技巧、表达疑惑等

【回答策略】
根据判断结果选择回答方式：

【情况一：新题目】
如果学生提出的是新题目，请严格按照以下四个部分回答：

【思路分析】
（简要分析题目要求和解题思路）

【解题步骤】
（分步骤详细说明解题过程，每一步都要清晰明了）

【参考答案】
（给出准确的答案）

【鼓励交流】
（温和地鼓励学生，如："你理解了吗？如果有不清楚的地方，随时问我哦！"）

【情况二：非新题目】
如果学生提出的是概念询问、疑惑讨论或解题技巧等非新题目内容：
- 针对学生的具体问题直接回答
- 结合对话历史保持连贯性
- 用通俗易懂的语言解释
- 保持亲切专业的教师形象
- 鼓励学生继续交流

【语言要求】
1. 使用小学生能理解的语言，避免复杂学术用语
2. 保持专业和严谨，亲切但不滑稽，鼓励但不浮夸
3. 如果【参考知识】中没有相关信息，诚实告知并尝试用基础知识引导

【输出格式】
请根据判断结果，直接输出相应的回答内容，不需要标注"情况一"或"情况二"。

【教师回答】
""")
        chain = prompt_tpl | self.llm
        return chain.invoke({
            "context": context if context else "未找到直接相关的参考资料。",
            "history": history if history else "暂无对话历史。",
            "question": question
        })

# ================= 4. CLI 交互界面 =================
def run_cli():
    engine = CampusRAGEngine()
    history_list = []
    MAX_HISTORY_TURNS = 5  # 仅保留最近 5 轮对话，防止上下文爆炸

    print("\n" + "="*30)
    print("🏫 欢迎使用智慧小学 RAG 问答系统")
    print("输入 'exit' 或 'quit' 退出程序")
    print("="*30)

    while True:
        try:
            query = input("\n👤 学生: ").strip()
            if not query: continue
            if query.lower() in ["exit", "quit"]: break

            # 提取最近的对话历史
            history_str = "\n".join(history_list[-MAX_HISTORY_TURNS*2:])

            # 1. 查询改写
            rewritten_q = engine.query_rewrite(query, history_str)

            # 2. 检索与重排
            relevant_docs = engine.retrieve_and_rerank(rewritten_q)
            context_text = "\n\n".join([
                f"[资料{i+1}]: {d.page_content}" 
                for i, d in enumerate(relevant_docs)
            ])

            # 3. 生成回答
            response = engine.generate(query, context_text, history_str)
            ans_content = response.content

            print(f"\n🤖 老师:\n{ans_content}")

            # 4. 更新历史记录
            history_list.append(f"学生: {query}")
            history_list.append(f"老师: {ans_content}")

            # 5. 显存清理
            ResourceManager.cleanup()

        except KeyboardInterrupt:
            break
        except Exception as e:
            logger.error(f"系统运行错误: {e}")
            print("❌ 抱歉，服务器繁忙，请稍后再试。")

if __name__ == "__main__":
    run_cli()