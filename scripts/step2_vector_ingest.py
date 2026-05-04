# -*- coding: utf-8 -*-
import os
import re
import hashlib
import json
import time
import gc
from pathlib import Path
from dotenv import load_dotenv
import torch

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["MODELSCOPE_CACHE"] = "./models"

from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

# ================= 1. 配置 =================
load_dotenv()
BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "output"
DB_DIR = BASE_DIR / "chroma_db"
RECORD_FILE = BASE_DIR / "scripts" / "ingested_hashes.json"
LOG_FILE = BASE_DIR / "logs" / "vector_ingest.log"

for p in [DB_DIR, RECORD_FILE.parent, LOG_FILE.parent]:
    p.mkdir(parents=True, exist_ok=True)

def log_message(message: str, level: str = "INFO"):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] [{level}] {message}")
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"[{timestamp}] [{level}] {message}\n")

# ================= 2. 核心功能 =================
def table_to_semantic_text(md_content: str) -> str:
    """增强表格解析，支持更复杂的结构"""
    table_pattern = re.compile(r'^\|.*\|$\n^\|[-:| ]+\|$\n(?:^\|.*\|$\n*)+', re.MULTILINE)
    def process(match):
        table_str = match.group(0)
        lines = [l.strip('|') for l in table_str.strip().split('\n')]
        if len(lines) < 3: return table_str
        headers = [h.strip() for h in lines[0].split('|')]
        rows = [r.split('|') for r in lines[2:]]
        res = []
        for r in rows:
            cells = [c.strip() for c in r]
            if len(cells) >= len(headers):
                items = [f"{headers[i]}为{cells[i]}" for i in range(len(headers)) if cells[i]]
                res.append(f"【记录】" + "；".join(items) + "。")
        return table_str + "\n\n**[语义增强]**\n" + "\n".join(res)
    return table_pattern.sub(process, md_content)

class VectorIngestor:
    def __init__(self):
        log_message("🧮 初始化 BGE-M3 向量引擎...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-m3",
            model_kwargs={'device': device},
            encode_kwargs={'normalize_embeddings': True, 'batch_size': 32}
        )
        self.vectorstore = Chroma(persist_directory=str(DB_DIR), embedding_function=self.embeddings)
        self.records = self._load_records()

    def _load_records(self):
        if RECORD_FILE.exists():
            with open(RECORD_FILE, 'r', encoding='utf-8') as f: return json.load(f)
        return {}

    def cleanup_deleted_files(self, current_keys: set):
        """如果 output 里的 md 没了，自动清理数据库向量"""
        removed_keys = [k for k in self.records.keys() if k not in current_keys]
        for rk in removed_keys:
            log_message(f"🔥 文件已删除，正在清理数据库向量: {rk}", "WARNING")
            try:
                self.vectorstore.delete(where={"source": rk})
                del self.records[rk]
            except Exception as e: log_message(f"❌ 清理 {rk} 失败: {e}", "ERROR")

    def run(self):
        log_message("🚀 阶段二启动：向量库同步...")
        md_files = list(OUTPUT_DIR.rglob("*.md"))
        current_keys = {str(md.relative_to(OUTPUT_DIR)) for md in md_files}
        
        md_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=[("#","H1"),("##","H2"),("###","H3")])
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)

        for md_path in md_files:
            file_key = str(md_path.relative_to(OUTPUT_DIR))
            with open(md_path, 'r', encoding='utf-8') as f: content = f.read()
            curr_hash = hashlib.md5(content.encode('utf-8')).hexdigest()

            if self.records.get(file_key) != curr_hash:
                log_message(f"♻️  更新内容: {file_key}")
                try: self.vectorstore.delete(where={"source": file_key})
                except: pass

                processed_text = table_to_semantic_text(content)
                docs = md_splitter.split_text(processed_text)
                final_docs = []
                for d in docs:
                    d.metadata["source"] = file_key
                    final_docs.extend(text_splitter.split_documents([d]))
                
                if final_docs:
                    self.vectorstore.add_documents(final_docs)
                    self.records[file_key] = curr_hash
                    log_message(f"   ✅ 已同步 {len(final_docs)} 个切片")

        # 执行清理动作
        self.cleanup_deleted_files(current_keys)
        
        with open(RECORD_FILE, 'w', encoding='utf-8') as f:
            json.dump(self.records, f, indent=4, ensure_ascii=False)
        
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        log_message("🎉 阶段二执行完毕")

if __name__ == "__main__":
    VectorIngestor().run()