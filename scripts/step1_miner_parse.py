# -*- coding: utf-8 -*-
import os
import subprocess
import shutil
import hashlib
import json
import time
from pathlib import Path
from typing import List, Dict

# ================= 1. 核心配置 =================
os.environ["MINERU_MODEL_SOURCE"] = "modelscope"

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "output"
HASH_RECORD = BASE_DIR / "scripts" / "pdf_hashes.json"
LOG_FILE = BASE_DIR / "logs" / "miner_parse.log"

for p in [DATA_DIR, OUTPUT_DIR, HASH_RECORD.parent, LOG_FILE.parent]:
    p.mkdir(parents=True, exist_ok=True)

# ================= 2. 日志工具 =================
def log_message(message: str, level: str = "INFO"):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    log_line = f"[{timestamp}] [{level}] {message}"
    print(log_line)
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(log_line + '\n')

# ================= 3. 工具逻辑 =================
def get_file_hash(file_path: Path) -> str:
    hasher = hashlib.md5()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hasher.update(chunk)
    return hasher.hexdigest()

def cleanup_orphans(current_pdf_stems: set):
    """如果 PDF 没了，自动删掉对应的 output 文件夹"""
    log_message("🧹 正在检查孤儿 output 文件夹...")
    orphan_count = 0
    for folder in OUTPUT_DIR.iterdir():
        if folder.is_dir() and folder.name not in current_pdf_stems:
            log_message(f"🗑️ 检测到 PDF 已移除，清理孤儿文件夹: {folder.name}", "WARNING")
            shutil.rmtree(folder)
            orphan_count += 1
    if orphan_count > 0:
        log_message(f"✅ 已清理 {orphan_count} 个不再需要的解析目录")

# ================= 4. 主解析流程 =================
def run_mineru(pdf_path: Path) -> bool:
    output_path = OUTPUT_DIR / pdf_path.stem
    # 尝试 GPU，失败切 CPU
    for attempt in range(2):
        mode = "GPU" if attempt == 0 else "CPU"
        cmd = ["mineru", "-p", str(pdf_path), "-o", str(OUTPUT_DIR.resolve())]
        if attempt > 0: cmd.extend(["-b", "pipeline"])
        
        try:
            if output_path.exists(): shutil.rmtree(output_path)
            subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=600)
            log_message(f"✅ {pdf_path.name} {mode} 模式解析成功")
            return True
        except Exception as e:
            log_message(f"⚠️ {pdf_path.name} {mode} 模式失败: {e}", "ERROR")
    return False

def parse_all_pdfs():
    log_message("🚀 阶段一启动：MinerU 智能同步解析器...")
    
    if HASH_RECORD.exists():
        with open(HASH_RECORD, 'r', encoding='utf-8') as f: old_hashes = json.load(f)
    else: old_hashes = {}

    all_pdfs = list(DATA_DIR.glob("*.pdf"))
    current_pdf_stems = {p.stem for p in all_pdfs}
    
    # 1. 自动清理已删除 PDF 的遗留文件
    cleanup_orphans(current_pdf_stems)

    # 2. 筛选需要处理的文件
    to_process = []
    new_hashes = {}
    for pdf in all_pdfs:
        h = get_file_hash(pdf)
        new_hashes[pdf.name] = h
        if old_hashes.get(pdf.name) != h or not (OUTPUT_DIR / pdf.stem).exists():
            to_process.append(pdf)

    log_message(f"📊 待处理: {len(to_process)} / 总计: {len(all_pdfs)}")

    for idx, pdf in enumerate(to_process, 1):
        log_message(f"[{idx}/{len(to_process)}] 正在处理: {pdf.name}")
        if not run_mineru(pdf):
            if pdf.name in new_hashes: del new_hashes[pdf.name]

    # 3. 只保留当前存在的 PDF 哈希记录
    final_hashes = {name: h for name, h in new_hashes.items() if (DATA_DIR / name).exists()}
    with open(HASH_RECORD, 'w', encoding='utf-8') as f:
        json.dump(final_hashes, f, indent=4, ensure_ascii=False)
    log_message("🎉 阶段一执行完毕")

if __name__ == "__main__":
    parse_all_pdfs()