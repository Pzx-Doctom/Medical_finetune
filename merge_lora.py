"""
LoRA 合并脚本：将 LoRA adapter 合并到基座模型，生成完整权重。

Ollama 不支持直接加载 PEFT/LoRA adapter，因此需要先将 adapter 与基座模型合并，
然后导出为完整的 HuggingFace 格式模型，再转换为 GGUF 格式供 Ollama 使用。

用法：
    python merge_lora.py

输出：
    ./merged_model/  —— 合并后的完整模型（可直接被 llama.cpp convert 脚本处理）
"""

import gc
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

import config

OUTPUT_DIR = "./merged_model"


def main():
    print("=" * 60)
    print("LoRA Adapter 合并")
    print("=" * 60)

    # 1. 检查 adapter 是否存在
    if not os.path.exists(config.ADAPTER_PATH):
        print(f"\n✗ 未找到 LoRA adapter: {config.ADAPTER_PATH}")
        print("  请先运行: python train_lora.py")
        sys.exit(1)

    # 2. 加载基座模型（CPU 模式，避免显存不足）
    print(f"\n[1/4] 加载基座模型: {config.BASE_MODEL}")
    print("      使用 CPU + float32（合并不需要 GPU）")
    base_model = AutoModelForCausalLM.from_pretrained(
        config.BASE_MODEL,
        torch_dtype=torch.float32,
        trust_remote_code=True,
        device_map="cpu",
        low_cpu_mem_usage=True,
    )

    # 3. 加载 LoRA adapter 并合并
    print(f"\n[2/4] 加载 LoRA adapter: {config.ADAPTER_PATH}")
    model = PeftModel.from_pretrained(base_model, config.ADAPTER_PATH)
    
    print("\n[3/4] 合并 adapter 到基座模型（merge_and_unload）...")
    model = model.merge_and_unload()
    # merge_and_unload 会将 LoRA 权重合并到原始权重中，并移除 PEFT 包装

    # 4. 保存合并后的模型
    print(f"\n[4/4] 保存合并后模型到: {OUTPUT_DIR}")
    model.save_pretrained(OUTPUT_DIR, safe_serialization=True)

    # 保存 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.save_pretrained(OUTPUT_DIR)

    # 清理内存
    del model, base_model, tokenizer
    gc.collect()

    # 验证输出
    required_files = ["model.safetensors", "config.json", "tokenizer.json"]
    missing = [f for f in required_files if not os.path.exists(os.path.join(OUTPUT_DIR, f))]
    
    print("\n" + "=" * 60)
    if missing:
        print(f"⚠ 缺少以下文件: {missing}")
    else:
        print("✓ LoRA 合并完成！")
        print(f"  合并后模型路径: {os.path.abspath(OUTPUT_DIR)}")

    print("\n下一步（Ollama 部署）：")
    print("  1. 安装 llama.cpp: pip install llama-cpp-python")
    print("  2. 转换为 GGUF: python convert_to_gguf.py")
    print("  3. 创建 Modelfile 并导入: ollama create medical-assistant -f Modelfile")


if __name__ == "__main__":
    main()
