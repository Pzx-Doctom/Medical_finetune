"""
GGUF 格式转换脚本：将合并后的 HuggingFace 模型转换为 GGUF 格式（Ollama 使用的格式）。

需要先安装 llama.cpp：
    git clone https://github.com/ggerganov/llama.cpp.git
    cd llama.cpp
    pip install -r requirements/requirements-convert.txt

用法（需先运行 merge_lora.py）：
    python convert_to_gguf.py              # 默认 F16 精度
    python convert_to_gguf.py --q8_0        # Q8_0 量化（体积减半）
    python convert_to_gguf.py --q4_k_m      # Q4_K_M 量化（约 4GB → 400MB）

输出：
    ./merged_model/medical-qwen1.5-0.5b-F16.gguf （或对应量化版本）
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

# 合并后模型路径
MERGED_MODEL_DIR = "./merged_model"
GGUF_OUTPUT_NAME = "medical-qwen1.5-0.5b"
LLAMA_CPP_DIR = "./llama.cpp"


def check_merged_model():
    """检查合并后的模型是否存在。"""
    if not os.path.exists(MERGED_MODEL_DIR):
        print(f"✗ 未找到合并后的模型: {MERGED_MODEL_DIR}")
        print("  请先运行: python merge_lora.py")
        return False
    
    required = ["config.json", "tokenizer.json"]
    missing = [f for f in required if not os.path.exists(os.path.join(MERGED_MODEL_DIR, f))]
    if missing:
        print(f"✗ 模型文件不完整，缺少: {missing}")
        return False
    
    return True


def setup_llama_cpp():
    """确保 llama.cpp 仓库可用，克隆或确认存在。"""
    if os.path.isdir(LLAMA_CPP_DIR):
        convert_script = os.path.join(LLAMA_CPP_DIR, "convert_hf_to_gguf.py")
        if os.path.isfile(convert_script):
            print(f"✓ 找到 llama.cpp: {LLAMA_CPP_DIR}")
            return convert_script

    print(f"\n正在克隆 llama.cpp 到 {LLAMA_CPP_DIR} ...")
    try:
        subprocess.run(
            ["git", "clone", "--depth", "1", "https://github.com/ggerganov/llama.cpp.git", LLAMA_CPP_DIR],
            check=True,
        )
        # 安装转换所需依赖
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-r",
             os.path.join(LLAMA_CPP_DIR, "requirements", "requirements-convert_hf_to_gguf.txt")],
            check=True,
        )
        print("✓ llama.cpp 克隆并安装依赖完成")
        return os.path.join(LLAMA_CPP_DIR, "convert_hf_to_gguf.py")
    except subprocess.CalledProcessError as e:
        print(f"\n✗ 无法 clone llama.cpp: {e}")
        print("\n请手动操作：")
        print(f"  git clone https://github.com/ggerganov/llama.cpp.git {LLAMA_CPP_DIR}")
        print(f"  pip install -r {LLAMA_CPP_DIR}/requirements/requirements-convert.txt")
        return None


def convert_to_gguf(convert_script: str, outtype: str):
    """调用 llama.cpp 的 convert_hf_to_gguf.py 进行转换。"""
    output_file = os.path.join(MERGED_MODEL_DIR, f"{GGUF_OUTPUT_NAME}-{outtype.upper().replace('_', '')}.gguf")
    
    cmd = [
        sys.executable,
        convert_script,
        MERGED_MODEL_DIR,           # 输入：HF 模型目录
        "--outfile", output_file,   # 输出：GGUF 文件
        "--outtype", outtype,       # 精度类型
        "--model-name", "medical-qwen1.5-0.5b",
    ]
    
    print(f"\n转换命令: {' '.join(cmd)}")
    print(f"转换中（根据模型大小可能需要几分钟）...\n")
    
    result = subprocess.run(cmd, capture_output=False)
    
    if result.returncode != 0:
        print(f"\n✗ 转换失败，返回码: {result.returncode}")
        return None
    
    if os.path.isfile(output_file):
        size_mb = os.path.getsize(output_file) / (1024 * 1024)
        print(f"\n✓ GGUF 转换成功！")
        print(f"  文件: {output_file}")
        print(f"  大小: {size_mb:.1f} MB")
        return output_file
    else:
        print(f"\n✗ 未生成预期的输出文件: {output_file}")
        return None


def main():
    parser = argparse.ArgumentParser(
        description="将合并后的 HF 模型转换为 GGUF 格式供 Ollama 使用"
    )
    parser.add_argument(
        "--outtype",
        default="f16",
        choices=["f16", "f32", "q8_0", "q4_0", "q4_1", "q5_0", "q5_1", "q4_k_m", "q5_k_m", "q8_k"],
        help="输出精度类型。f16=原始半精度(~1GB), q8_0=8bit(~550MB), q4_k_m=4bit(~350MB)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("HuggingFace → GGUF 格式转换")
    print(f"精度: {args.outtype}")
    print("=" * 60)

    if not check_merged_model():
        sys.exit(1)

    convert_script = setup_llama_cpp()
    if not convert_script:
        sys.exit(1)

    gguf_path = convert_to_gguf(convert_script, args.outtype)
    if not gguf_path:
        sys.exit(1)

    # 给出下一步指令
    print("\n" + "=" * 60)
    print("下一步（Ollama 导入）：")
    print("=" * 60)
    print(f'\n  方法 1 - 使用 Modelfile（推荐）：')
    print(f'    ollama create medical-assistant -f Modelfile')
    print(f'\n  方法 2 - 直接用 GGUF：')
    print(f'    ollama create medical-assistant -f Modelfile')
    print(f'\n  注意：Modelfile 中的 GGUF 路径需要与转换输出一致。')
    print(f'  当前转换输出: {os.path.basename(gguf_path)}')


if __name__ == "__main__":
    main()
