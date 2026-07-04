"""
数据准备脚本：下载、抽样、格式转换。

流程：
1. 从 HuggingFace 下载 shibing624/medical 的 finetune/train_zh_0.json（纯中文，约 194 万条）
2. 随机抽样 2000 条（seed=42 保证可复现）
3. 字段处理：input 非空时拼成 {instruction}\n{input}，为空时只用 instruction
4. 转换为 Qwen ChatML 指令格式
5. 保存为 JSONL 文件供训练脚本使用

用法：
    python data/prepare_data.py
"""

import json
import os
import sys

# 将项目根目录加入 sys.path，以便导入 config
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets import load_dataset

import config


def build_user_content(instruction: str, input_text: str) -> str:
    """构造 user 消息内容。

    input 非空时拼成 {instruction}\\n{input}（保留病人描述信息）
    input 为空时直接用 instruction
    """
    if input_text and input_text.strip():
        return f"{instruction}\n{input_text}"
    return instruction


def format_chatml(system_prompt: str, user_content: str, output: str) -> str:
    """将一条问答数据格式化为 Qwen ChatML 文本。"""
    return (
        f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        f"<|im_start|>user\n{user_content}<|im_end|>\n"
        f"<|im_start|>assistant\n{output}<|im_end|>"
    )


def main():
    print("=" * 60)
    print("数据准备：下载 + 抽样 + 格式转换")
    print("=" * 60)

    # 1. 下载数据集
    print(f"\n[1/4] 下载数据集: {config.DATASET_NAME}")
    print(f"      加载文件: {config.DATASET_SPLIT}")
    dataset = load_dataset(
        config.DATASET_NAME,
        data_files=config.DATASET_SPLIT,
        split="train",
    )
    print(f"      原始数据量: {len(dataset):,} 条")

    # 2. 随机抽样
    print(f"\n[2/4] 随机抽样: {config.DATA_SAMPLE_SIZE} 条 (seed={config.SEED})")
    sampled = dataset.shuffle(seed=config.SEED).select(
        range(min(config.DATA_SAMPLE_SIZE, len(dataset)))
    )

    # 3. 字段处理 + 格式转换
    print(f"\n[3/4] 转换为 ChatML 格式")
    processed = []
    input_nonempty_count = 0
    for item in sampled:
        instruction = item.get("instruction", "")
        input_text = item.get("input", "")
        output = item.get("output", "")

        if input_text and input_text.strip():
            input_nonempty_count += 1

        user_content = build_user_content(instruction, input_text)
        chatml_text = format_chatml(config.SYSTEM_PROMPT, user_content, output)

        processed.append(
            {
                "instruction": instruction,
                "input": input_text,
                "output": output,
                "user_content": user_content,
                "text": chatml_text,
            }
        )

    total = len(processed)
    print(
        f"      input 非空: {input_nonempty_count} 条 "
        f"({input_nonempty_count / total * 100:.1f}%)"
    )
    print(f"      input 为空: {total - input_nonempty_count} 条")

    # 4. 保存为 JSONL
    print(f"\n[4/4] 保存到: {config.PROCESSED_DATA_PATH}")
    os.makedirs(os.path.dirname(config.PROCESSED_DATA_PATH), exist_ok=True)
    with open(config.PROCESSED_DATA_PATH, "w", encoding="utf-8") as f:
        for item in processed:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"\n✓ 完成！共保存 {total} 条数据")

    # 打印样例
    print("\n" + "=" * 60)
    print("样例展示（第 1 条）:")
    print("=" * 60)
    sample = processed[0]
    print(f"\n[instruction] {sample['instruction']}")
    print(f"\n[input] {sample['input'] if sample['input'] else '(空)'}")
    truncated_output = sample["output"][:200]
    print(f"\n[output] {truncated_output}{'...' if len(sample['output']) > 200 else ''}")
    print(f"\n[ChatML 格式]")
    truncated_text = sample["text"][:500]
    print(truncated_text + ("..." if len(sample["text"]) > 500 else ""))


if __name__ == "__main__":
    main()
