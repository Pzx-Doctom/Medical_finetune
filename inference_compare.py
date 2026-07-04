"""
推理对比脚本：基座模型 vs 微调后模型

对 5 个典型医学问题，分别用基座模型和加载 LoRA adapter 的模型生成回答，
输出并排对比，展示微调带来的专业性提升。

用法：
    python inference_compare.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

import config


def build_chatml_prompt(question: str) -> str:
    """构造 ChatML 格式的推理 prompt（不含 assistant 回答）。"""
    return (
        f"<|im_start|>system\n{config.SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{question}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def generate_answer(model, tokenizer, question: str) -> str:
    """用给定模型生成回答。"""
    prompt = build_chatml_prompt(question)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=config.MAX_NEW_TOKENS,
            temperature=config.TEMPERATURE,
            top_p=config.TOP_P,
            do_sample=config.DO_SAMPLE,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # 只取新生成的 token（去掉 prompt 部分）
    generated = outputs[0][inputs["input_ids"].shape[1] :]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def main():
    print("=" * 60)
    print("推理对比：基座模型 vs 微调后模型")
    print("=" * 60)

    # 1. 加载 tokenizer
    print(f"\n[1/3] 加载 Tokenizer: {config.BASE_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(
        config.BASE_MODEL, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. 加载基座模型
    print(f"\n[2/3] 加载基座模型: {config.BASE_MODEL}")
    base_model = AutoModelForCausalLM.from_pretrained(
        config.BASE_MODEL,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        device_map="auto",
    )
    base_model.eval()

    # 3. 加载微调后模型（基座 + LoRA adapter）
    adapter_path = config.ADAPTER_PATH
    if not os.path.exists(adapter_path):
        print(f"\n✗ 错误: LoRA adapter 不存在: {adapter_path}")
        print(f"  请先运行: python train_lora.py")
        sys.exit(1)

    print(f"\n[3/3] 加载 LoRA adapter: {adapter_path}")
    tuned_model = PeftModel.from_pretrained(base_model, adapter_path)
    tuned_model.eval()

    # 4. 推理对比
    print(f"\n{'=' * 60}")
    print(f"开始对比 {len(config.MEDICAL_QUESTIONS)} 个医学问题")
    print(f"{'=' * 60}")

    results = []
    for i, question in enumerate(config.MEDICAL_QUESTIONS, 1):
        print(f"\n{'─' * 60}")
        print(f"问题 {i}/{len(config.MEDICAL_QUESTIONS)}: {question}")
        print(f"{'─' * 60}")

        print(f"\n[基座模型回答]")
        base_answer = generate_answer(base_model, tokenizer, question)
        print(base_answer)

        print(f"\n[微调后模型回答]")
        tuned_answer = generate_answer(tuned_model, tokenizer, question)
        print(tuned_answer)

        results.append(
            {
                "question": question,
                "base_answer": base_answer,
                "tuned_answer": tuned_answer,
            }
        )

    # 5. 保存对比结果
    print(f"\n{'=' * 60}")
    print(f"保存对比结果到: {config.COMPARISON_PATH}")
    os.makedirs(os.path.dirname(config.COMPARISON_PATH), exist_ok=True)

    with open(config.COMPARISON_PATH, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("推理对比结果：基座模型 vs LoRA 微调后模型\n")
        f.write(f"基座: {config.BASE_MODEL}\n")
        f.write(
            f"LoRA: r={config.LORA_R}, alpha={config.LORA_ALPHA}, "
            f"target={config.LORA_TARGET_MODULES}\n"
        )
        f.write("=" * 60 + "\n\n")

        for i, r in enumerate(results, 1):
            f.write(f"{'─' * 60}\n")
            f.write(f"问题 {i}: {r['question']}\n")
            f.write(f"{'─' * 60}\n\n")
            f.write(f"[基座模型回答]\n{r['base_answer']}\n\n")
            f.write(f"[微调后模型回答]\n{r['tuned_answer']}\n\n\n")

    print(f"\n✓ 完成！对比结果已保存到: {config.COMPARISON_PATH}")


if __name__ == "__main__":
    main()
