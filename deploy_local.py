"""
本地部署脚本：加载微调后模型，提供交互式问答。
支持两种模式：
    python deploy_local.py              # 交互模式（一问一答）
    python deploy_local.py "感冒怎么办"   # 单次问答

用法：
    pip install gradio  # 可选，Web UI 模式需要
    python deploy_local.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

import config


def build_prompt(question: str) -> str:
    return (
        f"<|im_start|>system\n{config.SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{question}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def load_model(adapter_path: str = config.ADAPTER_PATH):
    """加载基座 + LoRA adapter 的合并模型（FP16 省显存）。"""
    print(f"加载模型: {config.BASE_MODEL} + {adapter_path}")

    tokenizer = AutoTokenizer.from_pretrained(
        config.BASE_MODEL, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        config.BASE_MODEL,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        device_map="auto",
    )

    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()

    print(f"✓ 模型加载完成（基座 1.24GB + adapter 3MB）")
    return model, tokenizer


def answer(model, tokenizer, question: str) -> str:
    """生成回答。"""
    prompt = build_prompt(question)
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

    generated = outputs[0][inputs["input_ids"].shape[1] :]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def interactive_mode(model, tokenizer):
    """交互模式：持续问答直到输入 exit。"""
    print("\n" + "=" * 60)
    print("医疗助手已就绪（输入 exit 退出）")
    print("=" * 60)

    while True:
        try:
            question = input("\n你: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            break

        if not question:
            continue
        if question.lower() in ("exit", "quit", "q"):
            print("再见！")
            break

        response = answer(model, tokenizer, question)
        print(f"\n助手: {response}")


def gradio_mode(model, tokenizer):
    """Gradio Web UI 模式。"""
    try:
        import gradio as gr
    except ImportError:
        print("\n✗ 请先安装 gradio: pip install gradio")
        return

    def chat_fn(question, history):
        return answer(model, tokenizer, question)

    iface = gr.ChatInterface(
        fn=chat_fn,
        title="医疗助手 (Qwen1.5-0.5B + LoRA)",
        description="基于 2000 条医疗问答数据微调的医疗助手。",
        examples=config.MEDICAL_QUESTIONS,
    )
    iface.launch(share=False)


def main():
    if not os.path.exists(config.ADAPTER_PATH):
        print(f"✗ 未找到 LoRA adapter: {config.ADAPTER_PATH}")
        print("  请先运行: python train_lora.py")
        sys.exit(1)

    model, tokenizer = load_model()

    # 命令行参数：单次问答
    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
        print(f"\n问题: {question}")
        print(f"\n回答: {answer(model, tokenizer, question)}")
        return

    # 尝试 Gradio Web UI
    try:
        import gradio  # noqa: F401
        print("\n启动 Web UI...")
        gradio_mode(model, tokenizer)
    except ImportError:
        interactive_mode(model, tokenizer)


if __name__ == "__main__":
    main()
