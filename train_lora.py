"""
LoRA 训练脚本：FP16 加载 + LoRA 注入 + 显存优化 + 参数量打印 + loss 回调

流程：
1. 加载处理好的 ChatML 格式数据
2. FP16 加载 Qwen1.5-0.5B 基座模型 + gradient checkpointing
3. 注入 LoRA 适配器（r=8, target_modules=q_proj/v_proj）
4. 打印可训练参数量（验证"参数骤减"效果）
5. tokenization + label masking（prompt 部分设为 -100）
6. Trainer 训练 + loss 回调记录
7. 保存 LoRA adapter 权重

显存优化策略（适配 8GB 以下 GPU）：
- FP16 半精度加载
- gradient checkpointing
- batch_size=1 + 梯度累积=8
- max_seq_len=512

用法：
    python train_lora.py
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

import config


class LossLoggerCallback(TrainerCallback):
    """训练 loss 记录回调，将每个 logging step 的 loss 写入 JSON 文件。

    输出格式: [{"step": 10, "loss": 2.34}, {"step": 20, "loss": 1.87}, ...]
    实时写入，防止训练中断时丢失数据。
    """

    def __init__(self, log_path: str):
        self.log_path = log_path
        self.loss_records = []
        os.makedirs(os.path.dirname(log_path), exist_ok=True)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        loss = logs.get("loss")
        if loss is not None:
            record = {"step": state.global_step, "loss": float(loss)}
            self.loss_records.append(record)
            with open(self.log_path, "w", encoding="utf-8") as f:
                json.dump(self.loss_records, f, ensure_ascii=False, indent=2)


def tokenize_with_mask(examples, tokenizer, max_seq_len):
    """tokenization + label masking。

    对 prompt 部分（system + user）的 token 进行 label masking（设为 -100），
    仅在 assistant 回答部分计算 loss，确保模型学习"如何回答"而非"记忆问题"。
    """
    all_input_ids = []
    all_labels = []

    assistant_marker = "<|im_start|>assistant\n"

    for text in examples["text"]:
        # 分割 prompt 和 assistant 回答
        idx = text.find(assistant_marker)
        if idx == -1:
            continue

        prompt_text = text[: idx + len(assistant_marker)]
        response_text = text[idx + len(assistant_marker) :]

        # tokenize（不加特殊 token，ChatML 标记已包含在文本中）
        prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        response_ids = tokenizer(response_text, add_special_tokens=False)["input_ids"]

        # 拼接
        input_ids = prompt_ids + response_ids
        # label masking: prompt 部分设为 -100，response 部分保留
        labels = [-100] * len(prompt_ids) + response_ids[:]

        # 截断到最大长度
        if len(input_ids) > max_seq_len:
            input_ids = input_ids[:max_seq_len]
            labels = labels[:max_seq_len]

        all_input_ids.append(input_ids)
        all_labels.append(labels)

    return {
        "input_ids": all_input_ids,
        "labels": all_labels,
    }


def main():
    print("=" * 60)
    print("LoRA 微调训练")
    print("=" * 60)

    # 1. 检查数据文件
    if not os.path.exists(config.PROCESSED_DATA_PATH):
        print(f"\n✗ 错误: 数据文件不存在: {config.PROCESSED_DATA_PATH}")
        print(f"  请先运行: python data/prepare_data.py")
        sys.exit(1)

    # 2. 加载 tokenizer
    print(f"\n[1/6] 加载 Tokenizer: {config.BASE_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(
        config.BASE_MODEL,
        trust_remote_code=True,
        padding_side="right",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 3. 加载基座模型（FP16）
    print(f"\n[2/6] 加载基座模型: {config.BASE_MODEL}")
    print(f"      dtype: float16")
    model = AutoModelForCausalLM.from_pretrained(
        config.BASE_MODEL,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        device_map="auto",
    )
    model.config.use_cache = False
    # gradient checkpointing + LoRA 必须启用 input_require_grads
    model.enable_input_require_grads()

    # 4. 注入 LoRA 适配器
    print(f"\n[3/6] 注入 LoRA 适配器")
    print(
        f"      r={config.LORA_R}, alpha={config.LORA_ALPHA}, "
        f"target_modules={config.LORA_TARGET_MODULES}, dropout={config.LORA_DROPOUT}"
    )
    lora_config = LoraConfig(
        r=config.LORA_R,
        lora_alpha=config.LORA_ALPHA,
        lora_dropout=config.LORA_DROPOUT,
        target_modules=config.LORA_TARGET_MODULES,
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    # 5. 打印可训练参数量（验证 LoRA "参数骤减"效果）
    print(f"\n[4/6] 可训练参数量验证")
    model.print_trainable_parameters()

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"\n  可训练参数: {trainable:,}")
    print(f"  总参数:     {total:,}")
    print(f"  可训练占比: {trainable / total * 100:.4f}%")
    print(f"  压缩倍数:   {total / trainable:.0f}x")

    # 6. 加载并处理数据
    print(f"\n[5/6] 加载训练数据: {config.PROCESSED_DATA_PATH}")
    raw_dataset = load_dataset(
        "json", data_files=config.PROCESSED_DATA_PATH, split="train"
    )
    print(f"      原始数据量: {len(raw_dataset)} 条")

    tokenized_dataset = raw_dataset.map(
        lambda examples: tokenize_with_mask(examples, tokenizer, config.MAX_SEQ_LEN),
        batched=True,
        remove_columns=raw_dataset.column_names,
        desc="Tokenizing",
    )
    print(f"      处理后数据量: {len(tokenized_dataset)} 条")

    # 7. 训练
    print(f"\n[6/6] 开始训练")
    print(
        f"      epochs={config.NUM_EPOCHS}, lr={config.LEARNING_RATE}, "
        f"batch={config.BATCH_SIZE}x{config.GRAD_ACCUM_STEPS} (等效 {config.BATCH_SIZE * config.GRAD_ACCUM_STEPS})"
    )

    training_args = TrainingArguments(
        output_dir=config.ADAPTER_PATH,
        num_train_epochs=config.NUM_EPOCHS,
        per_device_train_batch_size=config.BATCH_SIZE,
        gradient_accumulation_steps=config.GRAD_ACCUM_STEPS,
        learning_rate=config.LEARNING_RATE,
        warmup_ratio=config.WARMUP_RATIO,
        logging_steps=config.LOGGING_STEPS,
        save_strategy="epoch",
        save_total_limit=3,
        seed=config.SEED,
        fp16=True,
        gradient_checkpointing=True,
        report_to="none",
        optim="adamw_torch",
        lr_scheduler_type="cosine",
        ddp_find_unused_parameters=False,
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        return_tensors="pt",
        label_pad_token_id=-100,
    )

    loss_logger = LossLoggerCallback(config.LOSS_LOG_PATH)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
        callbacks=[loss_logger],
    )

    try:
        trainer.train()
    except torch.cuda.OutOfMemoryError as e:
        print(f"\n✗ CUDA 显存不足 (OOM)！")
        print(f"  错误信息: {e}")
        print(f"\n  降级建议:")
        print(f"  1. 减小 max_seq_len（当前 {config.MAX_SEQ_LEN}，可尝试 256 或 128）")
        print(f"  2. 使用 Google Colab T4 运行 colab/medical_lora_finetune.ipynb")
        print(f"  3. 尝试 QLoRA 4-bit 量化（需安装 bitsandbytes）")
        sys.exit(1)

    # 8. 保存 LoRA adapter
    print(f"\n保存 LoRA adapter 到: {config.ADAPTER_PATH}")
    model.save_pretrained(config.ADAPTER_PATH)
    tokenizer.save_pretrained(config.ADAPTER_PATH)

    print(f"\n✓ 训练完成！")
    print(f"  Adapter 权重: {config.ADAPTER_PATH}")
    print(f"  Loss 日志:    {config.LOSS_LOG_PATH}")
    print(f"\n下一步:")
    print(f"  运行推理对比: python inference_compare.py")
    print(f"  绘制 loss 曲线: python plot_loss.py")


if __name__ == "__main__":
    main()
