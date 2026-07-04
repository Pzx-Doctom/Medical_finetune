"""
Loss 可视化脚本：从 loss_log.json 读取数据，绘制训练损失曲线图。

用法：
    python plot_loss.py
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib

matplotlib.use("Agg")  # 非交互式后端，无需 GUI
import matplotlib.pyplot as plt

import config

# 中文字体配置（尝试常见中文字体，找不到则用默认）
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def main():
    print("=" * 60)
    print("绘制 Loss 曲线")
    print("=" * 60)

    # 1. 检查日志文件
    if not os.path.exists(config.LOSS_LOG_PATH):
        print(f"\n✗ 错误: loss 日志不存在: {config.LOSS_LOG_PATH}")
        print(f"  请先运行: python train_lora.py")
        sys.exit(1)

    # 2. 读取 loss 数据
    print(f"\n[1/2] 读取 loss 日志: {config.LOSS_LOG_PATH}")
    with open(config.LOSS_LOG_PATH, "r", encoding="utf-8") as f:
        loss_data = json.load(f)

    if not loss_data:
        print(f"\n✗ 错误: loss 日志为空")
        sys.exit(1)

    steps = [item["step"] for item in loss_data]
    losses = [item["loss"] for item in loss_data]
    print(f"      共 {len(losses)} 条记录")
    print(f"      初始 loss: {losses[0]:.4f}")
    print(f"      最终 loss: {losses[-1]:.4f}")
    print(f"      最低 loss: {min(losses):.4f}")

    # 3. 绘制曲线
    print(f"\n[2/2] 绘制并保存: {config.LOSS_CURVE_PATH}")
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(steps, losses, color="#2563eb", linewidth=1.5, alpha=0.7, label="Loss")
    # 移动平均平滑线
    if len(losses) > 10:
        window = min(10, len(losses) // 5)
        ma_loss = []
        for i in range(len(losses)):
            start = max(0, i - window // 2)
            end = min(len(losses), i + window // 2 + 1)
            ma_loss.append(sum(losses[start:end]) / (end - start))
        ax.plot(
            steps,
            ma_loss,
            color="#dc2626",
            linewidth=2,
            label=f"移动平均 (window={window})",
        )

    ax.set_xlabel("训练步数 (Step)", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.set_title(
        f"LoRA 微调训练 Loss 曲线\n"
        f"基座: {config.BASE_MODEL} | 数据: {config.DATA_SAMPLE_SIZE} 条 | "
        f"epochs: {config.NUM_EPOCHS}",
        fontsize=13,
    )
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    os.makedirs(os.path.dirname(config.LOSS_CURVE_PATH), exist_ok=True)
    plt.tight_layout()
    plt.savefig(config.LOSS_CURVE_PATH, dpi=150, bbox_inches="tight")

    print(f"\n✓ 完成！Loss 曲线已保存到: {config.LOSS_CURVE_PATH}")


if __name__ == "__main__":
    main()
