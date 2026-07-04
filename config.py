"""统一配置中心：所有脚本共享的常量定义。

包含基座模型、LoRA 参数、训练超参、数据路径、推理问题等。
所有脚本（prepare_data / train_lora / inference_compare / plot_loss）均从此处导入配置。
"""

# ============ 基座模型 ============
BASE_MODEL = "Qwen/Qwen1.5-0.5B"
ADAPTER_PATH = "./results/lora_adapter"

# ============ LoRA 配置 ============
# 与交接文档对齐：r=8, alpha=16, q_proj+v_proj
# 原理：ΔW = A·B，用 [d, r]·[r, d] 两个低秩矩阵近似 [d, d] 增量
# r=8 表示秩为 8，大幅减少可训练参数（约减少 256 倍）
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ["q_proj", "v_proj"]

# ============ 训练超参 ============
# 显存友好配置：batch=1 + 梯度累积=8（等效 batch_size=8）
MAX_SEQ_LEN = 512
BATCH_SIZE = 1
GRAD_ACCUM_STEPS = 8
LEARNING_RATE = 2e-4
NUM_EPOCHS = 3
WARMUP_RATIO = 0.03
SEED = 42
LOGGING_STEPS = 10
SAVE_STEPS = 500

# ============ 数据 ============
DATASET_NAME = "shibing624/medical"
DATASET_SPLIT = "finetune/train_zh_0.json"  # 纯中文 SFT 子集（约 194 万条）
DATA_SAMPLE_SIZE = 2000
PROCESSED_DATA_PATH = "./data/medical_qa_2000.jsonl"

# ============ Prompt 模板（Qwen ChatML）============
SYSTEM_PROMPT = "你是一个专业的医疗助手。"

# ============ 推理对比问题 ============
MEDICAL_QUESTIONS = [
    "感冒了应该怎么办？",
    "高血压患者的饮食注意事项有哪些？",
    "什么是糖尿病？有哪些常见症状？",
    "儿童发烧38.5度需要怎么处理？",
    "长期失眠应该怎么调理？",
]

# ============ 生成参数 ============
MAX_NEW_TOKENS = 512
TEMPERATURE = 0.7
TOP_P = 0.9
DO_SAMPLE = True

# ============ 路径 ============
RESULTS_DIR = "./results"
LOSS_LOG_PATH = "./results/loss_log.json"
LOSS_CURVE_PATH = "./results/loss_curve.png"
COMPARISON_PATH = "./results/comparison_results.txt"
