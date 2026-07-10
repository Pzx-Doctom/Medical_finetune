# 📘 第4阶段：Ollama 本地模型接入实战

> 本文档以 SyncHealth 项目为案例，完整讲解如何将 Ollama 本地大模型接入现有 AI 对话系统，实现 DeepSeek 云端 + Ollama 本地的双 Provider 架构，并支持自动降级（Fallback）。

---

## 目录

1. [需求背景与架构概览](#1-需求背景与架构概览)
2. [Provider 工厂模式：单例 + 多 Provider 管理](#2-provider-工厂模式单例--多-provider-管理)
3. [新建 OllamaProvider：本地模型的专用适配器](#3-新建-ollamaprovider本地模型的专用适配器)
4. [Fallback 辅助函数：DeepSeek 失败时自动降级](#4-fallback-辅助函数deepseek-失败时自动降级)
5. [后端 API 修改：model 路由 + 新端点](#5-后端-api-修改model-路由--新端点)
6. [前端修改：模型选择器 + 状态指示灯](#6-前端修改模型选择器--状态指示灯)
7. [环境配置与开关控制](#7-环境配置与开关控制)
8. [数据流全景图](#8-数据流全景图)
9. [踩坑实战](#9-踩坑实战)
10. [验证清单](#10-验证清单)

---

## 1. 需求背景与架构概览

### 1.1 改造前：单一 Provider

```
用户 → 前端 → Backend API → OpenAIProvider → DeepSeek API（唯一路径）
```

项目已有 Provider 工厂模式：`get_provider()` 根据 `AI_PROVIDER` 环境变量返回一个 `OpenAIProvider` 实例。DeepSeek 不好用时**直接报错**，用户无法使用。

### 1.2 改造后：双 Provider + 自动 Fallback

```
用户 → 前端（模型选择器 + 状态灯）
           ↓
     Backend API（model 参数路由）
        ↙         ↘
  DeepSeek（主）   Ollama（备）
       ↘         ↙
    Fallback 辅助函数（DeepSeek 失败 → 自动切 Ollama）
```

### 1.3 核心设计原则

| 原则 | 说明 |
|------|------|
| **原路径零破坏** | `get_provider()` 完全不变，DeepSeek 正常时等价于原逻辑 |
| **Ollama 纯兜底** | 本地模型能力弱，只在 DeepSeek 故障时启用，不作为主路径 |
| **开关可控** | `AI_FALLBACK_ENABLED=false` 时完全不引入 Ollama |
| **模型动态发现** | 前端下拉框通过 `ollama list` 动态查询，不写死模型名 |

### 1.4 文件清单

```
新增（2 个）：
  backend/app/services/ai/provider_ollama.py    ← Ollama 专用 Provider
  backend/app/services/ai/provider_fallback.py  ← Fallback 辅助函数

修改（10 个）：
  backend/app/config.py                         ← +AI_FALLBACK_ENABLED + OLLAMA_*
  backend/app/services/ai/base.py               ← GenerationConfig +model 字段
  backend/app/services/ai/factory.py            ← +get_ollama_provider() 单例
  backend/app/schemas/ai.py                     ← +新 schema
  backend/app/api/ai.py                         ← +模型路由 + 新端点
  backend/app/services/ai/agent_runtime.py      ← +model 参数
  backend/.env                                  ← +Ollama 配置
  frontend/src/types/ai.ts                      ← +新类型
  frontend/src/api/ai.ts                        ← +getModels/getHealth
  frontend/src/stores/ai.ts                     ← +模型状态管理
  frontend/src/views/AIChatView.vue             ← +模型选择器 UI
```

---

## 2. Provider 工厂模式：单例 + 多 Provider 管理

### 2.1 已有实现（改造前）

```python
# factory.py（改造前）
_provider_instance: BaseLLMProvider | None = None

def get_provider() -> BaseLLMProvider:
    global _provider_instance
    if _provider_instance is not None:
        return _provider_instance          # ← 单例：只创建一次

    provider_type = settings.AI_PROVIDER   # openai | domestic | local
    # 所有分支都返回 OpenAIProvider（因为 DeepSeek/OpenAI 都用同一协议）
    from app.services.ai.provider_openai import OpenAIProvider
    _provider_instance = OpenAIProvider()
    return _provider_instance
```

**关键点**：单例模式 + 全局变量缓存。任何时候调用 `get_provider()` 都返回同一个实例。

### 2.2 改造后：新增 Ollama 独立单例

```python
# factory.py（改造后）
_provider_instance: BaseLLMProvider | None = None
_ollama_instance: BaseLLMProvider | None = None   # ← 新增！Ollama 独立缓存

def get_provider() -> BaseLLMProvider:
    """主 provider（DeepSeek）—— 完全不变！"""
    # ... 完全不变 ...

def get_ollama_provider() -> BaseLLMProvider:  # ← 新增！
    """Ollama 备用 provider 独立单例"""
    global _ollama_instance
    if _ollama_instance is not None:
        return _ollama_instance
    from app.services.ai.provider_ollama import OllamaProvider
    _ollama_instance = OllamaProvider()
    return _ollama_instance

def reset_provider():
    """重置所有单例"""
    global _provider_instance, _ollama_instance
    _provider_instance = None
    _ollama_instance = None
```

**设计要点**：

| 问题 | 解决方案 |
|------|---------|
| 为什么不用 FallbackProvider 包装 `get_provider()`？ | 会改变所有调用方的行为，原路径被破坏 |
| `get_provider()` 和 `get_ollama_provider()` 为什么独立？ | DeepSeek 正常时根本不需要 Ollama 存在，独立单例互不干扰 |
| `reset_provider()` 为什么也要重置 Ollama？ | 保证重置行为的完整性，配置变更后两个都重建 |

### 2.3 单例模式的核心思想

```python
# Python 单例的经典实现：模块级变量 + lazy import
_instance = None

def get_instance():
    global _instance
    if _instance is None:      # ← 只在第一次调用时创建
        from x import X
        _instance = X()
    return _instance           # ← 后续调用都返回同一个
```

**为什么延迟导入？** `from x import X` 写在函数内而不是文件顶部，因为 `X` 可能依赖其他模块，而在模块加载时这些依赖还不存在。这避免了循环导入。

---

## 3. 新建 OllamaProvider：本地模型的专用适配器

### 3.1 为什么不能复用 OpenAIProvider？

尽管 Ollama 提供了 OpenAI 兼容端点 `/v1/chat/completions`，但直接复用 `OpenAIProvider` 有三大问题：

| 问题 | OpenAIProvider 行为 | Ollama 需要的行为 |
|------|-------------------|------------------|
| **认证** | 发送 `Authorization: Bearer {api_key}` | Ollama 免认证，不需要 Authorization 头 |
| **超时** | 60s / 120s | 本地推理慢，需要 180s / 300s |
| **上下文窗口** | 硬编码 128000 | 本地模型窗口小，需要可配置（默认 4096） |
| **模型列表** | 不支持 | 需要调用 `/api/tags` 动态获取 |
| **健康检查** | 不支持 | 需要检测 Ollama 服务是否在线 |

### 3.2 类结构

```python
class OllamaProvider(BaseLLMProvider):
    def __init__(self):
        # 1. 读取专属配置（OLLAMA_BASE_URL 优先，回退到 AI_BASE_URL）
        self.base_url = settings.OLLAMA_BASE_URL.rstrip("/")
        self.v1_url = f"{self.base_url}/v1"            # OpenAI 兼容端点
        self.model = settings.OLLAMA_MODEL               # 默认模型
        self.timeout = settings.OLLAMA_TIMEOUT           # 180s
        self.stream_timeout = settings.OLLAMA_STREAM_TIMEOUT  # 300s

    # 继承自 BaseLLMProvider 的抽象方法
    async def chat()           # 同步对话
    async def stream_chat()    # 流式对话
    def get_model_info()       # 模型信息

    # Ollama 专属方法（BaseLLMProvider 没有的）
    async def health_check()   # 健康检查（调 /api/tags）
    async def list_models()    # 列出本地模型（调 /api/tags）
```

### 3.3 免认证头

```python
# OllamaProvider（不需要 Authorization）
def _headers(self) -> dict:
    return {"Content-Type": "application/json"}

# 对比：OpenAIProvider（需要 Authorization）
def _headers(self) -> dict:
    return {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {settings.AI_API_KEY}",  # ← Ollama 不需要这个
    }
```

### 3.4 trust_env=False（关键！）

```python
async def chat(self, messages, config=None) -> str:
    #                 ↓↓↓ 必须加！否则 HTTP_PROXY 环境变量会导致 502
    async with httpx.AsyncClient(timeout=self.timeout, trust_env=False) as client:
        response = await client.post(...)
```

**为什么必须 `trust_env=False`？**

httpx 默认会读取系统环境变量 `HTTP_PROXY`、`HTTPS_PROXY`、`ALL_PROXY`。如果你在 IDE（如 CodeBuddy）的终端启动 uvicorn，IDE 可能注入代理环境变量，导致所有 httpx 请求都走代理服务器。代理服务器访问不到 `localhost:11434`，返回 502。

**经验法则**：访问 `localhost` 或 `127.0.0.1` 的服务时，永远设置 `trust_env=False`。

### 3.5 流式对话的 SSE 解析

```python
async def stream_chat(self, messages, config=None) -> AsyncIterator[str]:
    async with httpx.AsyncClient(...) as client:
        async with client.stream("POST", url, ...) as response:
            async for line in response.aiter_lines():
                if not line.startswith("data: "):     # ← 只处理 SSE data 行
                    continue
                data_str = line[6:]                    # ← 去掉 "data: " 前缀
                if data_str.strip() == "[DONE]":       # ← 流结束标志
                    break
                chunk = json.loads(data_str)
                content = chunk["choices"][0]["delta"].get("content", "")
                if content:
                    yield content                      # ← 逐 token 返回
```

SSE 流式格式（OpenAI 协议）：
```
data: {"choices":[{"delta":{"content":"你"}}]}

data: {"choices":[{"delta":{"content":"好"}}]}

data: [DONE]
```

### 3.6 模型管理与健康检查

```python
# 调用 Ollama 原生端点 /api/tags
# 返回格式：{"models": [{"name": "qwen2.5:7b", "size": 4683087332, ...}]}

async def health_check(self) -> dict:
    """检查服务是否在线 + 列出已安装模型"""
    try:
        resp = await client.get(f"{self.base_url}/api/tags")
        return {"status": "online", "models_count": N, "models": [...]}
    except httpx.ConnectError:
        return {"status": "offline", ...}      # ← 连接拒绝：Ollama 没启动
    except Exception as e:
        return {"status": "error", "error": str(e)}  # ← 其他错误（如 502）

async def list_models(self) -> list[dict]:
    """列出模型详细信息（name, size, parameter_size, quantization 等）"""
    # 同样调 /api/tags，但返回更详细的结构化信息
```

### 3.7 运行时模型覆盖

```python
def _build_body(self, messages, config, stream=False):
    cfg = config or GenerationConfig()
    model = cfg.model or self.model   # ← config.model 优先，没指定则用默认

# 使用示例：
config = GenerationConfig(model="qwen2.5:7b")   # 运行时切到 7b
config = GenerationConfig()                      # 用默认模型（.env 配置的）
```

---

## 4. Fallback 辅助函数：DeepSeek 失败时自动降级

### 4.1 为什么用辅助函数而不是包装类？

这是本次实现最关键的架构决策：

| 方案 | 做法 | 风险 |
|------|------|------|
| ❌ **包装类** FallbackProvider | `get_provider()` 返回 `FallbackProvider(DeepSeek, Ollama)` | 所有调用都经过包装层，DeepSeek 正常路径被改变 |
| ✅ **辅助函数** chat_with_fallback | `get_provider()` 不变，调用方改用 `chat_with_fallback()` 替代 `provider.chat()` | DeepSeek 成功时等价于原逻辑 + 一层 try |

```python
# ❌ 包装类方案（有风险）
def get_provider():
    return FallbackProvider(DeepSeek(), Ollama())  # 改变了返回值！

# ✅ 辅助函数方案（安全）
def get_provider():
    return OpenAIProvider()  # 完全不变！

# 调用方改动：
# 原：response = await provider.chat(messages)
# 改：response = await chat_with_fallback(messages)
```

### 4.2 chat_with_fallback 完整实现

```python
# 可降级的异常类型：仅限网络连接级别
_FALLBACK_EXCEPTIONS = (
    httpx.ConnectError,       # 连接被拒绝（服务挂了、网络不通）
    httpx.TimeoutException,   # 请求超时
    httpx.HTTPStatusError,    # HTTP 错误（5xx、401、403 等）
)

async def chat_with_fallback(messages, config=None) -> str:
    provider = get_provider()            # ← 主 provider（DeepSeek）
    try:
        return await provider.chat(messages, config)  # ← 先试主 provider
    except _FALLBACK_EXCEPTIONS as e:
        if not settings.AI_FALLBACK_ENABLED:          # ← 开关关了就不降级
            raise
        logger.warning(f"Primary failed: {e}, falling back to Ollama")
        ollama = get_ollama_provider()                # ← 切换到 Ollama
        return await ollama.chat(messages, config)
```

**为什么只捕获这三类异常？**

```python
# ✅ 应该降级的：网络级故障
httpx.ConnectError      # "连都连不上" → 切 Ollama 合理
httpx.TimeoutException  # "等了太久没回应" → 切 Ollama 合理
httpx.HTTPStatusError   # "服务器返回 5xx/401" → 切 Ollama 合理

# ❌ 不该降级的：业务异常
ValueError              # 参数错误 → 切 Ollama 也没用
json.JSONDecodeError    # 响应格式异常 → 可能是 provider 内部问题
```

**三层安全保障**：

```
1. AI_FALLBACK_ENABLED=false → try 块内直接 re-raise，Ollama 代码完全不执行
2. 只捕获连接级异常 → 业务异常不会误触发降级
3. get_provider() 完全不变 → DeepSeek 正常时零额外开销
```

### 4.3 stream_chat_with_fallback 的边界条件

流式 fallback 比同步复杂得多，因为**部分响应已经发给了用户**：

```python
async def stream_chat_with_fallback(messages, config=None):
    provider = get_provider()
    first_token_received = False

    try:
        async for chunk in provider.stream_chat(messages, config):
            first_token_received = True           # ← 标记：已有输出
            yield chunk                           # ← 直接发给前端
        return  # ← 成功完成，不 fallback

    except _FALLBACK_EXCEPTIONS as e:
        if first_token_received:
            raise  # ← 无法切换！部分内容已返回前端，贸然切换会导致内容重复
        if not settings.AI_FALLBACK_ENABLED:
            raise
        # 只在第一个 token 到达前失败才切换
        ollama = get_ollama_provider()
        async for chunk in ollama.stream_chat(messages, config):
            yield chunk
```

**时间线示例**：

```
场景 A（可降级）：
  DeepSeek → [连接失败] → Ollama 重新开始 → 用户看到完整回复

场景 B（不可降级）：
  DeepSeek → "你的心率..." → [超时] → 抛出异常
                ↑ 这部分已经发给用户了，无法撤回，只能报错
```

---

## 5. 后端 API 修改：model 路由 + 新端点

### 5.1 model 参数路由逻辑

当请求携带 `model` 字段时，后端根据值决定走哪个 provider：

```python
# API 层（ai.py）和 Agent Runtime 层（agent_runtime.py）的 model 路由逻辑：

if model and model != settings.AI_MODEL:        # model 是 ollama 模型名
    ollama = get_ollama_provider()              # 直接走 Ollama，不经过 fallback
    response = await ollama.chat(messages, config)
else:                                            # model 为空或等于 DeepSeek 模型
    response = await chat_with_fallback(messages, config)  # 走主 provider + 降级
```

**关键点**：

| model 值 | 走的路径 |
|----------|---------|
| `None` / `""` | `chat_with_fallback`（DeepSeek → 失败切 Ollama） |
| `"deepseek-chat"` | `chat_with_fallback`（同上） |
| `"qwen2.5:1.5b"` | 直接 `get_ollama_provider().chat()`（不降级） |

### 5.2 新增 API 端点

```python
# GET /ai/models —— 列出所有可用模型
@router.get("/models")
async def list_models(current_user=Depends(get_current_user)):
    cloud_models = [{"name": settings.AI_MODEL, "is_cloud": True}]
    local_models = await get_ollama_provider().list_models()  # 动态查 /api/tags
    return {
        "cloud_models": cloud_models,
        "local_models": local_models,     # ← 动态的！ollama pull 后自动出现
        "default_model": settings.AI_MODEL,
    }

# GET /ai/health —— 双 provider 健康检查
@router.get("/health")
async def ai_health(current_user=Depends(get_current_user)):
    ollama_status = await get_ollama_provider().health_check()
    return {
        "primary": {"status": "online", "models": [settings.AI_MODEL]},
        "ollama": ollama_status,  # online | offline | error
        "fallback_enabled": settings.AI_FALLBACK_ENABLED,
    }
```

### 5.3 关键的安全守门人

```python
if hasattr(ollama, "list_models"):     # ← 如果不是 OllamaProvider，fallback 到空列表
    models = await ollama.list_models()
```

`hasattr` 守卫确保：
- 如果未来 `get_ollama_provider()` 被换成其他 provider（没有 `list_models` 方法），不会报错
- Ollama 挂了返回空列表，不影响 API 响应

---

## 6. 前端修改：模型选择器 + 状态指示灯

### 6.1 数据流

```
页面加载 → onMounted() 调用 fetchModels()
    → GET /ai/models 获取模型列表
    → GET /ai/health 获取健康状态
    → 更新 Pinia store: cloudModels, localModels, aiHealth
    → Vue 响应式渲染下拉框 + 状态灯
```

### 6.2 Pinia Store 新增状态

```typescript
// stores/ai.ts
const cloudModels = ref<OllamaModel[]>([])     // 云端模型列表
const localModels = ref<OllamaModel[]>([])     // 本地模型列表（动态！）
const currentModel = ref<string>('')           // 当前选中模型
const aiHealth = ref<AIHealth | null>(null)     // 双 provider 健康状态

async function fetchModels() {
    const [modelsRes, healthRes] = await Promise.all([
        aiApi.getModels(),     // ← 两个请求并发
        aiApi.getHealth(),
    ])
    localModels.value = modelsRes.data.local_models  // ← 动态模型列表
    defaultModel.value = modelsRes.data.default_model
    aiHealth.value = healthRes.data
    if (!currentModel.value) {
        currentModel.value = modelsRes.data.default_model  // ← 默认选 DeepSeek
    }
}
```

### 6.3 sendMessage 携带 model 参数

```typescript
ws.onopen = () => {
    ws.send(JSON.stringify({
        message,
        session_id: currentSessionId.value || undefined,
        agent_id: agentId,
        model: currentModel.value || undefined,  // ← 新增！发给后端做路由
    }))
}
```

### 6.4 模型选择器 UI

```html
<!-- AIChatView.vue -->
<div class="chat-toolbar">
    <select v-model="aiStore.currentModel" class="model-selector">
        <optgroup label="云端模型">
            <option v-for="m in aiStore.cloudModels" :value="m.name">
                {{ m.name }} (云端)
            </option>
        </optgroup>
        <optgroup label="本地模型" v-if="aiStore.localModels.length > 0">
            <option v-for="m in aiStore.localModels" :value="m.name">
                {{ m.name }}{{ m.parameter_size ? ` (${m.parameter_size})` : '' }} (本地)
            </option>
        </optgroup>
    </select>

    <!-- 双状态指示灯 -->
    <div class="status-group">
        <span class="status-item">
            <span class="status-dot" :class="aiStore.aiHealth?.primary?.status"></span>
            DeepSeek
        </span>
        <span class="status-item">
            <span class="status-dot" :class="aiStore.aiHealth?.ollama?.status"></span>
            Ollama
        </span>
        <span v-if="aiStore.aiHealth?.fallback_enabled" class="fallback-badge">
            自动降级
        </span>
    </div>
</div>
```

**CSS 状态灯**：

```css
.status-dot { width: 8px; height: 8px; border-radius: 50%; }
.status-dot.online  { background: #10B981; }  /* 绿色：在线 */
.status-dot.offline { background: #9CA3AF; }  /* 灰色：离线 */
.status-dot.error   { background: #EF4444; }  /* 红色：错误 */
.status-dot.unknown { background: #D1D5DB; }  /* 浅灰：未知 */
```

---

## 7. 环境配置与开关控制

### 7.1 新增配置项

```env
# backend/.env

# === DeepSeek（主 provider，保持不变） ===
AI_PROVIDER=domestic
AI_BASE_URL=https://api.deepseek.com/v1
AI_API_KEY=sk-xxx
AI_MODEL=deepseek-chat

# === Fallback 开关 ===
AI_FALLBACK_ENABLED=true         # true=开启降级, false=关闭（行为与改造前一致）

# === Ollama（备用 provider） ===
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen2.5:1.5b       # fallback 触发时用的默认模型
OLLAMA_CONTEXT_WINDOW=4096       # 本地模型上下文窗口大小
OLLAMA_TIMEOUT=180               # 本地推理慢，超时给足
OLLAMA_STREAM_TIMEOUT=300
```

### 7.2 配置的 Pydantic 定义

```python
# config.py
class Settings(BaseSettings):
    # 原有配置
    AI_PROVIDER: str = "openai"
    AI_BASE_URL: str = "https://api.openai.com/v1"
    AI_API_KEY: str = ""
    AI_MODEL: str = "gpt-4o"

    # Fallback 开关
    AI_FALLBACK_ENABLED: bool = False  # ← 默认关闭，显式开启才启用

    # Ollama 配置
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "qwen2.5:7b"
    OLLAMA_CONTEXT_WINDOW: int = 4096
    OLLAMA_TIMEOUT: int = 180
    OLLAMA_STREAM_TIMEOUT: int = 300
```

### 7.3 开关的三种用法

| AI_FALLBACK_ENABLED | 行为 |
|---------------------|------|
| `false`（默认） | 降级代码完全不执行，行为与改造前完全一致 |
| `true` + DeepSeek 正常 | DeepSeek 直接响应，Ollama 代码不执行 |
| `true` + DeepSeek 故障 | 自动降级到 Ollama，日志输出警告 |

---

## 8. 数据流全景图

### 场景 A：DeepSeek 正常（默认路径，95% 的时间）

```
用户发消息 → WS /ai/chat/ws {model: "deepseek-chat"}
  → stream_chat_with_fallback()
    → get_provider().stream_chat()           ← 直接调 DeepSeek
    → DeepSeek 正常返回 tokens
    → yield 给前端
```

与改造前唯一差异：多了一层 `try/except`，但 try 块正常执行时不触发 except，**性能等价**。

### 场景 B：DeepSeek 不可用（自动降级）

```
用户发消息 → WS /ai/chat/ws {model: "deepseek-chat"}
  → stream_chat_with_fallback()
    → get_provider().stream_chat()
    → DeepSeek 抛出 ConnectError / TimeoutException
    → 捕获异常 → first_token == False → 可以切换
    → get_ollama_provider().stream_chat()
    → Ollama 返回 tokens → yield 给前端
    → 日志: "Primary provider failed: ..., falling back to Ollama"
```

### 场景 C：用户手动选 Ollama 模型

```
用户选 qwen2.5:1.5b → WS /ai/chat/ws {model: "qwen2.5:1.5b"}
  → model != settings.AI_MODEL → 直接走 Ollama
  → get_ollama_provider().stream_chat()
  → Ollama 返回 tokens → 不经过 fallback 逻辑
```

### 8.1 完整请求链路追踪（以场景 A 为例）

下面逐步追踪"用户在前端输入 `我的睡眠怎么样？`"从浏览器到 DeepSeek 再回到浏览器的完整过程。

#### 阶段 ①：前端发起请求

```
用户点击发送按钮
    ↓
AIChatView.vue: sendMessage()
    ↓
stores/ai.ts: sendMessage(message)
    ├─ 1. 将用户消息推到 messages[]（立即显示在聊天界面）
    ├─ 2. 创建空的 assistant 占位消息（streaming: true）
    ├─ 3. 创建 WebSocket 连接：
    │     ws://127.0.0.1:8000/api/v1/ai/chat/ws?token=eyJxxx...
    │     端口：开发环境 8000，生产环境走 Nginx 同源
    └─ 4. ws.onopen 时发送 JSON：
          {
            "message": "我的睡眠怎么样？",
            "session_id": 5,
            "model": "deepseek-chat"     ← 从下拉框读取
          }
```

**关键代码**（`stores/ai.ts` 第 116-123 行）：

```typescript
ws.onopen = () => {
    ws!.send(JSON.stringify({
        message,                                // 用户输入
        session_id: currentSessionId.value,     // 当前会话 ID
        agent_id: agentId,                      // 可选：自定义 Agent
        model: currentModel.value,              // 选中的模型名
    }))
}
```

#### 阶段 ②：后端 WebSocket 握手与认证

```
FastAPI 收到 WebSocket 升级请求
    ↓
ai.py: chat_ws(websocket, token)
    ↓
decode_token(token)
    ├─ 成功 → user_id = payload["sub"]（用户 ID）
    └─ 失败 → websocket.close(code=4001)
    ↓
await websocket.accept()    ← WebSocket 握手完成，通道建立
```

**WebSocket 用 query param 传 token**（不是 HTTP Header），因为浏览器 WebSocket API 不支持自定义 Header。

#### 阶段 ③：解析消息 + 创建/加载会话

```
while True:  ← 可以发多轮消息，同一连接复用
    ↓
raw = await websocket.receive_text()  ← 等待客户端消息
data = json.loads(raw)
message = "我的睡眠怎么样？"
session_id = 5
model = "deepseek-chat"
    ↓
# 创建或加载会话（ChatSession 表）
if session_id:
    session = await db.get(ChatSession, 5)
else:
    session = ChatSession(user_id=2)   ← 新建会话
```

#### 阶段 ④：构建 LLM 上下文（核心编排逻辑）

这是 SyncHealth 最核心的部分——不是简单传消息，而是构建**三层上下文注入**：

```
messages = []

# 第 1 层：系统提示词
messages.append(ChatMessage(role="system", content=
    "You are SyncHealth AI, a knowledgeable and friendly health assistant..."
))

# 第 2 层：用户健康数据（从数据库查询并格式化）
health_context = await build_health_context(db, user_id=2, message="我的睡眠怎么样？")
# health_context 示例：
#   ## Sleep Summary
#   - Average sleep: 7h 12min (last 7 days)
#   - Sleep stages: Deep 1.5h, Core 4h, REM 1.7h
#   - Trend: Sleep duration ↑ 15min vs previous week
messages.append(ChatMessage(role="system", content=f"User's Health Data:\n\n{health_context}"))

# 第 3 层：Dify 医学知识库检索（RAG）
dify_records = await retrieve_from_dify("我的睡眠怎么样？")  ← 语义检索
dify_context = format_dify_context(dify_records)            ← 格式化为文本
# dify_context 示例：
#   [Document: 内科.txt, Score: 0.89]
#   成年人每晚建议睡眠 7-9 小时，深度睡眠应占总睡眠的 15-25%...
messages.append(ChatMessage(role="system", content=f"Medical Knowledge:\n\n{dify_context}"))

# 第 4 层：对话历史（最近 20 条）
for h in history:
    messages.append(ChatMessage(role=h.role, content=h.content))

# 第 5 层：用户当前消息
messages.append(ChatMessage(role="user", content="我的睡眠怎么样？"))
```

**最终发给 LLM 的 messages 数组**：

```
[0] system:  "You are SyncHealth AI..."              ← 系统提示词
[1] system:  "User's Health Data:\n\n..."             ← 健康数据
[2] system:  "Medical Knowledge Reference:\n\n..."    ← 医学知识
[3] user:    "上周的心率怎么样？"                     ← 历史对话
[4] assistant:"您上周的心率..."                       ← 历史回复
[5] user:    "我的睡眠怎么样？"                       ← 当前消息
```

#### 阶段 ⑤：保存用户消息 + 模型路由

```python
# 1. 保存用户消息到数据库
user_msg = ChatMessageModel(
    session_id=5,
    role="user",
    content="我的睡眠怎么样？",
    health_context_snapshot=health_context,   # ← 保存上下文快照
    dify_context_snapshot=dify_context,       # ← 保存知识库快照
)
db.add(user_msg)

# 2. 模型路由判断
config = GenerationConfig(model="deepseek-chat")

if model and model != settings.AI_MODEL:
    # model="qwen2.5:1.5b" ≠ settings.AI_MODEL="deepseek-chat"
    # → 走 Ollama 分支
    ollama = get_ollama_provider()
    async for chunk in ollama.stream_chat(messages, config):
        ...
else:
    # model 为空或等于 "deepseek-chat"
    # → 走 Fallback 分支
    async for chunk in stream_chat_with_fallback(messages, config):
        ...
```

#### 阶段 ⑥：Fallback 辅助函数内部

```python
# provider_fallback.py: stream_chat_with_fallback()
provider = get_provider()                    # → OpenAIProvider 实例
first_token_received = False

try:
    async for chunk in provider.stream_chat(messages, config):
        first_token_received = True          # ← 标记：已有输出
        yield chunk                          # → 发给 ai.py WS handler

except (ConnectError, TimeoutException, HTTPStatusError) as e:
    if first_token_received:                 # ← 已输出内容，无法切换
        raise
    if not settings.AI_FALLBACK_ENABLED:     # ← 开关关闭
        raise
    # 降级到 Ollama
    ollama = get_ollama_provider()
    async for chunk in ollama.stream_chat(messages, config):
        yield chunk
```

#### 阶段 ⑦：Provider 实际发送 HTTP 请求

```python
# provider_openai.py: OpenAIProvider.stream_chat()
async with httpx.AsyncClient(timeout=120.0) as client:    # 120s 超时
    async with client.stream(
        "POST",
        "https://api.deepseek.com/v1/chat/completions",   # ← DeepSeek API
        headers={
            "Authorization": "Bearer sk-xxx",              # ← API Key
            "Content-Type": "application/json",
        },
        json={
            "model": "deepseek-chat",
            "messages": [                                  # ← 前面构建的 6 条消息
                {"role": "system", "content": "You are SyncHealth AI..."},
                {"role": "system", "content": "User's Health Data:..."},
                {"role": "system", "content": "Medical Knowledge:..."},
                {"role": "user", "content": "上周的心率怎么样？"},
                {"role": "assistant", "content": "您上周的心率..."},
                {"role": "user", "content": "我的睡眠怎么样？"},
            ],
            "temperature": 0.7,
            "max_tokens": 2048,
            "stream": true,                                # ← 流式模式
        },
    ) as response:
        # 解析 SSE 流
        async for line in response.aiter_lines():
            # data: {"choices":[{"delta":{"content":"您"}}]}
            → yield "您"
            # data: {"choices":[{"delta":{"content":"的"}}]}
            → yield "的"
            # data: {"choices":[{"delta":{"content":"睡"}}]}
            → yield "睡"
            # ...逐 token 产出
            # data: [DONE]  → break
```

#### 阶段 ⑧：逐 token 返回前端

```
provider.stream_chat() yield "您"
    ↓
fallback 辅助函数 yield "您"    ← try 块内正常执行
    ↓
ai.py WS handler:
    full_response += "您"
    await websocket.send_text('{"type":"token","content":"您"}')
    ↓
前端 ws.onmessage:
    data.type === "token"
    streamingMsg.content += "您"   ← Vue 响应式更新 → 界面实时显示
    ↓
下一个 token ... "的" → "睡" → "眠" → ...
    ↓
最后一条 token 发送完毕
    ↓
provider.stream_chat() 结束 → fallback 函数 return
    ↓
ai.py: 保存 assistant 消息到数据库
    await websocket.send_text('{"type":"done","session_id":5,"dify_references":[...]}')
    ↓
前端 ws.onmessage:
    data.type === "done"
    streamingMsg.streaming = false   ← 停止加载动画
    closeWs()                        ← 关闭 WebSocket
```

#### 8.1.1 为什么要用 WebSocket 而不是 HTTP？

核心原因：**流式对话需要"打字机效果"**，不能等 AI 生成完再一次性返回。

**HTTP 方式（等 3 秒一次性返回）**：

```
客户端 ──POST /ai/chat──→ 服务器
                             ↓
                       DeepSeek 生成中...（3秒后完成）
                             ↓
客户端 ←── 完整回复 ────  服务器
```

用户盯着空白界面等 3 秒 → 体验很差。

**WebSocket 方式（边生成边推送）**：

```
客户端 ──WebSocket 握手──→ 服务器
                             ↓
客户端 ←── {"token":"您"} ──── 0.2s  ← 第一个字就到了！
客户端 ←── {"token":"的"} ──── 0.3s
客户端 ←── {"token":"睡"} ──── 0.4s
客户端 ←── {"token":"眠"} ──── 0.5s
        ...
客户端 ←── {"type":"done"} ──── 3.0s  ← 完成
```

0.2 秒用户就看到第一个字了——**不需要干等**。

另外，WebSocket 支持**同一连接多轮复用**。后端 WS Handler 用了 `while True`，一个连接可以持续发多轮消息。如果用 SSE（Server-Sent Events），每轮对话都要重建 HTTP 连接、重新认证，开销更大。

#### 8.1.2 WebSocket 四个回调是如何工作的？

看完阶段 ① 的代码你可能会问：`sendMessage()` 里只调用了 `ws.onopen` 发消息，后面的 `ws.onmessage`、`ws.onerror`、`ws.onclose` 是谁调用的？

**答案：浏览器内核调用。你的代码只赋值，浏览器在合适的时机调用。**

这四个属性是 **W3C 标准 API**（2011 年定稿），所有浏览器（Chrome / Firefox / Safari / Edge）行为一致：

| 属性 | 你做的 | 浏览器做的（调用时机） |
|------|--------|----------------------|
| `ws.onopen` | 赋值一个函数 | TCP 握手成功时调用 |
| `ws.onmessage` | 赋值一个函数 | 收到服务器推送数据时调用（可能 N 次） |
| `ws.onerror` | 赋值一个函数 | 网络层出错时调用 |
| `ws.onclose` | 赋值一个函数 | 连接关闭时调用 |

不是你的代码调用它们，而是你把函数"寄存"在 `ws` 对象上，浏览器发现对应事件时替你调用。所有浏览器 API 都遵循这个模式（`button.onclick`、`input.onkeydown`、`setTimeout(fn, 1000)` 等等）。

**为什么 `sendMessage()` 函数执行完了，后续代码还能跑？**

```typescript
ws.onopen = () => { ws.send(...) }    // 注册了回调
ws.onmessage = (e) => { ... }         // 注册了回调
ws.onclose = () => { ... }            // 注册了回调

} // ← sendMessage() 函数在这里返回了！

// 函数返回后，JavaScript 主线程没有被卡住
// 浏览器的事件循环（Event Loop）在后台持续运行：
//   "有事件吗？"
//     → TCP 握手成功？→ 调用 ws.onopen
//     → 收到数据？    → 调用 ws.onmessage
//     → 连接出错？    → 调用 ws.onerror
//     → 连接断开？    → 调用 ws.onclose
//   "还有吗？没有就等着..."
```

用一个餐厅类比帮助理解：

```
你（sendMessage）：服务员，牛排好了叫我 → 注册 onmessage 回调
你（主线程）：    然后回到座位继续喝饮料看手机 → 函数返回，不卡住
服务员（事件循环）：一直在前台，有事件就喊号
厨房（浏览器内核）：牛排好了 → 响铃
服务员（事件循环）：听到响铃 → 喊"123号！"
你（onmessage回调）：取餐 → 开始吃（追加 content）
```

三个概念的关系：
- **回调** = 你留给服务员的号码牌（`ws.onmessage = () => {}`——把函数存起来）
- **事件循环** = 前台服务员（浏览器内置调度器，不停检查"有新事件吗"）
- **异步调度** = 厨房喊号 + 服务员喊你（事件发生时，浏览器自动调用对应的回调函数）

#### 8.1.3 `ws.onopen` 是浏览器怎么发现并调用的？

```typescript
// 你的代码
ws.onopen = () => { console.log('连上了') }
//         ↑
//  把箭头函数的内存地址存到 ws.onopen 属性里

// 浏览器内部（伪代码，C++ 实现的）
class WebSocket {
    onopen: Function | null = null
    
    constructor(url) {
        this._startTCPConnection(url)     // 发起 TCP 三次握手
    }
    
    _onConnected() {                      // TCP 握手成功 → C++ 通知 JS
        if (this.onopen !== null) {       // 你存了回调吗？
            this.onopen()                 // 存了 → 我帮你调用
        }
    }
}
```

`new WebSocket(url)` 之后，浏览器在后台做 DNS 解析 → TCP 三次握手 → WebSocket 协议升级。这些都在 C++ 层完成，你完全看不到。握手成功时，浏览器检查 `ws.onopen` 是不是函数，是就调用它。

如果你不写 `ws.onopen = ...`（属性是 `null`），连接照样建立，但**没人响应这个事件**——就像服务员喊了号但没人回应。

### 8.2 同一请求的两种路径对比

| 步骤 | DeepSeek 路径（场景 A） | Ollama 路径（场景 C） |
|------|------------------------|---------------------|
| 模型路由 | `model == settings.AI_MODEL` → Fallback 分支 | `model != settings.AI_MODEL` → Ollama 分支 |
| Provider | `get_provider()` → `OpenAIProvider` | `get_ollama_provider()` → `OllamaProvider` |
| HTTP 请求 | `POST https://api.deepseek.com/v1/chat/completions` | `POST http://localhost:11434/v1/chat/completions` |
| 认证头 | `Authorization: Bearer sk-xxx` | 无（免认证） |
| 超时 | 120s | 300s |
| 失败处理 | 自动降级到 Ollama | 直接报错 |

### 8.3 关键时间节点（典型耗时）

```
T=0ms    用户点击发送
T=5ms    WebSocket 连接建立
T=10ms   认证通过
T=50ms   数据库查询 + 健康数据构建 + Dify 检索
T=80ms   向 DeepSeek 发起 HTTP 请求
T=200ms  第一个 token 到达（TTFT = Time To First Token）
T=3000ms 最后一个 token，流结束
T=3010ms 保存到数据库，发送 done 事件
T=3015ms 前端关闭 WebSocket，加载动画结束
```

**经验值**：TTFT（首 token 延迟）是用户体验的关键指标。DeepSeek 云端约 100-300ms，Ollama 本地约 500-2000ms（取决于模型大小和硬件）。

---

## 9. 踩坑实战

### 坑 1：502 Bad Gateway（代理拦截）

**现象**：

```json
{"ollama": {"status": "error", "error": "Server error '502 Bad Gateway' for url 'http://localhost:11434/api/tags'"}}
```

**根因**：

CodeBuddy 终端启动 uvicorn 时继承了 IDE 的代理环境变量（`HTTP_PROXY`）。httpx 默认 `trust_env=True`，检测到代理后把所有请求（包括 `localhost`）都转发给代理服务器。代理服务器访问不到 `localhost:11434`，返回 502。

**修复**：

```python
# provider_ollama.py —— 所有 httpx.AsyncClient 都加 trust_env=False
async with httpx.AsyncClient(timeout=10.0, trust_env=False) as client:
    resp = await client.get(f"{self.base_url}/api/tags")
```

**经验**：访问 `localhost` 或内网 IP 的服务时，永远要 `trust_env=False`。

### 坑 2：PowerShell 的 curl 不是真正的 curl

**现象**：

```powershell
curl -H "Content-Type: application/json" -d '{"email":"xxx"}'
# 报错：无法将 System.String 转换为 System.Collections.IDictionary
```

**根因**：PowerShell 的 `curl` 是 `Invoke-WebRequest` 的别名，参数语法不同。

**解决方案**：

```powershell
# 用 Invoke-RestMethod（推荐）
$body = '{"email":"pan@163.com","password":"qwe123"}'
Invoke-RestMethod -Uri "http://localhost:8000/api/v1/auth/login" -Method Post -ContentType "application/json" -Body $body

# 或者用真正的 curl.exe（Windows 10+ 自带）
curl.exe -X POST http://localhost:8000/api/v1/auth/login -H "Content-Type: application/json" -d "{\"email\":\"pan@163.com\",\"password\":\"qwe123\"}"
```

### 坑 3：PowerShell 中文乱码

**现象**：API 返回的中文在 PowerShell 显示为 `æ¨å¥½`。

**原因**：PowerShell 5.x 默认编码不是 UTF-8。

**解决**：

```powershell
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
```

**注意**：这只是命令行显示问题，前端浏览器的中文正常显示。

### 坑 4：虚拟环境 Python vs 全局 Python

**现象**：

```powershell
python -c "import httpx"   # ModuleNotFoundError
# 但 uvicorn 能跑起来，后端能正常调用 httpx
```

**原因**：当前 PowerShell 用的是全局 Python（`C:\Python311\python.exe`），而 uvicorn 用的是虚拟环境的 Python（`backend\venv\Scripts\python.exe`）。

**经验**：诊断环境问题时，永远用虚拟环境的 Python 执行命令：
```powershell
backend\venv\Scripts\python -c "..."
```

### 坑 5：不要捕获太宽的异常

```python
# ❌ 错误：捕获了所有异常，连 ValueError/TypeError 也会降级
except Exception:
    fallback()

# ✅ 正确：只捕获连接级异常
except (httpx.ConnectError, httpx.TimeoutException, httpx.HTTPStatusError):
    fallback()
```

宽异常捕获会掩盖代码 bug（比如 `KeyError`），导致问题难以排查。

---

## 10. 验证清单

### 后端验证（命令行）

```powershell
# 1. 登录
$loginBody = '{"email":"pan@163.com","password":"qwe123"}'
$resp = Invoke-RestMethod -Uri "http://localhost:8000/api/v1/auth/login" -Method Post -ContentType "application/json" -Body $loginBody
$token = $resp.access_token

# 2. 健康检查（预期：primary.online + ollama.online）
(Invoke-RestMethod -Uri "http://localhost:8000/api/v1/ai/health" -Headers @{ Authorization = "Bearer $token" }) | ConvertTo-Json -Depth 5

# 3. 模型列表（预期：cloud_models=[deepseek-chat] + local_models=[qwen2.5:7b, qwen2.5:1.5b]）
(Invoke-RestMethod -Uri "http://localhost:8000/api/v1/ai/models" -Headers @{ Authorization = "Bearer $token" }) | ConvertTo-Json -Depth 5

# 4. 默认对话（DeepSeek）
$chatBody = '{"message":"你好"}'
Invoke-RestMethod -Uri "http://localhost:8000/api/v1/ai/chat" -Method Post -ContentType "application/json" -Headers @{ Authorization = "Bearer $token" } -Body $chatBody

# 5. 手动选 Ollama 模型
$chatBody = '{"message":"你好","model":"qwen2.5:1.5b"}'
Invoke-RestMethod -Uri "http://localhost:8000/api/v1/ai/chat" -Method Post -ContentType "application/json" -Headers @{ Authorization = "Bearer $token" } -Body $chatBody
```

### 前端验证（浏览器）

| 检查项 | 预期 |
|--------|------|
| 模型下拉框默认选中 `deepseek-chat (云端)` | ✅ |
| 下拉框有"云端模型"和"本地模型"分组 | ✅ |
| 本地模型组动态列出 `qwen2.5:7b`/`qwen2.5:1.5b` | ✅ |
| DeepSeek 状态灯绿色 | ✅ |
| Ollama 状态灯绿色 | ✅ |
| 显示"自动降级"徽章 | ✅ |
| 默认模型对话正常 | ✅ |
| 切换到 Ollama 模型对话正常 | ✅ |

### Fallback 验证（可选）

```powershell
# 1. 临时改错 API Key（模拟 DeepSeek 故障）
# 编辑 backend/.env：AI_API_KEY=wrong_key
# 2. 重启后端
# 3. 用默认模型发消息
$chatBody = '{"message":"你好"}'
Invoke-RestMethod -Uri "http://localhost:8000/api/v1/ai/chat" -Method Post -ContentType "application/json" -Headers @{ Authorization = "Bearer $token" } -Body $chatBody
# 预期：返回 Ollama 的回复，后端日志出现 "falling back to Ollama"
# 4. 改回正确 key
```

---

## 总结：快速回顾

| 概念 | 一句话 |
|------|--------|
| **Provider 工厂** | 单例模式：模块级全局变量缓存实例，延迟导入避免循环依赖 |
| **OllamaProvider** | 继承 BaseLLMProvider，免认证头 + 长超时 + trust_env=False |
| **Fallback 辅助函数** | 比包装类安全：get_provider() 不改，DeepSeek 正常时零影响 |
| **model 路由** | model 为空/等于默认 → fallback；否则直接走 Ollama |
| **trust_env=False** | 访问 localhost 服务时必须设置，否则代理环境变量导致 502 |
| **前端动态模型列表** | GET /ai/models 调 /api/tags 实时获取，不写死模型名 |
| **hasattr 守卫** | 调用 provider 专属方法前先检查，非 OllamaProvider 优雅降级 |
| **AI_FALLBACK_ENABLED** | 总开关，false 时行为与改造前完全一致 |

---

## 附录：简历与面试准备

### A.1 两种简历写法

#### 方案 1：单独补充一条（推荐，亮点独立突出）

在现有"可视化与 AI 对话"条目后新增第四条：

>  **LLM 多端接入与自动容灾**：基于 Provider 工厂模式适配 Ollama 本地模型，实现 DeepSeek 云端 + Ollama 本地的双 Provider 架构；设计流式 Fallback 机制，在首个 token 到达前捕获 ConnectError / Timeout / 5xx 自动切换备 Provider，配合 `AI_FALLBACK_ENABLED` 开关实现灰度控制；前端集成动态模型选择器与双 Provider 健康状态指示灯，运行时可无感切换。

**效果：四条形成从数据 → 知识 → 模型 → 服务的完整技术纵深**

```
 Apple健康数据ETL管道          ← 数据层优化
 RAG检索增强生成（Dify降级）     ← 知识库容错
 可视化与AI对话模块              ← 前端交互 + 对话编排
 LLM多端接入与自动容灾           ← LLM 服务容错（新增）
```

#### 方案 2：结合到已有描述中（精炼，不增加条目）

将第三条"可视化与 AI 对话模块"修改为（修改部分加粗）：

>  **健康数据可视化与 AI 对话模块**：基于ECharts构建步数/心率/睡眠/活动能量趋势图（7/30/90 天可切换）与四维雷达健康评分。问答侧基于 Provider 工厂模式接入 Ollama 本地模型，实现 DeepSeek 云端 + Ollama 本地的双端模型路由与流式 Fallback 自动容灾；通过关键词路由动态裁剪数据维度 + Dify 知识库语义检索增强，将医学知识片段按相关度得分格式化注入回答，支持引用来源溯源与运行时模型切换。

**建议用方案 1**：单独一条与 RAG 降级形成递进——从知识库容错到 LLM 服务容错，面试话题更丰富。

### A.2 面试话术扩展

**Q：说说你的 AI 服务高可用方案？**

> 我们的场景是 DeepSeek 作为主模型，用户希望在 DeepSeek 不可用时能自动切到本地 Ollama。我设计了双 Provider 架构，核心思路不是改已有的调用路径，而是在旁边新增了一个 `chat_with_fallback()` 辅助函数。调用方从 `provider.chat()` 改成 `chat_with_fallback()`，DeepSeek 正常时只是多了一层 try 包裹，性能无差异。失败时只捕获三种连接级异常（ConnectError/Timeout/5xx），不捕获业务异常避免误降级。流式场景有个边界条件——如果第一个 token 已经到了前端、后面再失败，没办法悄悄切换，因为用户已经看到了一部分内容。

**Q：本地模型列表怎么做到动态更新的？**

> 我让 Provider 直接调用 Ollama 的 `/api/tags` 端点，每次请求 `/ai/models` 都实时返回当前已安装的模型。前端下拉框完全依赖这个接口动态渲染，所以运维上 `ollama pull` 拉一个新模型，刷新页面下拉框里就出现了，不需要改任何代码。

**Q：遇到过什么棘手的问题？**

> 调试时发现健康检查一直返回 502。一开始以为是 Ollama 没启动，但 `ollama list` 是正常的。后来发现是 IDE 的终端会注入系统代理环境变量，httpx 默认 `trust_env=True`，把所有请求——包括 `localhost` 的——都转发给代理服务器了，代理当然访问不到本地的 11434 端口。最终加了 `trust_env=False` 解决，也沉淀了一条规范：访问本地服务的 HTTP 客户端永远要禁用环境变量代理。

### A.3 关键词提炼（填在线简历的技能标签）

```
LLM 集成、Provider 工厂模式、流式 Fallback、高可用架构、
FastAPI、WebSocket SSE、Ollama 本地部署、Python 异步编程、
httpx、Pydantic Settings、Vue 3 + Pinia 状态管理、
Dify RAG、ETL 数据管道、ECharts 数据可视化
```
