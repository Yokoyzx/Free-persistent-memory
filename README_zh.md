# Mem0 Local — Hermes Agent 本地记忆提供者

为 [Hermes Agent](https://github.com/NousResearch/hermes-agent) 提供轻量级、完全本地化的记忆后端。**无需 Docker，不依赖任何外部服务**。

- **LLM 记忆提取** — 默认 [Agnes AI](https://agnes-ai.com/)（兼容 OpenAI 接口），可自由切换
- **中文 Embedding** — `bge-small-zh-v1.5` 通过 fastembed 本地运行（30MB，512维）
- **向量存储** — Qdrant 本地文件存储（无需服务端）
- **零 Docker** — 全部在 Hermes 的 Python 进程内运行

## 工作原理

```
Hermes Agent
  └─ memory.provider = mem0_local
       └─ plugins/memory/mem0_local/__init__.py
            ├─ LLM:       任选 OpenAI 兼容 API（Agnes / DeepSeek / 通义千问 等）
            ├─ Embedder:  fastembed + bge-small-zh（30MB，512维）
            └─ Vector DB:  Qdrant 本地磁盘存储
```

## 安装前提

- 已安装 [Hermes Agent](https://hermes-agent.nousresearch.com/)
- 一个 **LLM API Key**（任选一个，见下方说明）
- 网络连接（首次运行会下载约 30MB 的 embedding 模型）

---

## 选择 LLM（重要）

以下为国内可用的免费/低价 LLM 方案，任选其一即可：

| 提供商 | 推荐模型 | 获取 API Key |
|--------|---------|-------------|
| **[DeepSeek](https://platform.deepseek.com/)**（推荐） | `deepseek-chat` | [注册](https://platform.deepseek.com/) 有免费额度 |
| **[月之暗面 Kimi](https://platform.moonshot.cn/)** | `moonshot-v1-8k` | [注册](https://platform.moonshot.cn/) |
| **[阿里通义千问](https://help.aliyun.com/zh/model-studio/)** | `qwen-turbo` | [注册](https://help.aliyun.com/zh/model-studio/) |
| **[零一万物 Yi](https://platform.lingyiwanwu.com/)** | `yi-lightning` | [注册](https://platform.lingyiwanwu.com/) |
| **[Agnes AI](https://agnes-ai.com/)**（默认） | `agnes-2.0-flash` | [注册](https://agnes-ai.com/) |
| **[Groq](https://console.groq.com/)**（需海外网络） | `mixtral-8x7b-32768` | [注册](https://console.groq.com/keys) |

> ⚠️ **关于 Google Gemini**：Gemini API 在中国大陆地区无法使用。如果你有海外网络环境，也可以用 Google AI Studio 获取 Key，配置方式见下方。

## 快速安装

### 1. 安装 Python 依赖

```bash
# 找到 Hermes 的 Python
HERMES_PYTHON=$(dirname $(which hermes))/python

# Windows (git-bash) 下：
# HERMES_PYTHON=/c/Users/你的用户名/AppData/Local/hermes/hermes-agent/venv/Scripts/python

# 安装
$HERMES_PYTHON -m pip install mem0ai fastembed
```

> **Windows 注意：** 如果 `pip` 不存在，先运行 `$HERMES_PYTHON -m ensurepip`。

### 2. 配置 API Key

根据你选择的 LLM，在 `~/.hermes/.env`（或 `$HERMES_HOME/.env`）中添加对应的配置：

**DeepSeek（推荐国内用户使用）：**
```bash
# DeepSeek
DEEPSEEK_API_KEY=***   # 改为你的 DeepSeek API Key
```

**月之暗面 Kimi：**
```bash
# Kimi（Moonshot）
KIMI_API_KEY=***  Key
```

**通义千问：**

```bash
# 阿里通义千问
DASHSCOPE_API_KEY=***   # 改为你的 DashScope API Key
```

**【默认】Agnes AI：**
```bash
AGNES_API_KEY=***
AGNES_BASE_URL=https://apihub.agnes-ai.com/v1
AGNES_MODEL=agnes-2.0-flash
```

**Google Gemini（需海外网络）：**
```bash
GOOGLE_API_KEY=***   # 从 https://aistudio.google.com/apikey 获取
```

### 3. 创建配置文件

`~/.hermes/mem0_local.json`：

```json
{
  "embedder_model": "BAAI/bge-small-zh-v1.5",
  "user_id": "你的用户名",
  "agent_id": "hermes",
  "rerank": false
}
```

### 4. 安装插件

将插件复制到 Hermes 的插件目录：

```bash
cp -r plugins/memory/mem0_local ~/.hermes/plugins/memory/
```

或者将 [`plugins/memory/mem0_local/__init__.py`](plugins/memory/mem0_local/__init__.py) 的内容保存到 `~/.hermes/plugins/memory/mem0_local/__init__.py`。

### 5. 激活

```bash
hermes config set memory.provider mem0_local
```

### 6. 验证

```bash
# 启动测试会话
hermes chat -q "你记得关于我的什么？"

# 或运行测试脚本
$HERMES_PYTHON test_mem0_local.py
```

## 自定义 LLM（高级）

插件默认使用 `"provider": "openai"`（兼容 OpenAI 接口），DeepSeek、Kimi、通义千问、Agnes 等均可直接使用。

如果你想换用其他 provider，需要修改 `plugins/memory/mem0_local/__init__.py` 中 `_get_memory()` 方法的 `config_dict`：

```python
# 示例：换成 Kimi（月之暗面）
config_dict = {
    "llm": {
        "provider": "openai",  # Kimi 也是 OpenAI 兼容接口
        "config": {
            "model": "moonshot-v1-8k",
            "openai_base_url": "https://api.moonshot.cn/v1",
            "api_key": os.environ.get("KIMI_API_KEY"),
            "max_tokens": 2000,
        },
    },
    # ... embedder 和 vector_store 保持不动
}
```

## 记忆工具

激活后，Hermes 会获得以下工具：

| 工具 | 用途 |
|------|------|
| `mem0_profile` | 列出所有存储的记忆 |
| `mem0_search` | 语义搜索记忆 |
| `mem0_conclude` | 保存一条事实性记忆 |

对话中的记忆会被自动提取，并在每次对话前自动召回。

## 重置数据

删除本地数据库以重新开始：

```bash
rm -rf ~/.hermes/mem0_data/
```

## 常见问题

| 问题 | 解决方法 |
|------|---------|
| 出现 503 模型未找到 | 模型名区分大小写，检查 `.env` 中的配置 |
| 向量维度不匹配 | 删除 `~/.hermes/mem0_data/` 后重启 |
| 插件未生效 | 运行 `hermes memory status`，然后**新开会话** |

## 文件清单

| 路径 | 用途 |
|------|------|
| `~/.hermes/.env` | LLM API 凭证 |
| `~/.hermes/mem0_local.json` | 提供者配置 |
| `~/.hermes/plugins/memory/mem0_local/` | 插件代码 |
| `~/.hermes/mem0_data/` | 本地向量数据库 |

## 致谢

- [Hermes Agent](https://github.com/NousResearch/hermes-agent) — AI 代理框架
- [Mem0](https://github.com/mem0ai/mem0) — 开源记忆层（Apache 2.0）
- [Agnes AI](https://agnes-ai.com/) — LLM API 提供商
- [fastembed](https://github.com/qdrant/fastembed) — 轻量级 Embedding 推理库
- [BAAI/bge-small-zh](https://huggingface.co/BAAI/bge-small-zh-v1.5) — 中文 Embedding 模型

## 许可证

Apache 2.0
