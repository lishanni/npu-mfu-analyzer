# 安装指南

## 环境要求

- Python >= 3.9, < 3.13
- pip >= 21.0
- 操作系统: Linux / macOS / Windows 11

## 安装步骤

### Linux / macOS

```bash
# 克隆仓库
git clone <repo-url>
cd npu-mfu-analyzer

# 创建虚拟环境
python3 -m venv .venv
source .venv/bin/activate

# 安装依赖
pip install -e ".[all]"

# 或分步安装
pip install -e .              # 基础功能
pip install -e ".[web]"       # Web 界面
pip install -e ".[llm]"       # LLM 后端
```

### Windows 11 (PowerShell)

```powershell
# 克隆仓库
git clone <repo-url>
cd npu-mfu-analyzer

# 创建虚拟环境
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 如果遇到执行策略限制，先运行:
# Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# 安装依赖
pip install -e ".[all]"
```

### Windows 11 (CMD)

```cmd
:: 克隆仓库
git clone <repo-url>
cd npu-mfu-analyzer

:: 创建虚拟环境
python -m venv .venv
.venv\Scripts\activate.bat

:: 安装依赖
pip install -e ".[all]"
```

## Ollama 本地 LLM 配置

### 安装 Ollama

**macOS:**
```bash
brew install ollama
```

**Linux:**
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

**Windows:**
从 [ollama.com](https://ollama.com) 下载安装包。

### 启动服务并下载模型

```bash
# 启动服务
ollama serve

# 下载推荐模型（新终端）
ollama pull qwen2.5:7b
```

### 验证安装

```bash
ollama list
# 应显示 qwen2.5:7b
```

## 可选依赖安装

NPU MFU Analyzer 使用 extras 管理可选依赖，按需安装：

```bash
# 安装所有依赖（推荐）
pip install -e ".[all]"

# 或按需安装
pip install -e .                  # 仅基础功能（CLI analyze/info/summary）
pip install -e ".[web]"           # Web 界面 + REST API + WebSocket
pip install -e ".[claude]"        # Claude API 后端
pip install -e ".[local-llm]"     # 本地大模型（transformers + torch）
pip install -e ".[langchain]"     # LangChain 集成
pip install -e ".[dev]"           # 开发工具（pytest/ruff/mypy）
```

### 依赖清单

| 安装组 | 主要依赖 | 用途 |
|--------|---------|------|
| **基础** | numpy, pandas, ijson, sqlalchemy, pyyaml, pydantic, click, networkx, jinja2, openai, aiohttp | CLI 分析核心 |
| **[web]** | fastapi, uvicorn, websockets, python-multipart, aiofiles | Web 界面 |
| **[claude]** | anthropic | Claude API 后端 |
| **[local-llm]** | transformers, torch, accelerate | 本地大模型推理 |
| **[langchain]** | langchain, langchain-openai | LangChain 集成 |
| **[dev]** | pytest, pytest-cov, pytest-asyncio, pytest-mock, ruff, mypy | 测试与检查 |

## 升级指南

```bash
# 激活虚拟环境
source .venv/bin/activate

# 拉取最新代码
git pull origin master

# 升级依赖
pip install -e ".[all]" --upgrade

# 验证安装
npu-analyzer version
```

## API Key 配置

### DeepSeek

```bash
export DEEPSEEK_API_KEY="sk-your-deepseek-key"
npu-analyzer analyze /path/to/profiling -b deepseek
```

### OpenAI

```bash
export OPENAI_API_KEY="sk-your-openai-key"
npu-analyzer analyze /path/to/profiling -b openai
```

### Claude / Anthropic

```bash
export ANTHROPIC_API_KEY="sk-ant-your-key"
npu-analyzer analyze /path/to/profiling -b claude
```

**使用兼容 API（如智谱 GLM）**：
```bash
export ANTHROPIC_BASE_URL="https://open.bigmodel.cn/api/anthropic"
export ANTHROPIC_API_KEY="your_glm_key"
npu-analyzer analyze /path/to/profiling -b claude -m GLM-4.7
```

## 验证安装

```bash
# 检查 CLI 是否可用
npu-analyzer version

# 检查 Profiling 数据加载
npu-analyzer info /path/to/profiling

# 测试 Mock 模式分析（不需要 LLM）
npu-analyzer analyze /path/to/profiling -b mock

# 运行测试
pytest tests/unit/ -v
```

## 常见问题

### Q: pip install 报错 "externally-managed-environment"
**A:** 使用虚拟环境安装，或添加 `--break-system-packages` 参数（不推荐）。

### Q: Windows 上 PowerShell 执行脚本被禁止
**A:** 以管理员身份运行：
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Q: Ollama 连接失败
**A:** 确认 Ollama 服务正在运行：
```bash
curl http://localhost:11434/api/tags
```

### Q: ijson 未安装导致大文件解析失败
**A:** `ijson` 是基础依赖，应随 `pip install -e .` 一起安装。如果缺失：
```bash
pip install ijson>=3.2.0
```

### Q: 如何在无网络环境使用
**A:** 使用 Ollama 本地部署或 Mock 后端：
```bash
# Ollama 本地模型
npu-analyzer analyze /path/to/profiling -b ollama

# Mock 模式（仅数据分析，无 LLM 推理）
npu-analyzer analyze /path/to/profiling -b mock
```
