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

## 依赖说明

### 基础依赖
- pandas >= 2.0
- numpy >= 1.24
- ijson >= 3.2
- pyyaml >= 6.0

### Web 依赖 (可选)
- fastapi >= 0.100
- uvicorn >= 0.23
- websockets >= 11.0

### LLM 依赖 (可选)
- httpx >= 0.24
- aiohttp >= 3.8

### 开发依赖
- pytest >= 7.0
- ruff >= 0.1
- mypy >= 1.0

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
