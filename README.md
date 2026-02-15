# NPU MFU Analyzer

**昇腾 NPU 大模型训练 MFU 智能分析与优化工具**

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)

## 概述

NPU MFU Analyzer 是一款专为昇腾 NPU 设计的大模型训练性能分析工具。基于 Multi-Agent 架构和专家技能引擎，提供从数据解析到性能优化建议的端到端分析能力。

### 核心特性

| 模块 | 功能 |
|------|------|
| **Multi-Agent 分析** | 5 大专业 Agent：Timeline/Operator/Memory/Communication/Jitter |
| **Profiling 对比分析** | 两次 Profiling 深度对比：相似度检测、5 层级差异分析、根因定位 |
| **硬件感知** | Atlas A2 (280T/313T/376T)、Atlas 300I 规格库 |
| **模式识别** | 自动识别 Megatron/DeepSpeed/FSDP/MindSpeed 框架 |
| **拓扑分析** | 多机多卡物理拓扑、HCCS/RDMA 带宽分析 |
| **专家技能** | 14 个内置技能（Python 精确计算 + Prompt 推理指导） |
| **性能建模** | Roofline 天花板分析、理论 MFU 上限 |
| **假设分析** | 并行配置/硬件升级/优化措施效果预测 |
| **AIKG 代码生成** | 自动生成 Triton-Ascend 融合算子代码 |

## 快速开始

### 安装

```bash
# 克隆仓库
git clone <repo-url>
cd npu-mfu-analyzer

# 创建虚拟环境并激活
python3 -m venv .venv
source .venv/bin/activate    # Linux/macOS
# .\.venv\Scripts\Activate.ps1  # Windows PowerShell

# 安装依赖
pip install -e ".[all]"
```

> 📖 详细安装说明（含 Windows、Ollama 配置）请参考 [安装指南](docs/installation.md)

### 基本使用

```bash
# 查看 Profiling 信息
npu-analyzer info /path/to/profiling

# 生成数据摘要（不需要 LLM）
npu-analyzer summary /path/to/profiling

# 完整分析（使用 Ollama）
npu-analyzer analyze /path/to/profiling --backend ollama

# 使用 Claude API
npu-analyzer analyze /path/to/profiling --backend claude -m claude-3-opus-20240229

# 使用 DeepSeek API
npu-analyzer analyze /path/to/profiling --backend deepseek

# 启动 Web 界面
npu-analyzer web --port 8000
```

### Profiling 对比分析

```bash
# 对比两次 Profiling（如升级前后）
npu-analyzer compare /path/to/profiling_before /path/to/profiling_after

# 带版本标签的对比
npu-analyzer compare /path/to/v1 /path/to/v2 \
    --label-a "CANN 8.0" --label-b "CANN 8.1" \
    -o comparison_report.md

# 使用 LLM 进行深度根因分析
npu-analyzer compare /path/to/v1 /path/to/v2 -b openai

# 不同并行策略对比（跳过相似度检查）
npu-analyzer compare /path/to/tp4 /path/to/tp8 \
    --label-a "TP=4" --label-b "TP=8" --force

# 输出 HTML 格式报告
npu-analyzer compare /path/to/v1 /path/to/v2 -f html -o report.html
```

对比分析支持 5 个层级的差异检测：
- **Summary 级**：Step 时间、MFU、计算/通信/空闲占比变化
- **Timeline 级**：Step 稳定性、各阶段绝对时间变化
- **Operator 级**：Top 劣化/改善算子、新增/消失算子
- **Communication 级**：通信总时间、掩盖率、通信模式变化
- **Memory 级**：峰值内存变化

### AIKG 融合算子生成

```bash
# 生成融合算子代码（AIKG 工作流）
npu-analyzer generate /path/to/profiling

# 使用 Claude API (GLM-4.7) 生成
export CLAUDE_API_KEY="your_api_key"
npu-analyzer generate /path/to/profiling -b claude -m GLM-4.7

# 自定义参数
npu-analyzer generate /path/to/profiling \
    --output ./my_kernels \
    --min-speedup 1.05 \
    --complexity 低
```

详细使用指南请参考：[AIKG 使用文档](docs/AIKG_USAGE.md)

### Python API

```python
import asyncio
from src.agents.orchestrator import Orchestrator
from src.llm import LLMConfig

async def analyze():
    config = LLMConfig(backend="ollama", model="qwen2.5:7b")
    orchestrator = Orchestrator("/path/to/profiling", llm_config=config)
    report = await orchestrator.run()
    print(report.final_report)

asyncio.run(analyze())
```

#### 对比分析 API

```python
import asyncio
from src.analyzers.comparison_orchestrator import ComparisonOrchestrator
from src.llm import LLMConfig

async def compare():
    orchestrator = ComparisonOrchestrator(
        path_a="/path/to/profiling_before",
        path_b="/path/to/profiling_after",
        label_a="升级前 v2.0",
        label_b="升级后 v2.1",
        llm_config=LLMConfig(backend="mock"),
    )
    report = await orchestrator.run()
    
    if report.success:
        print(f"结论: {report.diff.overall_verdict}")
        print(report.final_report)
    else:
        print(f"对比失败: {report.error}")

asyncio.run(compare())
```

## 系统架构

```
┌─────────────────────────────────────────────────────────────────┐
│                      NPU MFU Analyzer                           │
├─────────────────────────────────────────────────────────────────┤
│  CLI / Web Interface                                            │
│    analyze │ compare │ generate │ web │ ...                     │
├──────────────────────────┬──────────────────────────────────────┤
│  Multi-Agent Orchestrator│  Comparison Orchestrator             │
│  ┌────────┬────────┐    │  ┌────────────┬───────────────┐      │
│  │Timeline│Operator│... │  │ Similarity │ ProfilingDiff  │      │
│  │ Agent  │ Agent  │    │  │  Checker   │   Engine       │      │
│  └────────┴────────┘    │  └────────────┴───────────────┘      │
├──────────────────────────┴──────────────────────────────────────┤
│  Skill Engine │ Analyzers │ Roofline │ What-if Simulator       │
├─────────────────────────────────────────────────────────────────┤
│  Data Loader │ Hardware Registry │ Pattern Matcher │ Topology  │
├─────────────────────────────────────────────────────────────────┤
│  LLM Backend: Ollama │ DeepSeek │ OpenAI │ Claude │ Mock       │
└─────────────────────────────────────────────────────────────────┘
```

## 文档

| 文档 | 说明 |
|------|------|
| [安装指南](docs/installation.md) | 详细安装步骤、多系统支持、Ollama 配置 |
| [功能详解](docs/features.md) | 各模块功能说明和代码示例（含对比分析） |
| [配置说明](docs/configuration.md) | 环境变量、配置文件、LLM 后端配置 |
| [架构文档](docs/architecture.md) | 系统架构设计说明（含对比分析架构） |

## 项目结构

```
npu-mfu-analyzer/
├── src/
│   ├── agents/           # Multi-Agent 系统
│   ├── analyzers/        # 分析算法
│   ├── data_loader/      # 数据加载
│   ├── hardware/         # 硬件规格库
│   ├── pattern_matcher/  # 模式识别
│   ├── topology/         # 拓扑分析
│   ├── skills/           # 专家技能引擎
│   ├── roofline/         # 性能建模
│   ├── llm/              # LLM 接口
│   ├── report/           # 报告生成
│   ├── cli/              # 命令行
│   └── web/              # Web 界面
├── tests/
│   ├── unit/             # 单元测试
│   └── integration/      # 集成测试
├── docs/                 # 文档
├── examples/             # 示例
└── config/               # 配置
```

## 开发

```bash
# 安装开发依赖
pip install -e ".[dev]"

# 运行测试
pytest tests/ -v

# 代码检查
ruff check src/
```

## 版本历史

| 版本 | 更新内容 |
|------|---------|
| 0.4.0 | Profiling 对比分析：相似度检测、5 层级差异分析、LLM 根因分析 |
| 0.3.0 | Skill Engine、Roofline、What-if Simulator |
| 0.2.0 | Hardware Registry、Pattern Matcher、Topology |
| 0.1.0 | 基础功能、Multi-Agent、Web 界面 |

## License

Apache-2.0
