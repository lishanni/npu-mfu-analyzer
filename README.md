# NPU MFU Analyzer

**昇腾 NPU 大模型训练 MFU 智能分析与优化工具**

[![GitHub stars](https://img.shields.io/github/stars/lishanni/npu-mfu-analyzer?style=social)](https://github.com/lishanni/npu-mfu-analyzer/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/lishanni/npu-mfu-analyzer?style=social)](https://github.com/lishanni/npu-mfu-analyzer/network/members)
[![GitHub watchers](https://img.shields.io/github/watchers/lishanni/npu-mfu-analyzer?style=social)](https://github.com/lishanni/npu-mfu-analyzer/watchers)

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)
[![Ascend](https://img.shields.io/badge/NPU-Ascend%20A2%2F300I-red)](https://www.hiascend.com/)
[![LLM](https://img.shields.io/badge/LLM-Claude%20%7C%20GPT%20%7C%20DeepSeek%20%7C%20Ollama-purple)](https://github.com/lishanni/npu-mfu-analyzer)
[![MFU Analysis](https://img.shields.io/badge/MFU-Performance%20Analyzer-orange)](https://github.com/lishanni/npu-mfu-analyzer)
[![Multi-Agent](https://img.shields.io/badge/Architecture-Multi--Agent-blueviolet)](https://github.com/lishanni/npu-mfu-analyzer)

## 概述

NPU MFU Analyzer 是一款专为昇腾 NPU 设计的大模型训练性能分析工具。基于 Multi-Agent 架构和专家技能引擎，提供从数据解析到性能优化建议的端到端分析能力。

### 核心特性

| 模块 | 功能 |
|------|------|
| **Multi-Agent 分析** | 13 大专业 Agent：Timeline/Operator/Memory/Communication/Jitter/AIC Microarch/Cluster 等 |
| **Profiling 对比分析** | 两次 Profiling 深度对比：相似度检测、5 层级差异分析、根因定位 |
| **Host-Device 堆栈关联** | Python/C++ 堆栈解析、算子来源分类、torch.compile/eager/融合算子识别 |
| **根因推理引擎** | 基于规则引擎的自动根因识别，支持单版本分析和对比分析 |
| **AIC 微架构分析** | AI Core 指令级/内存层次/流水线瓶颈诊断，Cube/Vector 利用率、L2 命中率 |
| **通信矩阵分析** | 链路级带宽利用率、慢链路检测、HCCS/RDMA 传输识别、HTML 热力图 |
| **链路性能仪表板** | 交互式 HTML 可视化：带宽热力图、趋势图表、异常链路分析 |
| **融合算子集成** | Profiling 驱动的算子融合机会发现与集成方案自动生成 |
| **硬件感知** | Atlas A2 (280T/313T/376T)、Atlas 300I 规格库 |
| **模式识别** | 自动识别 Megatron/DeepSpeed/FSDP/MindSpeed 框架 |
| **拓扑分析** | 多机多卡物理拓扑、HCCS Ring 解析、RDMA 带宽分析 |
| **专家技能** | 14 个内置技能（Python 精确计算 + Prompt 推理指导） |
| **性能建模** | Roofline 天花板分析、理论 MFU 上限 |
| **假设分析** | 并行配置/硬件升级/优化措施效果预测 |
| **AIKG 代码生成** | 自动生成 Triton-Ascend 融合算子代码 |
| **弹性 LLM** | 多后端调度、自动重试/降级/超时、连接池管理 |
| **数据验证** | Profiling 数据质量检测、异常修复、容错解析 |

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
# 查看版本信息
npu-analyzer version

# 查看 Profiling 数据信息（硬件、框架、数据规模）
npu-analyzer info /path/to/profiling

# 生成数据摘要（不需要 LLM，快速预览关键指标）
npu-analyzer summary /path/to/profiling --max-steps 10

# 完整分析（使用 Ollama 本地模型）
npu-analyzer analyze /path/to/profiling --backend ollama

# 使用 Claude API
npu-analyzer analyze /path/to/profiling --backend claude -m claude-3-opus-20240229

# 使用 DeepSeek API
npu-analyzer analyze /path/to/profiling --backend deepseek

# 生成 HTML 格式报告
npu-analyzer analyze /path/to/profiling -b openai -f html -o report.html

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

### Host-Device 堆栈关联分析

```bash
# 默认启用 Host-Device 关联分析（analyze 命令）
npu-analyzer analyze /path/to/profiling

# 禁用 Host-Device 关联分析
npu-analyzer analyze /path/to/profiling --no-host-device-correlation

# 对比分析自动包含堆栈关联和根因推理
npu-analyzer compare /path/to/v1 /path/to/v2 -b openai
```

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

### AIC 硬件指标分析

```bash
# 分析 AI Core 硬件指标（Cube/Vector 利用率、L2 命中率、流水线停顿）
npu-analyzer analyze-aic /path/to/profiling

# 按 Cube 利用率排序，显示 Top 30
npu-analyzer analyze-aic /path/to/profiling -n 30 -s cube_util

# 仅显示严重瓶颈算子
npu-analyzer analyze-aic /path/to/profiling --severity critical

# 输出到 CSV 文件
npu-analyzer analyze-aic /path/to/profiling -o result.csv

# 显示所有指标（含 MTE/Vector/Scalar 流水线细节）
npu-analyzer analyze-aic /path/to/profiling --show-all
```

### 融合算子集成分析

```bash
# 基于 Profiling 发现融合机会并生成集成方案
npu-analyzer integrate /path/to/profiling

# 指定融合模式和时间窗口
npu-analyzer integrate /path/to/profiling \
    --patterns add,mul,slice,strided \
    --time-window 100

# 限制分析的算子调用数
npu-analyzer integrate /path/to/profiling --limit 50 -o ./integration_output
```

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

    # 查看根因分析结果
    if report.root_cause_findings:
        for finding in report.root_cause_findings:
            print(f"[{finding.priority}] {finding.rule_name}: {finding.root_cause}")

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
┌─────────────────────────────────────────────────────────────────────────────┐
│                           NPU MFU Analyzer                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│  CLI / Web Interface                                                         │
│  analyze│compare│generate│integrate│analyze-aic│info│summary│web            │
├──────────────────────────────────┬──────────────────────────────────────────┤
│  Multi-Agent Orchestrator        │  Comparison Orchestrator                 │
│  ┌────────────┬────────────┐    │  ┌────────────┬────────────────┐         │
│  │ Timeline   │ Operator   │    │  │ Similarity │ ProfilingDiff  │         │
│  │ Memory     │ Comm       │    │  │  Checker   │   Engine       │         │
│  │ Jitter     │ Advisor    │    │  └────────────┴────────────────┘         │
│  │ DetailedOp │ AICMicro   │    │                                          │
│  │ Cluster    │ AIKG       │    │  Root Cause Skill Engine                 │
│  └────────────┴────────────┘    │  • 单版本性能分析                        │
│                                  │  • 对比根因推理                          │
│  Host-Device Correlator          │  • 5+ 规则引擎                          │
│  • 堆栈解析/模式识别             │                                          │
│  • 算子来源分类                  │                                          │
├──────────────────────────────────┴──────────────────────────────────────────┤
│  Analyzers (15+)                                                              │
│  Overlap│SlowRank│Bubble│MFU│CommSplitter│CommMatrix│LinkDashboard          │
│  Topology│HCCS Ring│Collective│Roofline│What-if│RootCause                  │
├──────────────────────────────────────────────────────────────────────────────┤
│  Data Layer                                                                   │
│  ProfilingLoader│DataSummarizer│StreamParser│DBQuery│DataValidator          │
│  StackParser│AICMetrics│HardwareRegistry│PatternMatcher                      │
├──────────────────────────────────────────────────────────────────────────────┤
│  Infrastructure                                                               │
│  LLM (OpenAI│Claude│DeepSeek│Ollama│Mock) + ResilientLLM                     │
│  Report (Markdown│HTML│Excel│JSON) │ Dashboard (交互式 HTML)                 │
└──────────────────────────────────────────────────────────────────────────────┘
```

## 文档

| 文档 | 说明 |
|------|------|
| [安装指南](docs/installation.md) | 详细安装步骤、多系统支持、Ollama 配置 |
| [功能详解](docs/features.md) | 全部模块功能说明和代码示例 |
| [配置说明](docs/configuration.md) | 环境变量、配置文件、LLM 后端配置 |
| [架构文档](docs/architecture.md) | 系统架构设计、数据流、扩展指南 |
| [API 参考](docs/api_reference.md) | CLI 命令、Web API、Python API 完整参考 |
| [AIKG 使用](docs/AIKG_USAGE.md) | AIKG 融合算子生成工作流详细指南 |
| [故障排查](docs/troubleshooting.md) | 常见问题与解决方案 |
| [变更日志](CHANGELOG.md) | 版本变更历史 |

## 项目结构

```
npu-mfu-analyzer/
├── src/
│   ├── agents/           # Multi-Agent 系统 (13 个 Agent)
│   │   ├── orchestrator.py           # 分析编排器
│   │   ├── base_agent.py             # Agent 基类
│   │   ├── timeline_agent.py         # Timeline 分析
│   │   ├── operator_agent.py         # 算子分析 + 融合检测
│   │   ├── memory_agent.py           # 内存分析
│   │   ├── communication_agent.py    # 通信分析
│   │   ├── jitter_agent.py           # 抖动检测
│   │   ├── advisor_agent.py          # 综合建议
│   │   ├── comparison_agent.py       # 对比根因分析
│   │   ├── detailed_operator_agent.py # AIC 级算子分析 V1
│   │   ├── detailed_operator_agent_v2.py # AIC 级算子分析 V2
│   │   ├── aic_microarch_agent.py    # AIC 微架构分析
│   │   ├── cluster_agent.py          # 集群分析
│   │   ├── aikg_integration.py       # AIKG 融合算子生成
│   │   └── fusion_rules.py           # 融合规则库
│   ├── analyzers/        # 分析引擎 (15+ 个)
│   │   ├── overlap_calculator.py     # 计算/通信重叠分析
│   │   ├── slow_rank_detector.py     # 慢卡检测 (Dixon + 3σ)
│   │   ├── bubble_analyzer.py        # PP Bubble 分析
│   │   ├── mfu_calculator.py         # MFU 计算
│   │   ├── comm_splitter.py          # 通信拆分 (TP/DP/PP/CP/EP)
│   │   ├── history_comparator.py     # 历史对比
│   │   ├── similarity_checker.py     # 相似度评估
│   │   ├── profiling_diff.py         # 5 层级差异引擎
│   │   ├── comparison_orchestrator.py # 对比编排
│   │   ├── communication_matrix_analyzer.py  # 通信矩阵分析
│   │   ├── communication_matrix_visualizer.py # 通信矩阵可视化
│   │   ├── link_performance_dashboard.py # 链路性能仪表板
│   │   ├── host_device_correlator.py # Host-Device 关联分析
│   │   ├── operator_source_classifier.py # 算子来源分类
│   │   ├── root_cause_engine.py      # 根因推理引擎
│   │   └── aic/                      # AIC 微架构分析
│   │       ├── instruction_analyzer.py   # 指令级分析
│   │       ├── memory_hierarchy_analyzer.py # 内存层次分析
│   │       ├── pipeline_analyzer.py      # 流水线分析
│   │       └── microarch_report.py       # 微架构报告
│   ├── data_loader/      # 数据加载与验证
│   │   ├── profiling_loader.py       # 统一数据接口
│   │   ├── data_summarizer.py        # GB→KB 数据摘要
│   │   ├── stream_parser.py          # ijson 流式解析
│   │   ├── db_query.py               # SQLite 查询
│   │   ├── data_validator.py         # 数据质量检测与容错
│   │   ├── aic_metrics.py            # AIC 硬件指标结构
│   │   ├── stack_types.py            # 堆栈数据结构
│   │   └── stack_parser.py           # 堆栈解析器
│   ├── hardware/         # 硬件规格库
│   ├── pattern_matcher/  # 模式识别 (框架/并行/模型)
│   ├── topology/         # 拓扑分析 (含 HCCS Ring 解析)
│   ├── skills/           # 专家技能引擎 (10 Python + 4 Prompt)
│   │   └── root_cause_analysis/      # 根因推理 Skills
│   │       ├── SKILL.md              # Skill 主定义
│   │       ├── rules/*.md            # 根因推理规则
│   │       └── patterns/*.md         # 堆栈模式库
│   ├── roofline/         # Roofline + What-if 模拟
│   ├── llm/              # LLM 接口 (含 ResilientLLM)
│   ├── report/           # 报告生成 (Markdown/HTML/Excel)
│   ├── cli/              # 命令行 (8 个命令)
│   └── web/              # Web 界面 + REST API + WebSocket
├── tests/
│   ├── unit/             # 单元测试 (11 个文件, ~115 cases)
│   └── integration/      # 集成测试 (5 个文件)
├── docs/                 # 文档 (9 个)
├── examples/             # 示例脚本 (5 个)
└── config/               # 配置 (settings.yaml)
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
| 0.5.0 | Host-Device 堆栈关联分析、根因推理引擎、通信矩阵分析、链路性能仪表板、AIC 微架构深度分析 |
| 0.4.0 | Profiling 对比分析、AIC 硬件指标分析、融合算子集成、弹性 LLM |
| 0.3.0 | Skill Engine、Roofline、What-if Simulator |
| 0.2.0 | Hardware Registry、Pattern Matcher、Topology |
| 0.1.0 | 基础功能、Multi-Agent、CLI、Web 界面 |

> 详细变更记录请参考 [CHANGELOG.md](CHANGELOG.md)

## License

Apache-2.0
