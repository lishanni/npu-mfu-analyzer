# NPU MFU Analyzer

**昇腾 NPU 大模型训练 MFU 智能分析与优化工具**

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)

## 概述

NPU MFU Analyzer 是一款专为昇腾 NPU 设计的大模型训练性能分析工具。基于 Multi-Agent 架构和专家技能引擎，提供从数据解析到性能优化建议的端到端分析能力。

### 核心特性

| 模块 | 功能 | 说明 |
|------|------|------|
| **Multi-Agent 分析** | 5 大专业 Agent | Timeline/Operator/Memory/Communication/Advisor |
| **硬件感知** | Hardware Registry | Atlas A2 (280T/313T/376T)、Atlas 300I 规格库 |
| **模式识别** | Universal Pattern Matcher | 自动识别 Megatron/DeepSpeed/FSDP/MindSpeed |
| **拓扑分析** | Topology Analyzer | 多机多卡物理拓扑、HCCS/RDMA 带宽分析 |
| **专家技能** | Skill Engine | 14 个内置技能（Python 精确计算 + Prompt 推理指导） |
| **性能建模** | Roofline Model | 计算/内存天花板分析、理论 MFU 上限 |
| **假设分析** | What-if Simulator | 并行配置/硬件升级/优化措施效果预测 |

## 系统架构

```
┌─────────────────────────────────────────────────────────────────────┐
│                         NPU MFU Analyzer                            │
├─────────────────────────────────────────────────────────────────────┤
│  CLI / Web Interface                                                │
├─────────────────────────────────────────────────────────────────────┤
│                    Multi-Agent Orchestrator                         │
│  ┌──────────┬──────────┬──────────┬──────────┬──────────┐          │
│  │ Timeline │ Operator │  Memory  │  Comm    │  Jitter  │          │
│  │  Agent   │  Agent   │  Agent   │  Agent   │  Agent   │          │
│  └──────────┴──────────┴──────────┴──────────┴──────────┘          │
├─────────────────────────────────────────────────────────────────────┤
│                       Skill Engine                                  │
│  ┌─────────────────────────┐  ┌─────────────────────────┐          │
│  │   Python Skills (10)    │  │   Prompt Skills (4)     │          │
│  │ • calculate_mfu         │  │ • diagnosis_flow        │          │
│  │ • check_bandwidth       │  │ • report_format         │          │
│  │ • detect_slow_rank      │  │ • optimization_strategy │          │
│  │ • check_overlap_ratio   │  │ • expert_reasoning      │          │
│  └─────────────────────────┘  └─────────────────────────┘          │
├─────────────────────────────────────────────────────────────────────┤
│                      Analysis Layer                                 │
│  ┌──────────┬──────────┬──────────┬──────────┬──────────┐          │
│  │ Overlap  │  Bubble  │  MFU     │ SlowRank │  Comm    │          │
│  │Calculator│ Analyzer │Calculator│ Detector │ Splitter │          │
│  └──────────┴──────────┴──────────┴──────────┴──────────┘          │
│  ┌──────────┬──────────┬──────────┐                                │
│  │ Topology │Collective│ Roofline │                                │
│  │ Analyzer │ Profiler │ Modeler  │                                │
│  └──────────┴──────────┴──────────┘                                │
├─────────────────────────────────────────────────────────────────────┤
│                       Data Layer                                    │
│  ┌──────────┬──────────┬──────────┬──────────┐                     │
│  │Profiling │ Hardware │ Pattern  │  What-if │                     │
│  │ Loader   │ Registry │ Matcher  │Simulator │                     │
│  └──────────┴──────────┴──────────┴──────────┘                     │
├─────────────────────────────────────────────────────────────────────┤
│                      LLM Backend                                    │
│  ┌──────────┬──────────┬──────────┬──────────┐                     │
│  │  Ollama  │ DeepSeek │  OpenAI  │   Mock   │                     │
│  └──────────┴──────────┴──────────┴──────────┘                     │
└─────────────────────────────────────────────────────────────────────┘
```

## 功能详解

### 1. Multi-Agent 智能分析

| Agent | 职责 | 关键指标 |
|-------|------|---------|
| **TimelineAgent** | Timeline 事件分析 | Computing/Communication/Free 时间分布 |
| **OperatorAgent** | 算子性能分析 | 热点算子、执行效率、Tiling 效率 |
| **MemoryAgent** | 内存使用分析 | 峰值内存、碎片率、OOM 风险 |
| **CommunicationAgent** | 通信性能分析 | 带宽利用率、集合操作效率 |
| **JitterAgent** | 抖动检测 | 计算/通信抖动、跨 Rank 方差、慢卡识别 |

### 2. 硬件感知与模式识别

#### Hardware Registry

自动检测或手动指定硬件规格：

```python
from src.hardware import get_registry, detect_hardware

registry = get_registry()
spec = registry.get_spec("Atlas A2", "280T")
print(f"FP16 算力: {spec.fp16_tflops} TFLOPS")
print(f"HBM 带宽: {spec.hbm_bandwidth} GB/s")
```

支持的硬件：
| 型号 | FP16 算力 | HBM 带宽 | HCCS 带宽 |
|------|----------|---------|----------|
| Atlas A2 (280T) | 280 TFLOPS | 1.5 TB/s | 56 GB/s |
| Atlas A2 (313T) | 313 TFLOPS | 1.8 TB/s | 56 GB/s |
| Atlas A2 (376T) | 376 TFLOPS | 2.0 TB/s | 56 GB/s |
| Atlas 300I (310P) | 22 TFLOPS | 68 GB/s | - |

#### Universal Pattern Matcher

自动识别训练框架和并行策略：

```python
from src.pattern_matcher import UniversalPatternMatcher

matcher = UniversalPatternMatcher()
pattern = matcher.detect_from_loader(loader)

print(f"框架: {pattern.framework.framework.value}")  # megatron/deepspeed/fsdp
print(f"并行: TP={pattern.parallel.tp_size}, PP={pattern.parallel.pp_size}")
print(f"模型: {pattern.model.num_layers} layers, hidden={pattern.model.hidden_size}")
```

### 3. 集群拓扑与通信诊断

#### Topology Analyzer

```python
from src.topology import TopologyAnalyzer

analyzer = TopologyAnalyzer(world_size=16, npus_per_machine=8)
topology = analyzer.build_topology()

print(f"机器数: {topology.num_machines}")
print(f"节点内带宽利用率: {metrics.intra_node_utilization:.1%}")
print(f"节点间带宽利用率: {metrics.inter_node_utilization:.1%}")
```

#### Collective Profiler

```python
from src.topology import CollectiveProfiler

profiler = CollectiveProfiler(theoretical_bandwidth=56.0)
analysis = profiler.analyze(collective_ops)

print(f"AllReduce 效率: {analysis.efficiency:.1%}")
print(f"推荐算法: {profiler.get_optimal_algorithm(data_size)}")
```

### 4. 专家技能引擎

#### Python Skills（精确计算）

| 技能 | 功能 | 输出 |
|------|------|------|
| `calculate_mfu` | MFU 计算 | MFU 百分比、效率等级 |
| `check_bandwidth_efficiency` | 带宽效率 | 效率百分比、瓶颈判断 |
| `check_overlap_ratio` | 通信掩盖率 | 掩盖率、潜在加速比 |
| `detect_slow_rank` | 慢卡检测 | 慢卡列表、偏差分析 |
| `detect_compute_jitter` | 计算抖动 | CV 值、异常值数量 |
| `analyze_collective_ops` | 集合操作分析 | 带宽效率、算法推荐 |

```python
from src.skills import get_engine

engine = get_engine()
result = engine.execute_skill(
    "calculate_mfu",
    model_flops=2e15,
    step_time_ms=500,
    peak_tflops=280,
    num_gpus=8,
)
print(result.to_prompt_text())
```

#### Prompt Skills（推理指导）

| 技能 | 用途 |
|------|------|
| `diagnosis_flow` | 标准化性能诊断流程 |
| `report_format` | 专业报告格式规范 |
| `optimization_strategy` | 昇腾 NPU 优化策略库 |
| `expert_reasoning` | 专家级推理分析框架 |

### 5. Roofline 性能建模

```python
from src.roofline import RooflineModeler, PrecisionType

modeler = RooflineModeler(hardware_name="atlas_a2_280t")

# 计算脊点
ridge = modeler.get_ridge_point(PrecisionType.FP16)
print(f"脊点: {ridge:.1f} FLOP/Byte")

# 估算理论 MFU 上限
result = modeler.estimate_theoretical_mfu(
    model_flops=42e12,
    model_memory_bytes=50e9,
    step_time_ms=500,
    num_devices=8,
)
print(f"理论最大 MFU: {result['theoretical_max_mfu_percent']:.1f}%")
print(f"实际 MFU: {result['actual_mfu_percent']:.1f}%")
```

### 6. What-if 假设分析

```python
from src.roofline import WhatIfSimulator, CurrentState

state = CurrentState(
    hardware_name="Atlas A2 (280T)",
    num_devices=8,
    tp_size=1, pp_size=1, dp_size=8,
    step_time_ms=500,
    mfu_percent=38,
)

simulator = WhatIfSimulator(state)

# 模拟硬件升级
scenario = simulator.simulate_hardware_upgrade("376T")
print(f"预测加速: {scenario.predicted_speedup:.2f}x")

# 运行所有场景
result = simulator.run_all_scenarios()
print(f"推荐方案: {result.best_scenario.name}")
```

## 快速开始

### 安装

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

### 命令行使用

```bash
# 查看 Profiling 信息
npu-analyzer info /path/to/profiling

# 生成数据摘要
npu-analyzer summary /path/to/profiling

# 完整分析（使用 Mock LLM）
npu-analyzer analyze /path/to/profiling --backend mock

# 完整分析（使用 Ollama）
npu-analyzer analyze /path/to/profiling --backend ollama --model qwen2.5:7b

# 指定输出格式
npu-analyzer analyze /path/to/profiling -o report.html --format html
```

### Web 界面

```bash
# 启动服务
npu-analyzer web --port 8000

# 访问
# Web 界面: http://localhost:8000
# API 文档: http://localhost:8000/docs
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
    
    if report.success:
        print(report.final_report)
        for rec in report.recommendations:
            print(f"- {rec}")

asyncio.run(analyze())
```

## 支持的数据格式

| 格式 | 文件 | 说明 |
|------|------|------|
| **DB（推荐）** | `ascend_pytorch_profiler_*.db` | 支持索引查询，高效 |
| **JSON** | `trace_view.json` | 使用 ijson 流式解析，支持 GB 级文件 |
| **CSV** | `step_trace_time.csv` | 降级方案，当 DB 无数据时使用 |

## 配置

### 环境变量

| 变量 | 说明 |
|------|------|
| `OLLAMA_HOST` | Ollama 服务地址（默认 http://localhost:11434） |
| `DEEPSEEK_API_KEY` | DeepSeek API Key |
| `OPENAI_API_KEY` | OpenAI API Key |

### 配置文件

`config/settings.yaml`:

```yaml
llm:
  backend: ollama
  ollama:
    model: qwen2.5:7b
    host: http://localhost:11434

hardware:
  auto_detect: true
  default: atlas_a2_280t

analysis:
  overlap:
    enabled: true
  slow_rank:
    enabled: true
    method: three_sigma
  roofline:
    enabled: true
```

## 项目结构

```
npu-mfu-analyzer/
├── src/
│   ├── agents/              # Multi-Agent 系统
│   │   ├── base_agent.py
│   │   ├── orchestrator.py
│   │   ├── timeline_agent.py
│   │   ├── operator_agent.py
│   │   ├── memory_agent.py
│   │   ├── communication_agent.py
│   │   ├── advisor_agent.py
│   │   └── jitter_agent.py
│   ├── analyzers/           # 分析算法
│   │   ├── overlap_calculator.py
│   │   ├── slow_rank_detector.py
│   │   ├── bubble_analyzer.py
│   │   ├── comm_splitter.py
│   │   ├── mfu_calculator.py
│   │   └── history_comparator.py
│   ├── data_loader/         # 数据加载
│   │   ├── profiling_loader.py
│   │   ├── stream_parser.py
│   │   └── data_summarizer.py
│   ├── hardware/            # 硬件规格
│   │   ├── registry.py
│   │   └── specs/
│   ├── pattern_matcher/     # 模式识别
│   │   ├── framework_detector.py
│   │   ├── parallel_detector.py
│   │   ├── model_detector.py
│   │   └── universal_matcher.py
│   ├── topology/            # 拓扑分析
│   │   ├── topology_analyzer.py
│   │   └── collective_profiler.py
│   ├── skills/              # 专家技能
│   │   ├── engine.py
│   │   ├── registry.py
│   │   ├── python_skills/
│   │   └── prompts/
│   ├── roofline/            # 性能建模
│   │   ├── roofline_model.py
│   │   └── whatif_simulator.py
│   ├── llm/                 # LLM 接口
│   │   ├── llm_interface.py
│   │   └── resilient_llm.py
│   ├── report/              # 报告生成
│   │   ├── report_generator.py
│   │   ├── templates.py
│   │   └── excel_exporter.py
│   ├── cli/                 # 命令行
│   └── web/                 # Web 界面
├── tests/
│   ├── unit/                # 单元测试
│   └── integration/         # 集成测试
├── docs/                    # 文档
├── examples/                # 示例
├── scripts/                 # 脚本
└── config/                  # 配置
```

## 开发

### 运行测试

```bash
# 安装开发依赖
pip install -e ".[dev]"

# 运行所有测试
pytest tests/ -v

# 运行单元测试
pytest tests/unit/ -v

# 运行集成测试
pytest tests/integration/ -v

# 运行特定测试
python tests/integration/test_skill_engine.py
python tests/integration/test_roofline_whatif.py
```

### 代码检查

```bash
ruff check src/
mypy src/
```

## 版本历史

| 版本 | 日期 | 更新内容 |
|------|------|---------|
| 0.3.0 | 2026-01 | Phase 8-9: Skill Engine、Roofline、What-if Simulator |
| 0.2.0 | 2026-01 | Phase 6-7: Hardware Registry、Pattern Matcher、Topology |
| 0.1.0 | 2026-01 | Phase 1-5: 基础功能、Multi-Agent、Web 界面 |

## 路线图

- [ ] 可视化 Roofline 图表
- [ ] 自动化性能回归测试
- [ ] 与 MindStudio 深度集成
- [ ] 支持更多硬件平台

## 贡献

欢迎提交 Issue 和 Pull Request！

## License

Apache-2.0
