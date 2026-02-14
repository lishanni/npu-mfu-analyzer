# 功能详解

## 1. Multi-Agent 智能分析

基于 Multi-Agent 架构，5 个专业 Agent 协同工作：

| Agent | 职责 | 关键指标 |
|-------|------|---------|
| **TimelineAgent** | Timeline 事件分析 | Computing/Communication/Free 时间分布 |
| **OperatorAgent** | 算子性能分析 | 热点算子、执行效率、Tiling 效率 |
| **MemoryAgent** | 内存使用分析 | 峰值内存、碎片率、OOM 风险 |
| **CommunicationAgent** | 通信性能分析 | 带宽利用率、集合操作效率 |
| **JitterAgent** | 抖动检测 | 计算/通信抖动、跨 Rank 方差、慢卡识别 |

## 2. Profiling 对比分析 (Comparison)

支持两个 Profiling 数据的深度对比分析，适用于以下场景：
- **软件版本升级前后对比**：如 CANN 版本升级后性能变化分析
- **并行策略调整对比**：如 TP=4 vs TP=8 的性能差异
- **参数调优前后对比**：如调整 micro_batch_size 后的效果验证
- **硬件迁移对比**：如从 910A 迁移到 910B 的性能变化

### 对比分析流程

```
Profiling A + Profiling B
         ↓
┌─────────────────────────────┐
│     SimilarityChecker       │
│  • 硬件匹配 (35%)           │
│  • 模型匹配 (30%)           │
│  • 框架匹配 (15%)           │
│  • 数据形状 (20%)           │
└──────────────┬──────────────┘
               ↓
     score >= 0.3 ?
      YES ↓     NO → 提示不适合对比
┌─────────────────────────────┐
│    ProfilingDiffEngine      │
│  Level 1: Summary Diff      │
│  Level 2: Timeline Diff     │
│  Level 3: Operator Diff     │
│  Level 4: Communication Diff│
│  Level 5: Memory Diff       │
└──────────────┬──────────────┘
               ↓
┌─────────────────────────────┐
│  ComparisonAdvisorAgent     │
│  • LLM 深度根因分析         │
│  • 规则引擎降级分析         │
└──────────────┬──────────────┘
               ↓
       对比分析报告 (MD/HTML)
```

### CLI 使用

```bash
# 基本对比
npu-analyzer compare /path/to/profiling_v1 /path/to/profiling_v2

# 带标签的对比
npu-analyzer compare /path/to/v1 /path/to/v2 \
    --label-a "CANN 8.0" --label-b "CANN 8.1"

# 使用 LLM 深度分析
npu-analyzer compare /path/to/v1 /path/to/v2 -b openai -o report.md

# 跳过相似度检查（如不同并行策略）
npu-analyzer compare /path/to/tp4 /path/to/tp8 --force
```

### Python API

```python
import asyncio
from src.analyzers.comparison_orchestrator import ComparisonOrchestrator
from src.llm import LLMConfig

async def compare():
    orchestrator = ComparisonOrchestrator(
        path_a="/path/to/profiling_before",
        path_b="/path/to/profiling_after",
        label_a="升级前",
        label_b="升级后",
        llm_config=LLMConfig(backend="mock"),
    )
    report = await orchestrator.run()

    if report.success:
        print(f"整体判断: {report.diff.overall_verdict}")
        for change in report.diff.primary_changes:
            print(f"  - {change}")
    elif report.error == "NOT_COMPARABLE":
        print(f"不建议对比: {report.similarity.summary}")

asyncio.run(compare())
```

### 对比报告内容

| 章节 | 内容 |
|------|------|
| **对比概览** | 两版本基本信息、整体判断（提升/劣化/混合/不变） |
| **相似度评估** | 4 维度评分表、对比可行性判断 |
| **核心指标对比** | Step 时间、计算/通信/空闲占比、掩盖率等 side-by-side |
| **算子级对比** | Top 劣化算子、Top 改善算子、新增/消失算子 |
| **Timeline 级对比** | Step 稳定性变化、各阶段时间变化 |
| **通信级对比** | 通信总时间、掩盖率、通信模式变化 |
| **深度根因分析** | LLM 生成的根因分析和优化建议 |
| **优化建议** | 按优先级排序的可操作建议 |

### Web API

```
POST /api/compare
{
    "path_a": "/path/to/profiling_before",
    "path_b": "/path/to/profiling_after",
    "label_a": "升级前",
    "label_b": "升级后",
    "llm_backend": "mock",
    "output_format": "html",
    "force": false
}
```

## 3. 硬件感知 (Hardware Registry)

自动检测或手动指定硬件规格：

```python
from src.hardware import get_registry, detect_hardware

registry = get_registry()
spec = registry.get_spec("Atlas A2", "280T")
print(f"FP16 算力: {spec.fp16_tflops} TFLOPS")
print(f"HBM 带宽: {spec.hbm_bandwidth} GB/s")
```

### 支持的硬件

| 型号 | FP16 算力 | HBM 带宽 | HCCS 带宽 |
|------|----------|---------|----------|
| Atlas A2 (280T) | 280 TFLOPS | 1.5 TB/s | 56 GB/s |
| Atlas A2 (313T) | 313 TFLOPS | 1.8 TB/s | 56 GB/s |
| Atlas A2 (376T) | 376 TFLOPS | 2.0 TB/s | 56 GB/s |
| Atlas 300I (310P) | 22 TFLOPS | 68 GB/s | - |

## 4. 模式识别 (Pattern Matcher)

自动识别训练框架和并行策略：

```python
from src.pattern_matcher import UniversalPatternMatcher

matcher = UniversalPatternMatcher()
pattern = matcher.detect_from_loader(loader)

print(f"框架: {pattern.framework.framework.value}")  # megatron/deepspeed/fsdp
print(f"并行: TP={pattern.parallel.tp_size}, PP={pattern.parallel.pp_size}")
print(f"模型: {pattern.model.num_layers} layers, hidden={pattern.model.hidden_size}")
```

### 支持的框架
- Megatron-LM
- DeepSpeed
- PyTorch FSDP
- MindSpeed
- PyTorch DDP

### 支持的并行策略
- Tensor Parallel (TP)
- Pipeline Parallel (PP)
- Data Parallel (DP)
- ZeRO Stage 1/2/3
- Sequence Parallel (CP)
- Expert Parallel (EP)

## 5. 集群拓扑分析 (Topology Analyzer)

```python
from src.topology import TopologyAnalyzer

analyzer = TopologyAnalyzer(world_size=16, npus_per_machine=8)
topology = analyzer.build_topology()

print(f"机器数: {topology.num_machines}")
metrics = analyzer.analyze_bandwidth(comm_events)
print(f"节点内带宽利用率: {metrics.intra_node_utilization:.1%}")
print(f"节点间带宽利用率: {metrics.inter_node_utilization:.1%}")
```

### Collective Profiler

```python
from src.topology import CollectiveProfiler

profiler = CollectiveProfiler(theoretical_bandwidth=56.0)
analysis = profiler.analyze(collective_ops)

print(f"AllReduce 效率: {analysis.efficiency:.1%}")
print(f"推荐算法: {profiler.get_optimal_algorithm(data_size)}")
```

## 6. 专家技能引擎 (Skill Engine)

### Python Skills (精确计算)

| 技能 | 功能 | 输出 |
|------|------|------|
| `calculate_mfu` | MFU 计算 | MFU 百分比、效率等级 |
| `estimate_model_flops` | 模型 FLOPS 估算 | 前向/反向 FLOPS |
| `check_bandwidth_efficiency` | 带宽效率 | 效率百分比、瓶颈判断 |
| `analyze_collective_ops` | 集合操作分析 | 带宽效率、算法推荐 |
| `check_overlap_ratio` | 通信掩盖率 | 掩盖率、潜在加速比 |
| `verify_overlap_strategy` | 验证掩盖策略 | 策略有效性 |
| `detect_compute_jitter` | 计算抖动检测 | CV 值、异常值 |
| `detect_comm_jitter` | 通信抖动检测 | CV 值、异常值 |
| `analyze_cross_rank_jitter` | 跨 Rank 抖动 | 方差、慢 Rank |
| `detect_slow_rank` | 慢卡检测 | 慢卡列表、偏差分析 |

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

### Prompt Skills (推理指导)

| 技能 | 用途 |
|------|------|
| `diagnosis_flow` | 标准化性能诊断流程 |
| `report_format` | 专业报告格式规范 |
| `optimization_strategy` | 昇腾 NPU 优化策略库 |
| `expert_reasoning` | 专家级推理分析框架 |

### Logic Chain (逻辑链)

```python
# 创建诊断逻辑链
chain = engine.build_chain("mfu_diagnosis") \
    .add_step("calculate_mfu", inputs={...}) \
    .add_step("check_bandwidth_efficiency", inputs={...}) \
    .add_step("check_overlap_ratio", inputs={...})

result = engine.execute_chain("mfu_diagnosis", context={...})
```

## 7. Roofline 性能建模

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
print(f"受限类型: {result['bound_description']}")
```

### Roofline 模型说明

```
性能天花板 (TFLOPS)
     ^
     │                    ┌─────────────── 计算天花板
     │                  ╱ │
     │                ╱   │
     │              ╱     │
     │            ╱       │
     │          ╱  内存   │  计算
     │        ╱   受限    │  受限
     │      ╱             │
     │    ╱               │
     │  ╱                 │
     │╱                   │
     └────────────────────┼──────────> 计算强度 (FLOP/Byte)
                       脊点
```

## 8. What-if 假设分析

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

# 模拟并行配置变化
scenario = simulator.simulate_parallel_change(tp=2, pp=1, dp=4)
print(f"预测加速: {scenario.predicted_speedup:.2f}x")

# 运行所有场景
result = simulator.run_all_scenarios()
print(f"推荐方案: {result.best_scenario.name}")
```

### 支持的场景

| 场景类型 | 示例 |
|---------|------|
| 并行配置变化 | TP/PP/DP 调整 |
| 硬件升级 | 280T → 376T |
| Batch Size 变化 | 8 → 16 |
| 优化措施 | 梯度累积、算子融合、通信掩盖优化 |
| 精度变化 | FP32 → BF16 |
