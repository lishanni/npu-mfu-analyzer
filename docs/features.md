# 功能详解

## 1. Multi-Agent 智能分析

基于 Multi-Agent 架构，11 个专业 Agent 协同工作：

### 核心分析 Agent

| Agent | 职责 | 关键指标 |
|-------|------|---------|
| **TimelineAgent** | Timeline 事件分析 | Computing/Communication/Free 时间分布 |
| **OperatorAgent** | 算子性能分析 | 热点算子、执行效率、融合机会检测 |
| **MemoryAgent** | 内存使用分析 | 峰值内存、碎片率、OOM 风险、泄漏检测 |
| **CommunicationAgent** | 通信性能分析 | 带宽利用率、TP/DP/PP 拆分、集合操作效率 |
| **JitterAgent** | 抖动检测 | 计算/通信/对齐抖动、跨 Rank 方差、慢卡识别 |
| **AdvisorAgent** | 综合建议 | 多 Agent 结果汇总、优先级建议、优化规则库 |
| **DetailedOperatorAgent** | AIC 级算子分析 | Cube/Vector 利用率、L2 命中率、流水线停顿 |
| **ComparisonAdvisorAgent** | 对比根因分析 | 两次 Profiling 差异根因、LLM / 规则引擎分析 |

### JitterAgent 详细说明

JitterAgent 检测因网络波动、CPU 调度、内存争用等导致的性能抖动，支持 4 种抖动类型：

| 抖动类型 | 检测方法 | 关键阈值 |
|---------|---------|---------|
| **计算抖动** | 各 Step 计算时间的 CV（变异系数） | CV > 10% 为异常 |
| **通信抖动** | 各 Step 通信时间的 CV | CV > 15% 为异常 |
| **对齐抖动** | 跨 Rank 的同步偏差（skew） | skew > 5ms 为异常 |
| **内存抖动** | 内存分配/释放时间波动 | 标准差显著偏离 |

```python
from src.agents.jitter_agent import JitterAgent, JitterDetector

# 使用 JitterDetector 进行离线检测
detector = JitterDetector()
metrics = detector.analyze(step_times, comm_times, rank_times)

print(f"计算抖动 CV: {metrics.compute_jitter_cv:.1%}")
print(f"通信抖动 CV: {metrics.comm_jitter_cv:.1%}")
print(f"慢 Rank: {metrics.slow_ranks}")
print(f"根因: {metrics.root_causes}")
```

### DetailedOperatorAgent 详细说明

利用 AIC Metrics 硬件指标对算子进行微架构级瓶颈分析：

```python
from src.agents.detailed_operator_agent import DetailedOperatorAgent

agent = DetailedOperatorAgent(llm=llm)
result = await agent.analyze({
    "aic_analysis": aic_analysis_result,
    "summary": profiling_summary,
})
# result 包含：瓶颈类型（计算/访存/流水线/混合）、AIKG 优化建议
```

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

### HCCS Ring 拓扑解析

HCCS (High-speed Chip-to-Chip Scalability) 是昇腾 NPU 的片间互联技术（类似 NVLink）。在 8 卡机器中通常有多个 HCCS Ring，跨 Ring 通信效率会降低。

```python
from src.topology import HCCSTopologyParser

parser = HCCSTopologyParser(npus_per_machine=8)
topology = parser.parse()

for ring in topology.rings:
    print(f"Ring {ring.ring_id}: NPU {ring.device_ids}, BW={ring.bandwidth_gbps} GB/s")

# 分析跨 Ring 通信开销
analysis = parser.analyze_communication(comm_events)
print(f"Ring 内通信占比: {analysis.intra_ring_ratio:.1%}")
print(f"跨 Ring 通信占比: {analysis.cross_ring_ratio:.1%}")
```

**典型拓扑示例**：
```
Atlas 800 (8x 910B):
  Ring 0: NPU 0 ─ 1 ─ 2 ─ 3  (HCCS 56 GB/s)
  Ring 1: NPU 4 ─ 5 ─ 6 ─ 7  (HCCS 56 GB/s)
  Ring 0 ←→ Ring 1: PCIe/Host (~28 GB/s，约 1/2 带宽)
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

## 9. AIC 硬件指标分析

基于 AI Core 微架构级硬件计数器的深度算子瓶颈分析（通过 `msprof op --aic-metrics` 采集）。

### 指标体系

| 指标类别 | 包含指标 | 用途 |
|---------|---------|------|
| **算术利用率** | Cube/Vector/Scalar 利用率、总周期数 | 计算资源瓶颈诊断 |
| **内存访问** | L2 命中率、L2 读写带宽、UB/L0 使用率 | 访存瓶颈诊断 |
| **流水线** | 流水线利用率、停顿率、资源冲突率 | 流水线瓶颈诊断 |

### 瓶颈分类

| 瓶颈类型 | 判断标准 | 优化方向 |
|---------|---------|---------|
| **计算瓶颈** | Cube 利用率高、访存效率正常 | 算子融合减少 Launch 开销 |
| **访存瓶颈** | L2 命中率低、带宽利用率高 | Tiling 优化、数据布局调整 |
| **流水线瓶颈** | 停顿率高、资源冲突率高 | 指令调度优化 |
| **混合瓶颈** | 多指标同时异常 | 综合优化策略 |

### 严重度分级

| 等级 | 说明 |
|------|------|
| `critical` | Cube 利用率 < 10% 或停顿率 > 50% |
| `high` | Cube 利用率 < 30% 或 L2 命中率 < 50% |
| `medium` | Cube 利用率 < 60% |
| `low` | 存在优化空间但不紧急 |

### CLI 使用

```bash
# 基本分析
npu-analyzer analyze-aic /path/to/profiling

# 按 Cube 利用率排序，显示 Top 30
npu-analyzer analyze-aic /path/to/profiling -n 30 -s cube_util

# 仅显示严重瓶颈
npu-analyzer analyze-aic /path/to/profiling --severity critical

# 输出到 CSV
npu-analyzer analyze-aic /path/to/profiling -o bottleneck.csv

# 显示所有指标
npu-analyzer analyze-aic /path/to/profiling --show-all
```

### Python API

```python
from src.data_loader import AICMetrics, AICAnalysisResult

# AICMetrics 包含单个算子的硬件指标
metrics = AICMetrics(
    op_name="MatMulV2",
    duration_us=150.0,
    arithmetic=ArithmeticUtilization(cube_utilization=85.0, vector_utilization=12.0),
    memory=MemoryMetrics(l2_cache_hit_rate=92.0),
    pipeline=PipelineMetrics(pipe_utilization=78.0, stall_rate=8.0),
)

# 获取瓶颈摘要
summary = metrics.get_bottleneck_summary()
print(f"瓶颈类型: {summary['bottleneck_type']}")
print(f"严重度: {summary['severity']}")
```

## 10. 融合算子集成工作流

基于 Profiling 数据自动发现融合机会并生成可直接集成的算子代码和补丁。

### 工作流程

```
Profiling DB 数据
        ↓
┌──────────────────────┐
│  API 调用栈分析       │
│  定位源代码位置       │
└──────────┬───────────┘
        ↓
┌──────────────────────┐
│  时间窗口融合发现     │
│  识别可融合的连续算子  │
└──────────┬───────────┘
        ↓
┌──────────────────────┐
│  LLM 生成融合代码     │
│  Triton-Ascend Kernel │
└──────────┬───────────┘
        ↓
┌──────────────────────┐
│  集成方案生成         │
│  补丁文件 + 集成指南   │
└──────────────────────┘
```

### 支持的融合模式

| 模式 | 说明 | 示例 |
|------|------|------|
| `add` | 加法融合 | `BiasAdd + Activation → FusedBiasAct` |
| `mul` | 乘法融合 | `Scale + Dropout → FusedScaleDropout` |
| `slice` | 切片融合 | `Slice + Concat → FusedSliceConcat` |
| `strided` | 跨步访问融合 | `Gather + Transform → FusedGatherTransform` |

### CLI 使用

```bash
# 基本集成分析
npu-analyzer integrate /path/to/profiling

# 指定融合模式和参数
npu-analyzer integrate /path/to/profiling \
    --patterns add,mul,slice,strided \
    --time-window 100 \
    --limit 50 \
    -o ./integration_output
```

### 输出文件

```
integration_output/
├── analysis_report.md     # 融合分析报告
├── fusion_opportunities/  # 融合机会详情
├── generated_kernels/     # 生成的算子代码
└── integration_patches/   # 集成补丁
```

## 11. 数据验证与容错

自动检测 Profiling 数据质量问题并提供容错处理能力。

### 数据质量等级

| 等级 | 说明 | 影响 |
|------|------|------|
| `excellent` | 完整且格式正确 | 所有分析正常运行 |
| `good` | 轻微问题 | 不影响分析结果 |
| `fair` | 中等问题 | 部分分析受限 |
| `poor` | 严重问题 | 大部分分析受限 |
| `critical` | 数据无法使用 | 无法进行分析 |

### 问题类型检测

| 问题类型 | 说明 | 自动修复 |
|---------|------|---------|
| `missing_field` | 缺少必要字段 | 填充默认值 |
| `invalid_type` | 类型错误（如 Decimal） | 自动类型转换 |
| `invalid_value` | 值超出范围 | 截断/修正 |
| `corrupted_data` | 数据损坏 | 跳过并记录 |
| `parse_error` | 解析错误 | 容错解析 |
| `inconsistent` | 数据不一致 | 告警 |

### Python API

```python
from src.data_loader.data_validator import (
    ProfilingDataValidator, ProfilingDataSanitizer, RobustTimelineParser
)

# 数据质量检测
validator = ProfilingDataValidator()
report = validator.validate(profiling_path)
print(f"质量等级: {report.quality_level.value}")
print(f"问题数: {report.total_issues} (Error: {report.error_count}, Warning: {report.warning_count})")

# 数据修复
sanitizer = ProfilingDataSanitizer()
clean_events = sanitizer.sanitize(raw_events)

# 容错 Timeline 解析
parser = RobustTimelineParser()
events = parser.parse(timeline_path)  # 遇到错误不中断，记录并跳过
```

## 12. 弹性 LLM 接口

提供重试、降级、超时处理等容错机制，确保 LLM 调用的可靠性。

### 容错机制

| 机制 | 说明 | 配置 |
|------|------|------|
| **自动重试** | 指数退避重试 | max_retries=3, base_delay=1s, max_delay=30s |
| **超时控制** | 单次请求 + 总超时 | request_timeout=120s, total_timeout=300s |
| **后端降级** | 主后端失败时自动切换 | fallback_backends=["ollama", "mock"] |
| **连接池** | 多后端连接池管理 | LLMPool 并行调度 |
| **错误统计** | 调用成功率/延迟监控 | 自动记录统计信息 |

### Python API

```python
from src.llm import LLMConfig
from src.llm.resilient_llm import ResilientLLM, ResilientConfig, RetryConfig, FallbackConfig

# 配置弹性 LLM
config = LLMConfig(backend="openai", model="gpt-4")
resilient_config = ResilientConfig(
    retry=RetryConfig(max_retries=3, base_delay=1.0),
    fallback=FallbackConfig(
        enabled=True,
        fallback_backends=["deepseek", "ollama", "mock"],
    ),
)

llm = ResilientLLM(config, resilient_config)
response = await llm.complete(messages)
# 如果 OpenAI 失败 → 自动重试 3 次 → 降级到 DeepSeek → 降级到 Ollama → Mock
```

### LLMPool 多后端调度

```python
from src.llm.resilient_llm import LLMPool

# 创建多后端连接池
pool = LLMPool(configs=[
    LLMConfig(backend="openai"),
    LLMConfig(backend="deepseek"),
    LLMConfig(backend="ollama", model="qwen2.5:7b"),
])

# 自动选择可用后端
response = await pool.complete(messages)
```
