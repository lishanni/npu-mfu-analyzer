# API 参考

## CLI 命令

NPU MFU Analyzer 提供 9 个 CLI 命令，入口为 `npu-analyzer`。

### npu-analyzer analyze

分析 Profiling 数据，生成性能报告。

```bash
npu-analyzer analyze <PROFILING_PATH> [OPTIONS]
```

| 参数/选项 | 类型 | 默认值 | 说明 |
|-----------|------|--------|------|
| `PROFILING_PATH` | 位置参数 | 必填 | Profiling 数据目录 |
| `-o, --output` | TEXT | 终端输出 | 输出报告路径 |
| `-b, --backend` | TEXT | openai | LLM 后端 (openai/claude/ollama/deepseek/mock) |
| `-m, --model` | TEXT | 后端默认 | LLM 模型名称 |
| `-f, --format` | TEXT | markdown | 报告格式 (markdown/html/md) |
| `--comm-matrix` / `--no-comm-matrix` | FLAG | 启用 | 启用/禁用通信矩阵分析 |
| `--comm-matrix-output` | TEXT | 自动 | 通信矩阵可视化输出路径 |
| `--dashboard` / `--no-dashboard` | FLAG | 启用 | 启用/禁用链路性能仪表板 |
| `--dashboard-output` | TEXT | 自动 | 仪表板 HTML 输出路径 |
| `--aic-microarch` / `--no-aic-microarch` | FLAG | 启用 | 启用/禁用 AIC 微架构分析 |
| `--aic-report-output` | TEXT | 自动 | AIC 微架构报告输出路径 |
| `--host-device-correlation` / `--no-host-device-correlation` | FLAG | 启用 | 启用/禁用 Host-Device 关联分析 |
| `-v, --verbose` | FLAG | - | 详细输出 |

**示例**：
```bash
# 基本分析
npu-analyzer analyze /data/profiling -b ollama -f html -o report.html

# 禁用所有可选分析（快速模式）
npu-analyzer analyze /data/profiling --no-comm-matrix --no-dashboard --no-aic-microarch

# 自定义输出路径
npu-analyzer analyze /data/profiling \
    --comm-matrix-output comm_matrix.html \
    --dashboard-output dashboard.html \
    --aic-report-output aic_report.html
```

---

### npu-analyzer compare

对比两个 Profiling 数据，支持 5 层级差异分析和根因推理。

```bash
npu-analyzer compare <PATH_A> <PATH_B> [OPTIONS]
```

| 参数/选项 | 类型 | 默认值 | 说明 |
|-----------|------|--------|------|
| `PATH_A` | 位置参数 | 必填 | 基准 Profiling 路径 |
| `PATH_B` | 位置参数 | 必填 | 对照 Profiling 路径 |
| `-o, --output` | TEXT | 终端输出 | 输出报告路径 |
| `-b, --backend` | TEXT | mock | LLM 后端 |
| `-m, --model` | TEXT | 后端默认 | LLM 模型名称 |
| `-f, --format` | TEXT | markdown | 报告格式 (markdown/html/md) |
| `--label-a` | TEXT | 路径名 | 基准版本标签 |
| `--label-b` | TEXT | 路径名 | 对照版本标签 |
| `--force` | FLAG | - | 跳过相似度检查，强制对比 |
| `--host-device-correlation` / `--no-host-device-correlation` | FLAG | 启用 | 启用/禁用 Host-Device 关联和根因分析 |
| `-v, --verbose` | FLAG | - | 详细输出 |

**对比分析输出包含**：
- **对比概览**：两版本基本信息、整体判断（提升/劣化/混合/不变）
- **相似度评估**：4 维度评分（硬件/模型/框架/数据形状）
- **5 层级差异分析**：Summary/Timeline/Operator/Communication/Memory
- **根因推理**：自动识别性能变化原因（如 torch.compile 融合问题、通信瓶颈等）
- **优化建议**：按优先级排序的可操作建议

**示例**：
```bash
# 基本对比（使用 Mock 后端，快速）
npu-analyzer compare /data/v1 /data/v2

# 带 LLM 深度根因分析
npu-analyzer compare /data/v1 /data/v2 -b openai

# 带标签对比
npu-analyzer compare /data/v1 /data/v2 \
    --label-a "CANN 8.0" --label-b "CANN 8.1" \
    -o comparison_report.md

# 不同并行策略对比（跳过相似度检查）
npu-analyzer compare /data/tp4 /data/tp8 \
    --label-a "TP=4" --label-b "TP=8" --force

# 输出 HTML 格式
npu-analyzer compare /data/v1 /data/v2 -f html -o report.html
```

---

### npu-analyzer generate

分析 Profiling 数据并生成融合算子代码（AIKG 工作流）。

```bash
npu-analyzer generate <PROFILING_PATH> [OPTIONS]
```

| 参数/选项 | 类型 | 默认值 | 说明 |
|-----------|------|--------|------|
| `PROFILING_PATH` | 位置参数 | 必填 | Profiling 数据目录 |
| `-o, --output` | TEXT | ./generated_kernels | 融合算子输出目录 |
| `-b, --backend` | TEXT | claude | LLM 后端 |
| `-m, --model` | TEXT | 后端默认 | LLM 模型名称 |
| `--min-speedup` | FLOAT | 1.05 | 最小加速比阈值 |
| `--complexity` | TEXT | 高 | 最大实现复杂度 (低/中等/高) |
| `--skip-native` | FLAG | 默认跳过 | 跳过昇腾已有融合算子 |
| `--include-native` | FLAG | - | 包含昇腾已有融合算子 |
| `--timeout` | INT | 300 | 单个算子生成超时（秒） |
| `--max-concurrent` | INT | 3 | 最大并发生成数 |
| `-v, --verbose` | FLAG | - | 详细输出 |

---

### npu-analyzer integrate

分析 Profiling 数据并生成融合算子集成方案。

```bash
npu-analyzer integrate <PROFILING_PATH> [OPTIONS]
```

| 参数/选项 | 类型 | 默认值 | 说明 |
|-----------|------|--------|------|
| `PROFILING_PATH` | 位置参数 | 必填 | Profiling 数据目录 |
| `-o, --output` | TEXT | ./integration_output | 集成输出目录 |
| `--patterns` | TEXT | add,mul,slice,strided | 融合模式列表（逗号分隔） |
| `--time-window` | INT | 100 | 时间窗口（微秒） |
| `--limit` | INT | 50 | 分析的最大调用数 |
| `-v, --verbose` | FLAG | - | 详细输出 |

---

### npu-analyzer analyze-aic

分析 AIC Metrics 硬件指标数据。

```bash
npu-analyzer analyze-aic <PROFILING_PATH> [OPTIONS]
```

| 参数/选项 | 类型 | 默认值 | 说明 |
|-----------|------|--------|------|
| `PROFILING_PATH` | 位置参数 | 必填 | Profiling 数据目录 |
| `-n, --top-n` | INT | 20 | 显示 Top N 算子 |
| `-s, --sort-by` | TEXT | duration | 排序方式 (duration/cube_util/l2_hit/stall_rate/name) |
| `-a, --show-all` | FLAG | - | 显示所有指标 |
| `-o, --output` | TEXT | 终端输出 | 输出文件（CSV 或 Markdown） |
| `--severity` | TEXT | all | 严重度筛选 (all/critical/high/medium/low) |

**示例**：
```bash
npu-analyzer analyze-aic /data/profiling -n 30 -s cube_util --severity critical
```

---

### npu-analyzer info

显示 Profiling 数据信息（硬件、框架、数据规模等）。

```bash
npu-analyzer info <PROFILING_PATH>
```

---

### npu-analyzer summary

生成 Profiling 数据摘要（不调用 LLM）。

```bash
npu-analyzer summary <PROFILING_PATH> [OPTIONS]
```

| 参数/选项 | 类型 | 默认值 | 说明 |
|-----------|------|--------|------|
| `PROFILING_PATH` | 位置参数 | 必填 | Profiling 数据目录 |
| `--max-steps` | INT | 10 | 采样的 Step 数量 |

---

### npu-analyzer web

启动 Web 服务。

```bash
npu-analyzer web [OPTIONS]
```

| 参数/选项 | 类型 | 默认值 | 说明 |
|-----------|------|--------|------|
| `-h, --host` | TEXT | 0.0.0.0 | 监听地址 |
| `-p, --port` | INT | 8000 | 监听端口 |
| `--reload` | FLAG | - | 开发模式，自动重载 |

启动后访问 `http://localhost:8000` 打开 Web 界面。

---

### npu-analyzer version

显示版本信息。

```bash
npu-analyzer version
```

---

## Web REST API

Web 服务基于 FastAPI，启动后自动生成 Swagger 文档：`http://localhost:8000/docs`

### GET /health

健康检查。

**响应**：
```json
{
    "status": "healthy",
    "timestamp": "2026-02-14T10:00:00"
}
```

---

### POST /api/upload

上传 Profiling 数据文件。

**请求**：`multipart/form-data`

| 字段 | 类型 | 说明 |
|------|------|------|
| `file` | File | Profiling 数据文件 |

**响应**：
```json
{
    "upload_id": "uuid",
    "filename": "profiling.tar.gz",
    "path": "/uploads/uuid/profiling.tar.gz",
    "message": "上传成功"
}
```

---

### POST /api/analyze

启动分析任务。

**请求体**：
```json
{
    "profiling_path": "/path/to/profiling",
    "llm_backend": "openai",
    "output_format": "html"
}
```

**响应**：
```json
{
    "task_id": "uuid",
    "status": "running",
    "message": "分析任务已启动"
}
```

---

### POST /api/compare

启动对比分析任务。

**请求体**：
```json
{
    "path_a": "/path/to/profiling_v1",
    "path_b": "/path/to/profiling_v2",
    "label_a": "版本 A",
    "label_b": "版本 B",
    "llm_backend": "mock",
    "output_format": "html",
    "force": false
}
```

**响应**：
```json
{
    "task_id": "uuid",
    "status": "running",
    "message": "对比分析任务已启动"
}
```

---

### GET /api/tasks

列出所有任务。

**响应**：
```json
{
    "tasks": [
        {
            "task_id": "uuid",
            "status": "completed",
            "created_at": "2026-02-14T10:00:00",
            "type": "analyze"
        }
    ]
}
```

---

### GET /api/tasks/{task_id}

获取任务状态。

**响应**：
```json
{
    "task": {
        "task_id": "uuid",
        "status": "completed",
        "progress": 100,
        "message": "分析完成"
    },
    "report_url": "/api/reports/uuid"
}
```

---

### GET /api/reports/{task_id}

获取分析报告。

**响应**：报告文件（HTML/JSON/Markdown），以 `Content-Type` 区分格式。

---

### DELETE /api/tasks/{task_id}

删除任务。

**响应**：
```json
{
    "message": "任务已删除",
    "task_id": "uuid"
}
```

---

## WebSocket API

### WS /ws/{task_id}

实时接收任务进度更新。

**连接**：`ws://localhost:8000/ws/{task_id}`

**消息格式**：
```json
{
    "type": "progress",
    "progress": 45,
    "message": "正在分析 Timeline...",
    "agent": "TimelineAgent"
}
```

```json
{
    "type": "completed",
    "progress": 100,
    "report_url": "/api/reports/uuid"
}
```

---

## Python API 参考

### 核心编排

```python
from src.agents.orchestrator import Orchestrator, AnalysisReport
from src.llm import LLMConfig

# 单次分析
orchestrator = Orchestrator(
    profiling_path="/path/to/profiling",
    llm_config=LLMConfig(backend="ollama", model="qwen2.5:7b"),
)
report: AnalysisReport = await orchestrator.run()
```

```python
from src.analyzers.comparison_orchestrator import ComparisonOrchestrator, ComparisonReport

# 对比分析
orchestrator = ComparisonOrchestrator(
    path_a="/path/a", path_b="/path/b",
    label_a="v1", label_b="v2",
    llm_config=LLMConfig(backend="mock"),
)
report: ComparisonReport = await orchestrator.run()
```

### 数据加载

```python
from src.data_loader import ProfilingLoader
from src.data_loader.data_summarizer import DataSummarizer

loader = ProfilingLoader("/path/to/profiling")
info = loader.get_profiling_info()       # ProfilingInfo
step_trace = loader.get_step_trace()     # DataFrame
operators = loader.get_operator_data()   # DataFrame

summarizer = DataSummarizer()
summary = summarizer.summarize(loader)   # ProfilingSummary
```

### 分析引擎

```python
from src.analyzers import (
    OverlapCalculator, SlowRankDetector, BubbleAnalyzer,
    MFUCalculator, SimilarityChecker, ProfilingDiffEngine,
    CommunicationMatrixAnalyzer, HostDeviceCorrelator,
    RootCauseSkillEngine,
)

# MFU 计算
mfu_calc = MFUCalculator(chip_info)
mfu_metrics = mfu_calc.calculate(operators)

# 慢卡检测
detector = SlowRankDetector()
result = detector.detect(rank_times)

# 对比差异
diff_engine = ProfilingDiffEngine()
diff = diff_engine.compute(summary_a, summary_b)

# 通信矩阵分析
comm_analyzer = CommunicationMatrixAnalyzer(world_size=8)
matrix = comm_analyzer.analyze_from_db(db_path)

# Host-Device 关联分析
correlator = HostDeviceCorrelator()
chains = correlator.build_call_chains(trace_events)

# 根因推理
root_cause_engine = RootCauseSkillEngine()
findings = root_cause_engine.analyze_single(chains, stats)
```

### 硬件/模式

```python
from src.hardware import get_registry, detect_hardware
from src.pattern_matcher import UniversalPatternMatcher

registry = get_registry()
spec = registry.get_spec("Atlas A2", "280T")

matcher = UniversalPatternMatcher()
pattern = matcher.detect_from_loader(loader)
```

### 技能引擎

```python
from src.skills import get_engine

engine = get_engine()
result = engine.execute_skill("calculate_mfu", model_flops=2e15, step_time_ms=500, peak_tflops=280)
```

### Roofline / What-if

```python
from src.roofline import RooflineModeler, WhatIfSimulator, CurrentState

modeler = RooflineModeler(hardware_name="atlas_a2_280t")
mfu_result = modeler.estimate_theoretical_mfu(model_flops=42e12, model_memory_bytes=50e9, step_time_ms=500, num_devices=8)

state = CurrentState(hardware_name="Atlas A2 (280T)", num_devices=8, tp_size=1, pp_size=1, dp_size=8, step_time_ms=500, mfu_percent=38)
simulator = WhatIfSimulator(state)
scenario = simulator.simulate_hardware_upgrade("376T")
```

### LLM 接口

```python
from src.llm import LLMConfig, LLMFactory, LLMInterface
from src.llm.resilient_llm import ResilientLLM, ResilientConfig

# 基础用法
llm = LLMFactory.create(LLMConfig(backend="openai"))
response = await llm.complete(messages)

# 弹性用法（重试 + 降级）
llm = ResilientLLM(LLMConfig(backend="openai"), ResilientConfig())
response = await llm.complete(messages)
```

### 报告生成

```python
from src.report import ReportGenerator, ReportFormat

generator = ReportGenerator()
report_text = generator.generate(report_data, format=ReportFormat.HTML)
generator.save(report_text, "/output/report.html")
```

### Host-Device 堆栈关联分析

```python
from src.analyzers.host_device_correlator import (
    HostDeviceCorrelator,
    analyze_from_trace_file,
    build_call_chains_from_file,
)
from src.analyzers.operator_source_classifier import (
    OperatorSourceClassifier,
    classify_operators,
    discover_new_patterns,
)
from src.data_loader.stack_types import HostDeviceChain, CorrelationStats

# 从 trace 文件构建调用链
chains = build_call_chains_from_file("/path/to/trace_view.json")
print(f"总调用链数: {len(chains)}")

# 分析算子来源分布
classifier = OperatorSourceClassifier()
stats = classifier.build_stats(chains)
print(f"来源分布: {stats.by_source_type}")
print(f"Eager 算子: {stats.eager_ops}")
print(f"融合算子: {stats.fusion_ops}")
```

### 根因推理引擎

```python
from src.analyzers.root_cause_engine import (
    RootCauseSkillEngine,
    RootCauseFinding,
    analyze_root_causes,
)

# 单版本根因分析
engine = RootCauseSkillEngine()
findings = engine.analyze_single(
    chains=chains,
    stats=stats,
    profiling_summary=summary,
)

for finding in findings:
    print(f"[{finding.priority}] {finding.rule_name}")
    print(f"  根因: {finding.root_cause}")
    print(f"  证据: {finding.evidence}")
    print(f"  建议: {finding.optimization_suggestions}")

# 对比分析根因推理
findings = engine.analyze(chains_a, chains_b, diff_result)
```

### 通信矩阵分析

```python
from src.analyzers.communication_matrix_analyzer import (
    CommunicationMatrixAnalyzer,
    CommunicationMatrix,
)
from src.analyzers.communication_matrix_visualizer import CommunicationMatrixVisualizer
from src.analyzers.link_performance_dashboard import generate_dashboard

# 分析通信矩阵
analyzer = CommunicationMatrixAnalyzer(world_size=8, npus_per_machine=8)
matrix = analyzer.analyze_from_db("/path/to/profiling.db")

print(f"总通信量: {matrix.total_comm_data_mb:.2f} MB")
print(f"平均带宽: {matrix.avg_bandwidth_gbps:.2f} GB/s")
print(f"慢链路数: {len(matrix.slow_links)}")

# 生成热力图可视化
visualizer = CommunicationMatrixVisualizer(matrix)
html = visualizer.generate_html()
with open("comm_matrix.html", "w") as f:
    f.write(html)

# 生成链路性能仪表板
dashboard_html = generate_dashboard(matrix)
with open("dashboard.html", "w") as f:
    f.write(dashboard_html)
```
