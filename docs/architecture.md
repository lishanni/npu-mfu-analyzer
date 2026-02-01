# NPU MFU Analyzer 架构文档

## 概述

NPU MFU Analyzer 是一款用于分析和优化昇腾 NPU 上大模型训练性能的工具。

## 系统架构

```
┌─────────────────────────────────────────────────────────────────┐
│                      NPU MFU Analyzer                           │
├─────────────────────────────────────────────────────────────────┤
│  CLI / Web Interface                                            │
├─────────────────────────────────────────────────────────────────┤
│                    Multi-Agent Orchestrator                     │
│  ┌──────────┬──────────┬──────────┬──────────┬──────────┐      │
│  │ Timeline │ Operator │  Memory  │  Comm    │ Advisor  │      │
│  │  Agent   │  Agent   │  Agent   │  Agent   │  Agent   │      │
│  └──────────┴──────────┴──────────┴──────────┴──────────┘      │
├─────────────────────────────────────────────────────────────────┤
│                      Analysis Layer                             │
│  ┌──────────┬──────────┬──────────┬──────────┬──────────┐      │
│  │ Overlap  │  Bubble  │  MFU     │  SlowRank│  Comm    │      │
│  │Calculator│ Analyzer │Calculator│ Detector │ Splitter │      │
│  └──────────┴──────────┴──────────┴──────────┴──────────┘      │
├─────────────────────────────────────────────────────────────────┤
│                      Data Layer                                 │
│  ┌──────────┬──────────┬──────────┬──────────┬──────────┐      │
│  │Profiling │  Stream  │ Hardware │ Pattern  │ Topology │      │
│  │ Loader   │  Parser  │ Registry │ Matcher  │ Analyzer │      │
│  └──────────┴──────────┴──────────┴──────────┴──────────┘      │
├─────────────────────────────────────────────────────────────────┤
│                      LLM Backend                                │
│  ┌──────────┬──────────┬──────────┐                            │
│  │  Ollama  │ DeepSeek │   Mock   │                            │
│  │ Backend  │ Backend  │ Backend  │                            │
│  └──────────┴──────────┴──────────┘                            │
└─────────────────────────────────────────────────────────────────┘
```

## 模块说明

### 1. Data Layer (数据层)

- **ProfilingLoader**: 加载 msprof 生成的 profiling 数据（.db, .json）
- **StreamParser**: 使用 ijson 流式解析大型 trace_view.json
- **HardwareRegistry**: NPU 硬件规格数据库 (Atlas A2, 300I 等)
- **PatternMatcher**: 跨框架模式识别 (Megatron/DeepSpeed/FSDP)
- **TopologyAnalyzer**: 集群物理拓扑分析

### 2. Analysis Layer (分析层)

- **OverlapCalculator**: 计算/通信重叠率分析
- **BubbleAnalyzer**: PP Bubble Time 分析
- **MFUCalculator**: Model FLOPS Utilization 计算
- **SlowRankDetector**: 慢卡检测 (Dixon's Q / 三 sigma)
- **CommSplitter**: TP/DP/PP/CP/EP 通信拆分

### 3. Agent Layer (代理层)

- **TimelineAgent**: Timeline 分析 (HostFree, Device Latency)
- **OperatorAgent**: 算子性能分析
- **MemoryAgent**: 内存使用分析
- **CommunicationAgent**: 通信性能分析
- **AdvisorAgent**: 综合优化建议

### 4. LLM Backend (大模型后端)

- **OllamaBackend**: 本地 Ollama 部署
- **DeepSeekBackend**: DeepSeek API
- **MockBackend**: 测试用模拟后端

## 数据流

```
msprof 采集 → Profiling 数据 → ProfilingLoader → 各 Agent 分析 → AdvisorAgent 汇总 → 报告生成
```

## 支持的硬件

| 型号 | 算力 | HBM 带宽 | HCCS 带宽 |
|------|------|---------|----------|
| Atlas A2 (280T) | 280 TFLOPS FP16 | 1.5 TB/s | 56 GB/s |
| Atlas A2 (313T) | 313 TFLOPS FP16 | 1.8 TB/s | 56 GB/s |
| Atlas A2 (376T) | 376 TFLOPS FP16 | 2.0 TB/s | 56 GB/s |
| Atlas 300I | 22 TFLOPS FP16 | 68 GB/s | - |
