# Changelog

所有重大变更均记录在此文件中。格式基于 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/)。

## [0.4.0] - 2026-02

### Added
- **Profiling 对比分析**：支持两次 Profiling 数据的深度对比
  - `SimilarityChecker`：4 维度加权相似度评估（硬件/模型/框架/数据形状）
  - `ProfilingDiffEngine`：5 层级差异计算（Summary/Timeline/Operator/Comm/Memory）
  - `ComparisonAdvisorAgent`：LLM 根因分析 + 规则引擎降级
  - `ComparisonOrchestrator`：端到端对比编排与报告生成
  - CLI `compare` 命令和 Web `/api/compare` 接口
- **AIC 硬件指标分析**：AI Core 微架构级瓶颈诊断
  - `AICMetrics` 数据结构：Cube/Vector/Scalar 利用率、L2 命中率、流水线停顿
  - `DetailedOperatorAgent`：微架构级算子分析
  - CLI `analyze-aic` 命令：支持排序、筛选、CSV/Markdown 输出
- **融合算子集成工作流**
  - `integration_workflow.py`：API 调用栈分析、时间窗口融合发现、集成方案生成
  - CLI `integrate` 命令：支持融合模式选择、时间窗口配置
- **弹性 LLM 接口**
  - `ResilientLLM`：自动重试（指数退避）、超时控制、后端降级
  - `LLMPool`：多后端连接池管理与调度
- **数据验证与容错**
  - `ProfilingDataValidator`：数据质量等级评估
  - `ProfilingDataSanitizer`：自动修复可恢复的数据问题
  - `RobustTimelineParser`：容错 Timeline 解析
- 对比分析单元测试（19 cases）
- 对比分析 Web UI "对比分析" Tab

### Changed
- `src/agents/__init__.py`：新增 `ComparisonAdvisorAgent` 导出
- `src/analyzers/__init__.py`：新增对比分析相关类导出
- `src/llm/prompts/system_prompts.py`：新增对比分析 System Prompt
- `src/web/static/index.html`：新增对比分析 Tab 和表单
- `history_comparator.py`：新增 `compare_profiling_paths` 便捷方法

## [0.3.0] - 2026-01

### Added
- **Skill Engine 专家技能引擎**
  - 10 个 Python Skills：MFU 计算、FLOPS 估算、带宽效率、集合操作分析、通信掩盖率、抖动检测（计算/通信/跨 Rank）、慢卡检测
  - 4 个 Prompt Skills：诊断流程、报告格式、优化策略、专家推理
  - Logic Chain 逻辑链：多技能串联执行
- **Roofline 性能建模**
  - `RooflineModeler`：计算天花板/内存天花板/脊点
  - 理论 MFU 上限估算
  - 计算强度分析（Compute Bound / Memory Bound）
- **What-if 假设分析**
  - `WhatIfSimulator`：模拟并行策略变化、硬件升级、Batch Size 变化、优化措施、精度变化
  - 自动推荐最优场景
- **AIKG 融合算子生成**
  - `AIKGIntegrator`：Profiling 驱动的融合机会发现
  - `AIKGRequestConverter`：算子数据→AIKG 请求
  - `AIKGKernelClient`：LLM 生成 Triton-Ascend 代码
  - CLI `generate` 命令
- **FusionRules 融合规则库**
  - 昇腾原生融合算子目录（FlashAttention, FusedLayerNorm 等）
  - 自定义融合模式匹配（BiasAdd+Act, MatMul+Bias 等）

### Changed
- `OperatorAgent`：集成融合机会检测
- `Orchestrator`：集成 MFU/Roofline 精确计算
- 文档全面更新

## [0.2.0] - 2026-01

### Added
- **Hardware Registry 硬件感知**
  - `NPUSpec`：完整硬件规格定义
  - Atlas A2 (280T/313T/376T)、Atlas 300I (310P) 规格库
  - 自动硬件检测
- **Pattern Matcher 模式识别**
  - `FrameworkDetector`：Megatron/DeepSpeed/FSDP/MindSpeed/DDP 框架检测
  - `ParallelDetector`：TP/PP/DP/ZeRO/CP/EP 并行策略识别
  - `ModelDetector`：LLAMA/GPT/BERT/MIXTRAL 模型结构推断
  - `UniversalPatternMatcher`：统一入口
- **Topology 拓扑分析**
  - `TopologyAnalyzer`：多机多卡物理拓扑构建
  - `CollectiveProfiler`：集合通信带宽效率分析
  - `HCCSTopologyParser`：HCCS Ring 拓扑解析
  - 节点内（HCCS）/节点间（RDMA）带宽分析
- **JitterAgent 抖动检测**
  - 计算/通信/对齐/内存 4 类抖动
  - 跨 Rank 方差分析和慢卡识别
- **AdvisorAgent 综合建议**
  - 多 Agent 结果汇总
  - 优先级排序的优化建议
  - 内置优化规则库

### Changed
- `BaseAgent`：`_extract_recommendations` 方法重构
- 分析流程增加硬件感知和模式识别步骤

## [0.1.0] - 2025-12

### Added
- **Multi-Agent 分析系统**
  - `Orchestrator`：分析编排器
  - `TimelineAgent`：Timeline 事件分析
  - `OperatorAgent`：算子性能分析
  - `MemoryAgent`：内存使用分析
  - `CommunicationAgent`：通信性能分析
- **核心分析器**
  - `OverlapCalculator`：计算/通信重叠分析
  - `SlowRankDetector`：慢卡检测（Dixon's Q + 三 sigma）
  - `BubbleAnalyzer`：PP Bubble 分析
  - `CommSplitter`：TP/DP/PP 通信拆分
  - `MFUCalculator`：MFU 精确计算
- **数据加载**
  - `ProfilingLoader`：DB/JSON/CSV 多格式支持
  - `StreamParser`：ijson 流式解析 GB 级 JSON
  - `DataSummarizer`：GB→KB 数据摘要化
  - `DBQuery`：SQLite 查询封装
- **LLM 接口**
  - OpenAI、Claude、DeepSeek、Ollama、Mock 5 大后端
  - System Prompt 模板库
- **CLI 工具**
  - `analyze`、`info`、`summary`、`web`、`version` 命令
- **Web 界面**
  - FastAPI 后端 + HTML/JS 前端
  - WebSocket 实时进度推送
  - 任务管理（创建/查询/删除）
- **报告生成**
  - Markdown/HTML/Excel/JSON 多格式
  - 历史数据对比
