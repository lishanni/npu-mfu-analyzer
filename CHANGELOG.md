# Changelog

所有重大变更均记录在此文件中。格式基于 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/)。

## [0.5.0] - 2026-03

### Added
- **Host-Device 堆栈关联分析模块**
  - `StackParser`：从 trace_view.json 解析 Python/C++ 堆栈
  - `HostDeviceCorrelator`：基于 connection_id 建立 Host-Device 调用链
  - `OperatorSourceClassifier`：算子来源分类（torch_compile/eager/fusion_op/mindspeed/torch_ascend/distributed/optimizer/unknown）
  - 支持的堆栈模式：torch.compile、融合算子、eager、mindspeed、torch-ascend、分布式通信、优化器
- **根因推理引擎**
  - `RootCauseSkillEngine`：基于规则引擎的自动根因识别
  - 支持 **analyze 命令**（单版本性能/MFU 低的原因分析）
  - 支持 **compare 命令**（对比根因推理）
  - 5 个单版本分析规则：小算子过多、混合执行模式、融合算子不足、通信占比过高、内存层次利用不佳
  - 3 个对比分析规则：torch.compile 融合问题、eager/compile 切换、通信瓶颈
  - Markdown Skills 格式规则库（`src/skills/root_cause_analysis/`）
- **通信矩阵分析**
  - `CommunicationMatrixAnalyzer`：链路级带宽利用率分析
  - `CommunicationMatrixVisualizer`：HTML 热力图可视化
  - HCCS/RDMA 传输类型识别
  - 慢链路/瓶颈链路自动检测
  - CLI `--comm-matrix` 和 `--comm-matrix-output` 选项
- **链路性能仪表板**
  - `LinkPerformanceDashboard`：交互式 HTML 可视化
  - 实时指标卡片：总通信量、带宽利用率、节点内/间比例、慢链路数
  - 交互式热力图：带宽利用率、通信量，支持传输类型和状态筛选
  - 趋势图表：带宽分布直方图、利用率分布图
  - 异常链路分析：慢链路列表、瓶颈链路列表
  - CLI `--dashboard` 和 `--dashboard-output` 选项
- **AIC 微架构深度分析**
  - `InstructionAnalyzer`：Cube/Vector/Scalar 指令混合比例、利用率、发射率
  - `MemoryHierarchyAnalyzer`：L2/UB/L0 缓存访问模式、命中率、局部性评分
  - `PipelineAnalyzer`：流水线利用率、停顿率、停顿原因细分（MTE/Vector/Scalar/Dependency/Memory/Sync）
  - `PMUDataParser`：PMU 数据解析器
  - `AICMicroarchAgent`：AIC 微架构分析 Agent
  - CLI `--aic-microarch` 和 `--aic-report-output` 选项
- `DetailedOperatorAgentV2`：深度算子分析 Agent V2
- `ClusterAgent`：集群分析 Agent
- 文档 `docs/design/communication_matrix_design.md`：通信矩阵设计文档

### Changed
- `Orchestrator`：新增 `enable_host_device_correlation`、`enable_dashboard`、`enable_aic_microarch` 参数
- `AnalysisReport`：新增 `source_analysis`、`host_device_chains`、`root_cause_findings` 字段
- `ProfilingDiff`：新增 `source_changes`、`root_cause_findings` 字段
- `src/analyzers/__init__.py`：新增 Host-Device/RootCause 相关类导出
- `src/data_loader/__init__.py`：新增堆栈相关类导出

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
