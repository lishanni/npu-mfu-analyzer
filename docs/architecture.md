# NPU MFU Analyzer 架构文档

## 概述

NPU MFU Analyzer 是一款用于分析和优化昇腾 NPU 上大模型训练性能的工具。采用 Multi-Agent + 专家技能的混合架构，结合 Roofline 性能建模，提供专业的性能分析和优化建议。

## 设计理念

### 1. 分层架构

```
┌─────────────────────────────────────────────────────────────────┐
│                     Presentation Layer                          │
│                   CLI / Web / Python API                        │
├─────────────────────────────────────────────────────────────────┤
│                      Agent Layer                                │
│              Multi-Agent Orchestration                          │
├─────────────────────────────────────────────────────────────────┤
│                      Skill Layer                                │
│           Python Skills + Prompt Skills                         │
├─────────────────────────────────────────────────────────────────┤
│                     Analysis Layer                              │
│        Analyzers + Topology + Roofline + What-if                │
├─────────────────────────────────────────────────────────────────┤
│                       Data Layer                                │
│     Profiling Loader + Hardware Registry + Pattern Matcher      │
├─────────────────────────────────────────────────────────────────┤
│                     Infrastructure                              │
│               LLM Backend + Report Generator                    │
└─────────────────────────────────────────────────────────────────┘
```

### 2. 对比分析架构

```
┌─────────────────────────────────────────────────────────────────┐
│                   Comparison Orchestrator                        │
│                                                                  │
│  ┌────────────┐    ┌──────────────┐    ┌──────────────────┐     │
│  │ Profiling A │    │ Profiling B  │    │  LLM Interface   │     │
│  │   Loader    │    │   Loader     │    │                  │     │
│  └──────┬──────┘    └──────┬───────┘    └────────┬─────────┘     │
│         ↓                  ↓                     │              │
│  ┌──────────────────────────────────┐            │              │
│  │      SimilarityChecker           │            │              │
│  │  硬件(35%) + 模型(30%)           │            │              │
│  │  框架(15%) + 数据形状(20%)       │            │              │
│  └──────────────┬───────────────────┘            │              │
│                 ↓                                │              │
│  ┌──────────────────────────────────┐            │              │
│  │      ProfilingDiffEngine         │            │              │
│  │  L1: Summary    L2: Timeline     │            │              │
│  │  L3: Operator   L4: Comm         │            │              │
│  │  L5: Memory                      │            │              │
│  └──────────────┬───────────────────┘            │              │
│                 ↓                                ↓              │
│  ┌──────────────────────────────────────────────────────┐       │
│  │        ComparisonAdvisorAgent                        │       │
│  │  • LLM 根因分析 / 规则引擎降级                      │       │
│  └──────────────────────────┬───────────────────────────┘       │
│                             ↓                                   │
│  ┌──────────────────────────────────────────────────────┐       │
│  │         Comparison Report (MD / HTML)                 │       │
│  └──────────────────────────────────────────────────────┘       │
└─────────────────────────────────────────────────────────────────┘
```

### 3. 核心设计原则

| 原则 | 说明 |
|------|------|
| **精确 + 智能** | Python 技能精确计算，LLM 智能推理，各司其职 |
| **硬件感知** | 所有分析基于真实硬件规格，避免泛化建议 |
| **数据驱动** | 每个结论必须有数据支撑，可验证可复现 |
| **模块化** | 独立模块可单独使用，便于扩展和维护 |

## 模块详解

### 1. Data Layer（数据层）

#### ProfilingLoader

负责加载 msprof 生成的 profiling 数据：

```python
class ProfilingLoader:
    def __init__(self, profiling_path: Path):
        self.path = profiling_path
        self.db_path = self._find_db()
        self.json_path = self._find_json()
    
    def get_step_trace(self) -> pd.DataFrame:
        """获取 Step Trace 数据，支持 DB 和 CSV 降级"""
    
    def get_timeline_events(self) -> List[Dict]:
        """获取 Timeline 事件，使用 ijson 流式解析"""
    
    def get_operator_data(self) -> pd.DataFrame:
        """获取算子数据"""
    
    def get_communication_data(self) -> pd.DataFrame:
        """获取通信数据"""
```

**支持的数据源优先级**：
1. SQLite DB（推荐，支持索引查询）
2. JSON（使用 ijson 流式解析大文件）
3. CSV（降级方案）

#### HardwareRegistry

NPU 硬件规格数据库：

```python
@dataclass
class NPUSpec:
    chip_name: str           # Atlas A2
    variant: str             # 280T/313T/376T
    aicore_count: int        # 24
    aicore_frequency_mhz: int
    fp16_tflops: float       # 280
    bf16_tflops: float       # 280
    fp32_tflops: float       # 140
    hbm_bandwidth: float     # 1500 GB/s
    hccs_bandwidth: float    # 56 GB/s
    l2_cache_mb: float       # 192
```

**支持的硬件**：
- Atlas A2 系列：280T、313T、376T
- Atlas 300I：310P

#### PatternMatcher

跨框架模式识别：

```
输入: 算子名称、通信组名称、算子序列
  ↓
FrameworkDetector → 识别框架 (Megatron/DeepSpeed/FSDP/MindSpeed)
  ↓
ParallelDetector → 识别并行策略 (TP/PP/DP/ZeRO/FSDP/CP/EP)
  ↓
ModelDetector → 推断模型结构 (layers/hidden_size/heads)
  ↓
输出: UniversalPattern
```

### 2. Analysis Layer（分析层）

#### 核心分析器

| 分析器 | 功能 | 算法 |
|--------|------|------|
| **OverlapCalculator** | 计算/通信重叠分析 | 时间区间交集计算 |
| **SlowRankDetector** | 慢卡检测 | Dixon's Q + 三 sigma |
| **BubbleAnalyzer** | PP Bubble 分析 | Stage 间隙时间统计 |
| **CommSplitter** | 通信拆分 | 组名/算子类型匹配 |
| **MFUCalculator** | MFU 计算 | FLOPS / (Peak × Time) |

#### TopologyAnalyzer

集群物理拓扑分析：

```python
class TopologyAnalyzer:
    def build_topology(self) -> TopologyInfo:
        """构建物理拓扑"""
        # 识别节点内（HCCS）和节点间（RDMA）链路
        # 计算 rank 到物理位置的映射
    
    def analyze_bandwidth(self, comm_events) -> TopologyMetrics:
        """分析带宽利用率"""
        # 计算节点内/节点间实测带宽
        # 对比理论带宽，识别瓶颈
```

#### CollectiveProfiler

集合通信分析：

```python
class CollectiveProfiler:
    def analyze(self, ops: List[CollectiveOpStats]) -> CollectiveAnalysis:
        """分析集合操作效率"""
        # 计算算法带宽（考虑 Ring/Tree 系数）
        # 计算带宽效率
        # 识别瓶颈操作
    
    def get_optimal_algorithm(self, data_size: int, group_size: int) -> str:
        """推荐最优算法"""
        # 小数据量 → Tree
        # 大数据量 → Ring
```

#### SimilarityChecker

两次 Profiling 可比性评估：

```python
class SimilarityChecker:
    """评估两次 Profiling 数据的相似度和可比性"""
    
    def check(
        self, info_a, info_b, summary_a, summary_b,
        operators_a=None, operators_b=None
    ) -> SimilarityResult:
        """
        返回 SimilarityResult:
          - overall_score: 0.0 ~ 1.0 综合评分
          - level: COMPARABLE / PARTIALLY_COMPARABLE / NOT_COMPARABLE
          - dimensions: 各维度评分详情
          - warnings: 警告信息
        """
```

**评估维度**：

| 维度 | 权重 | 检测内容 |
|------|------|---------|
| 硬件 | 35% | 芯片型号、变体一致性 |
| 模型 | 30% | 参数量级、层数 |
| 框架 | 15% | 框架类型、并行策略 |
| 数据形状 | 20% | batch_size、seq_length、world_size |

#### ProfilingDiffEngine

5 层级差异计算引擎：

```python
class ProfilingDiffEngine:
    """多层级 Profiling 差异分析"""
    
    def compute(
        self, summary_a, summary_b,
        operators_a=None, operators_b=None,
        comm_events_a=None, comm_events_b=None,
    ) -> ProfilingDiff:
        """
        返回 ProfilingDiff:
          - summary_diff: Step 时间/MFU/占比变化
          - timeline_diff: Step 稳定性/阶段时间变化
          - operator_diff: 热点算子劣化/改善/新增/消失
          - comm_diff: 通信时间/掩盖率/模式变化
          - memory_diff: 峰值内存变化
          - overall_verdict: 综合判断
          - primary_changes: 主要变化摘要
        """
```

#### ComparisonAdvisorAgent

对比分析专家 Agent：

```python
class ComparisonAdvisorAgent(BaseAgent):
    """基于对比差异数据，使用 LLM 或规则引擎进行根因分析"""
    
    async def analyze(self, data: Dict) -> AnalysisResult:
        # data 包含: diff (ProfilingDiff), label_a, label_b, summary_a, summary_b
        # LLM 可用时: 调用 LLM 进行深度根因分析
        # LLM 不可用: 降级为规则引擎分析
```

#### RooflineModeler

性能天花板分析：

```
性能 (TFLOPS)
     ^
     │           ┌─────────────── 计算天花板 (Peak TFLOPS)
     │         ╱ │
     │       ╱   │
     │     ╱     │  
     │   ╱ 内存  │ 计算
     │ ╱  受限   │ 受限
     └───────────┼──────────────> 计算强度 (FLOP/Byte)
              脊点
```

**关键公式**：
- 脊点 = Peak TFLOPS × 1000 / HBM Bandwidth (GB/s)
- 内存天花板 = HBM Bandwidth × 计算强度 / 1000
- 实际天花板 = min(计算天花板, 内存天花板)

### 3. Agent Layer（代理层）

#### Multi-Agent 架构

```
                    ┌─────────────┐
                    │ Orchestrator│
                    └──────┬──────┘
           ┌───────────────┼───────────────┐
           ↓               ↓               ↓
    ┌──────────────┐┌──────────────┐┌──────────────┐
    │TimelineAgent ││OperatorAgent ││ MemoryAgent  │
    └──────┬───────┘└──────┬───────┘└──────┬───────┘
           │               │               │
           ↓               ↓               ↓
    ┌──────────────┐┌──────────────┐┌──────────────┐
    │ CommAgent    ││ JitterAgent  ││ AdvisorAgent │
    └──────┬───────┘└──────┬───────┘└──────┬───────┘
           │               │               │
           └───────────────┴───────────────┘
                           ↓
                    ┌─────────────┐
                    │   Report    │
                    └─────────────┘


              对比分析独立编排

        ┌──────────────────────────┐
        │  ComparisonOrchestrator  │
        └─────────────┬────────────┘
                      ↓
        ┌──────────────────────────┐
        │    SimilarityChecker     │
        └─────────────┬────────────┘
                      ↓
        ┌──────────────────────────┐
        │   ProfilingDiffEngine    │
        └─────────────┬────────────┘
                      ↓
        ┌──────────────────────────┐
        │ ComparisonAdvisorAgent   │
        └─────────────┬────────────┘
                      ↓
        ┌──────────────────────────┐
        │   Comparison Report      │
        └──────────────────────────┘
```

#### Agent 协作流程

**单次分析流程**：

1. **Orchestrator** 初始化分析任务
2. **并行执行**：Timeline/Operator/Memory/Comm/Jitter Agent
3. **汇总阶段**：AdvisorAgent 综合各 Agent 结果
4. **报告生成**：生成 Markdown/HTML/Excel 报告

**对比分析流程**：

1. **ComparisonOrchestrator** 加载两组 Profiling 数据并摘要化
2. **SimilarityChecker** 评估可比性（不可比则终止并提示）
3. **ProfilingDiffEngine** 5 层级差异计算
4. **ComparisonAdvisorAgent** 根因分析（LLM / 规则引擎）
5. **报告生成**：生成对比分析报告（Markdown / HTML）

### 4. Skill Layer（技能层）

#### 混合技能架构

```
┌─────────────────────────────────────────────────────────────┐
│                    LLM Agent                                │
│              (决策 + 推理 + 表达)                           │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────────────┐    ┌─────────────────────┐        │
│  │   Prompt Skills     │    │   Python Skills     │        │
│  │   (推理指导)        │    │   (精确计算)        │        │
│  │                     │    │                     │        │
│  │ 输入: 场景描述      │    │ 输入: 数值数据      │        │
│  │ 输出: 指导文本      │    │ 输出: 计算结果      │        │
│  │                     │    │                     │        │
│  │ 用途:              │    │ 用途:              │        │
│  │ • 分析流程指导      │    │ • MFU 计算         │        │
│  │ • 报告格式规范      │    │ • 带宽效率计算      │        │
│  │ • 优化策略建议      │    │ • 慢卡检测         │        │
│  └─────────────────────┘    └─────────────────────┘        │
└─────────────────────────────────────────────────────────────┘
```

#### Python Skills 列表

| 技能 | 分类 | 输入 | 输出 |
|------|------|------|------|
| `calculate_mfu` | Compute | model_flops, step_time, peak_tflops | MFU%, 效率等级 |
| `estimate_model_flops` | Compute | params, batch, seq_len | FLOPS 估算 |
| `check_bandwidth_efficiency` | Comm | measured_bw, theoretical_bw | 效率%, 瓶颈判断 |
| `analyze_collective_ops` | Comm | op_type, data_size, duration | 带宽效率 |
| `check_overlap_ratio` | Comm | compute_time, comm_time, overlap | 掩盖率% |
| `verify_overlap_strategy` | Opt | overlap_ratio, parallel_strategy | 策略建议 |
| `detect_compute_jitter` | Diag | durations | CV%, 异常值 |
| `detect_comm_jitter` | Diag | durations | CV%, 异常值 |
| `analyze_cross_rank_jitter` | Diag | rank_durations | 方差, 慢 rank |
| `detect_slow_rank` | Diag | rank_times | 慢卡列表 |

#### Prompt Skills 列表

| 技能 | 用途 | 模板变量 |
|------|------|---------|
| `diagnosis_flow` | 性能诊断流程 | target_mfu |
| `report_format` | 报告格式规范 | - |
| `optimization_strategy` | 优化策略库 | scenario, specific_advice |
| `expert_reasoning` | 专家推理模式 | problem, available_skills |

### 5. What-if Simulator

假设场景模拟器：

```python
class WhatIfSimulator:
    def simulate_parallel_change(self, tp, pp, dp) -> WhatIfScenario:
        """模拟并行配置变化"""
        # 考虑 TP 通信开销增加
        # 考虑 PP Bubble 时间
        # 考虑 DP AllReduce 量
    
    def simulate_hardware_upgrade(self, new_hardware) -> WhatIfScenario:
        """模拟硬件升级"""
        # 计算算力提升比
        # 计算带宽提升比
    
    def simulate_optimization(self, opt_type) -> WhatIfScenario:
        """模拟优化措施"""
        # 梯度累积：通信频率降低
        # 掩盖优化：暴露时间减少
        # 算子融合：Launch 开销降低
```

## 数据流

### 单次分析流程

```
msprof 采集
    ↓
Profiling 数据 (.db / .json / .csv)
    ↓
┌─────────────────────────────────────┐
│           ProfilingLoader           │
│  • 自动检测数据格式                  │
│  • 流式解析大文件                    │
│  • 数据摘要化                        │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│      Hardware + Pattern Detection    │
│  • 硬件规格检测/加载                 │
│  • 框架/并行策略识别                 │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│          Multi-Agent Analysis        │
│  • Timeline/Operator/Memory/Comm    │
│  • 调用 Skill Engine 执行精确计算    │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│         Roofline + What-if          │
│  • 理论天花板分析                    │
│  • 优化场景预测                      │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│            Report Generation         │
│  • Markdown / HTML / Excel          │
└─────────────────────────────────────┘
```

### 对比分析数据流

```
Profiling A + Profiling B
         ↓
┌─────────────────────────────────────┐
│     ProfilingLoader × 2             │
│  加载并摘要化两组 Profiling 数据     │
└─────────────────────────────────────┘
         ↓
┌─────────────────────────────────────┐
│       SimilarityChecker             │
│  • 硬件/模型/框架/数据形状匹配      │
│  • score >= 0.3 ? → 继续            │
│  • score < 0.3 ?  → 终止并提示      │
└─────────────────────────────────────┘
         ↓
┌─────────────────────────────────────┐
│     ProfilingDiffEngine             │
│  L1: Summary (MFU/Step时间/占比)    │
│  L2: Timeline (稳定性/阶段时间)     │
│  L3: Operator (热点变化/新增/消失)  │
│  L4: Communication (时间/掩盖率)    │
│  L5: Memory (峰值内存)             │
└─────────────────────────────────────┘
         ↓
┌─────────────────────────────────────┐
│   ComparisonAdvisorAgent            │
│  LLM 根因分析 / 规则引擎降级       │
└─────────────────────────────────────┘
         ↓
┌─────────────────────────────────────┐
│      Comparison Report              │
│  • Markdown / HTML                  │
└─────────────────────────────────────┘
```

## 扩展指南

### 添加新的 Python Skill

```python
from src.skills import BaseSkill, SkillMetadata, SkillCategory, SkillResult

class MyNewSkill(BaseSkill):
    @property
    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="my_new_skill",
            display_name="我的新技能",
            description="技能描述",
            category=SkillCategory.DIAGNOSIS,
            inputs=[...],
            outputs=[...],
        )
    
    def execute(self, **kwargs) -> SkillResult:
        # 实现计算逻辑
        return SkillResult(
            skill_name=self.metadata.name,
            success=True,
            data={...},
            suggestions=[...],
        )
```

### 添加新的硬件规格

在 `src/hardware/specs/` 目录下创建 YAML 文件：

```yaml
# new_hardware.yaml
chip_name: "New Hardware"
variants:
  - name: "Variant A"
    aicore_count: 32
    fp16_tflops: 400
    hbm_bandwidth: 2000
    ...
```

### 添加新的 Agent

```python
from src.agents.base_agent import BaseAgent, AnalysisResult

class MyNewAgent(BaseAgent):
    @property
    def name(self) -> str:
        return "MyNewAgent"
    
    async def analyze(self, data: Dict) -> AnalysisResult:
        # 实现分析逻辑
        return AnalysisResult(
            agent_name=self.name,
            success=True,
            summary="分析摘要",
            details={...},
            recommendations=[...],
        )
```

### 扩展对比分析维度

在 `SimilarityChecker` 中添加新的评估维度：

```python
# src/analyzers/similarity_checker.py
# 在 check() 方法中添加新维度
dimensions.append(SimilarityDimension(
    name="my_custom_dimension",
    score=my_score,
    weight=0.1,  # 权重
    details="维度说明",
))
```

在 `ProfilingDiffEngine` 中添加新的 diff 层级，在 `compute()` 方法中扩展即可。

## 性能考量

### 大文件处理

- **流式解析**：使用 ijson 处理 GB 级 trace_view.json
- **数据摘要**：GB 级原始数据 → KB 级统计摘要
- **内存预算**：5GB JSON 文件，内存占用 < 2GB

### 并行处理

- **Agent 并行**：多个 Agent 并行分析
- **Skill 批量**：支持 Skill 链式执行
- **异步 I/O**：Web 服务使用异步处理

## 测试策略

### 测试分层

```
tests/
├── unit/           # 单元测试
│   ├── test_analyzers.py
│   ├── test_agents.py
│   ├── test_comparison.py       # 对比分析测试 (19 cases)
│   └── test_data_loader.py
└── integration/    # 集成测试
    ├── test_pattern_recognition.py
    ├── test_topology_and_jitter.py
    ├── test_skill_engine.py
    └── test_roofline_whatif.py
```

### 运行测试

```bash
# 全部测试
pytest tests/ -v

# 单元测试
pytest tests/unit/ -v

# 集成测试
pytest tests/integration/ -v

# 特定模块
python tests/integration/test_skill_engine.py
```
