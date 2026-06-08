# npu-mfu-analyzer Skills 架构重构设计

## 文档信息
- **版本**: v1.0
- **日期**: 2026-03-08
- **作者**: Agent架构分析

---

## 一、现状分析

### 1.1 当前架构概览

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           当前架构                                        │
├─────────────────────────────────────────────────────────────────────────┤
│  Agents Layer (13 Agents)                                                │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐            │
│  │Timeline │ │Operator │ │ Memory  │ │  Comm   │ │ Jitter  │            │
│  └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘            │
│       │           │           │           │           │                 │
│  ┌────┴────┐ ┌────┴────┐ ┌────┴────┐                                      │
│  │Detailed │ │ AICMicro│ │ Cluster │   ← 新增 Agent，职责重叠？          │
│  └─────────┘ └─────────┘ └─────────┘                                      │
├─────────────────────────────────────────────────────────────────────────┤
│  Skills Layer                                                            │
│  ┌─────────────────────┐    ┌─────────────────────┐                     │
│  │   Python Skills     │    │   Prompt Skills     │                     │
│  │   (硬编码计算)       │    │   (LLM 指导模板)    │                     │
│  └─────────────────────┘    └─────────────────────┘                     │
│                                                                           │
│  Root Cause Skills (独立实现)                                             │
│  ┌─────────────────────────────────────────────────────┐                │
│  │  RootCauseSkillEngine (独立于 SkillEngine)           │                │
│  │  - 自定义规则加载                                     │                │
│  │  - 自定义触发逻辑                                     │                │
│  │  - Markdown 规则文件                                 │                │
│  └─────────────────────────────────────────────────────┘                │
├─────────────────────────────────────────────────────────────────────────┤
│  Analyzers Layer (15+ 分析器)                                             │
│  ┌──────────────────────────────────────────────────────────┐           │
│  │  功能重复? Analyzers vs Skills vs Agents                  │           │
│  │  - MFUCalculator (Analyzer) vs MFUSkill (Skill)          │           │
│  │  - SlowRankDetector (Analyzer) vs SlowRankSkill (Skill)  │           │
│  │  - OverlapCalculator (Analyzer) vs OverlapSkill (Skill)  │           │
│  └──────────────────────────────────────────────────────────┘           │
└─────────────────────────────────────────────────────────────────────────┘
```

### 1.2 现有问题

#### 问题 1: 三层职责重叠
```
Agent 层：调用 LLM 做推理，内部调用 Skill/Analyzer
Skill 层：Python 精确计算 + Prompt 指导
Analyzer 层：纯 Python 计算逻辑

问题：
- MFUCalculator (Analyzer) vs MFUSkill (Skill) - 功能重复
- Agent 内部直接调用 Analyzer，绕过 Skill 抽象
- 新增功能时，需要决定放哪一层，增加心智负担
```

#### 问题 2: Skills 系统不统一
```python
# 当前存在两套 Skills 系统

# 系统 1: BaseSkill + SkillEngine (src/skills/)
class MFUSkill(BaseSkill):
    def execute(self, **kwargs) -> SkillResult:
        # 计算 MFU

# 系统 2: RootCauseSkillEngine (src/analyzers/root_cause_engine.py)
class RootCauseSkillEngine:
    def analyze(self, chains_a, chains_b, diff) -> List[RootCauseFinding]:
        # 根因推理，完全独立的实现
        # 没有继承 BaseSkill
        # 没有注册到 SkillRegistry
```

#### 问题 3: 规则硬编码
```python
# root_cause_engine.py 中规则硬编码在 Python 代码中
rules.append(RootCauseRule(
    name="torch_compile_fusion_issue",
    trigger_conditions={
        "stack_pattern": ["CompiledFunctionBackward", "torch._inductor"],
        "missing_fusion": ["NPUGroupedMatmul", "FlashAttention"],
        "small_op_increase": 50,
    },
    root_cause="torch.compile 图模式未开启对应的算子融合...",
    ...
))

# 问题：
# 1. 添加新规则需要修改 Python 代码
# 2. 无法动态加载规则
# 3. 专家经验难以沉淀和版本控制
```

#### 问题 4: Agent 与 Skills 耦合
```python
# 当前 Agent 内部直接调用 LLM + Analyzer
class TimelineAgent(BaseAgent):
    async def analyze(self, data):
        # 直接调用 LLM
        response = await self.call_llm(prompt)
        # 没有使用 Skill 抽象

# 问题：
# 1. Agent 行为不可预测（完全依赖 LLM 输出）
# 2. 无法复用 Skill 的精确计算能力
# 3. 测试困难
```

#### 问题 5: 缺乏标准化接口
```python
# 不同模块的输入输出格式不统一
class AnalysisResult:  # Agent 输出
    agent_name: str
    success: bool
    summary: str
    details: Dict[str, Any]
    recommendations: List[str]

class SkillResult:  # Skill 输出
    skill_name: str
    success: bool
    data: Dict[str, Any]
    summary: str
    suggestions: List[str]

class RootCauseFinding:  # 根因引擎输出
    rule_name: str
    root_cause: str
    evidence: List[str]
    affected_operators: List[str]
    optimization_suggestions: List[str]
    priority: str
```

---

## 二、重构目标

### 2.1 设计原则

| 原则 | 描述 |
|------|------|
| **单一职责** | Agent 负责编排，Skill 负责原子能力，Analyzer 作为 Skill 底层实现 |
| **统一抽象** | 所有能力统一为 Skill，包括计算、推理、诊断 |
| **可扩展** | 新增 Skill 只需创建文件/目录，无需修改核心代码 |
| **可移植** | Skills 可独立打包，集成到其他 Agent 平台 |
| **可观测** | Skill 执行有标准化日志、指标、追踪 |

### 2.2 目标架构

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           目标架构                                        │
├─────────────────────────────────────────────────────────────────────────┤
│  Orchestrator (编排层)                                                    │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │  - 任务分解                                                      │    │
│  │  - Skill 选择与调度                                              │    │
│  │  - 结果聚合                                                      │    │
│  │  - LLM 增强推理 (可选)                                           │    │
│  └─────────────────────────────────────────────────────────────────┘    │
├─────────────────────────────────────────────────────────────────────────┤
│  Skill Registry (技能注册中心)                                             │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐   │    │
│  │  │ Compute │ │ Diagnose│ │Analysis │ │Reasoning│ │Generate │   │    │
│  │  │ Skills  │ │ Skills  │ │ Skills  │ │ Skills  │ │ Skills  │   │    │
│  │  └─────────┘ └─────────┘ └─────────┘ └─────────┘ └─────────┘   │    │
│  │                                                                  │    │
│  │  统一接口: Skill.execute(context: SkillContext) -> SkillResult   │    │
│  └─────────────────────────────────────────────────────────────────┘    │
├─────────────────────────────────────────────────────────────────────────┤
│  Skill Implementation (技能实现层)                                        │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │  Python Skill         │  Markdown Skill       │  LLM Skill      │    │
│  │  ┌─────────────────┐  │  ┌─────────────────┐  │  ┌────────────┐ │    │
│  │  │ compute_mfu()   │  │  │ root_cause.md   │  │  │ diagnose() │ │    │
│  │  │ detect_slow()   │  │  │ optimization.md │  │  │ explain()  │ │    │
│  │  │ analyze_comm()  │  │  │ pattern_xxx.md  │  │  │ suggest()  │ │    │
│  │  └─────────────────┘  │  └─────────────────┘  │  └────────────┘ │    │
│  └─────────────────────────────────────────────────────────────────┘    │
├─────────────────────────────────────────────────────────────────────────┤
│  Context Layer (上下文层)                                                 │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │  ProfilingContext                                                │    │
│  │  - profiling_data: ProfilingSummary                             │    │
│  │  - hardware_spec: NPUSpec                                        │    │
│  │  - pattern: UniversalPattern                                     │    │
│  │  - previous_results: Dict[str, SkillResult]                      │    │
│  │  - user_intent: str                                              │    │
│  └─────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 三、核心设计

### 3.1 统一 Skill 接口

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from enum import Enum

class SkillType(Enum):
    """技能类型"""
    COMPUTE = "compute"        # 精确计算（不调用 LLM）
    DIAGNOSE = "diagnose"      # 规则诊断（规则引擎）
    ANALYSIS = "analysis"      # 深度分析（可能调用 LLM）
    REASONING = "reasoning"    # 推理（完全依赖 LLM）
    GENERATE = "generate"      # 代码生成


@dataclass
class SkillContext:
    """技能执行上下文"""
    # Profiling 数据
    profiling_summary: Any  # ProfilingSummary
    hardware_spec: Any      # NPUSpec
    pattern: Any            # UniversalPattern

    # 依赖数据（其他 Skill 的输出）
    previous_results: Dict[str, "SkillResult"] = field(default_factory=dict)

    # 用户意图
    user_intent: str = ""

    # 对比分析专用
    profiling_summary_b: Optional[Any] = None
    diff_result: Optional[Any] = None

    # 执行配置
    config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SkillResult:
    """技能执行结果（统一格式）"""
    skill_name: str
    skill_type: SkillType
    success: bool

    # 结构化输出
    data: Dict[str, Any] = field(default_factory=dict)

    # 人类可读输出
    summary: str = ""
    details: List[str] = field(default_factory=list)

    # 建议和行动项
    recommendations: List[str] = field(default_factory=list)
    actions: List[Dict[str, Any]] = field(default_factory=list)  # 可执行的操作

    # 元数据
    confidence: float = 1.0
    priority: str = "P2"  # P0/P1/P2
    execution_time_ms: float = 0
    error: Optional[str] = None

    def to_markdown(self) -> str:
        """转换为 Markdown 格式"""
        ...

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典（供 JSON 序列化）"""
        ...


class BaseSkill(ABC):
    """统一 Skill 基类"""

    @property
    @abstractmethod
    def metadata(self) -> SkillMetadata:
        """技能元数据"""
        pass

    @property
    def skill_type(self) -> SkillType:
        """技能类型"""
        return self.metadata.skill_type

    @abstractmethod
    def execute(self, context: SkillContext) -> SkillResult:
        """
        执行技能

        Args:
            context: 执行上下文，包含所有必需数据

        Returns:
            SkillResult: 统一格式的结果
        """
        pass

    def can_execute(self, context: SkillContext) -> bool:
        """
        检查是否可以执行

        用于条件触发，如数据不足时跳过
        """
        return True

    def validate_inputs(self, context: SkillContext) -> Optional[str]:
        """验证输入，返回错误信息或 None"""
        return None
```

### 3.2 三种 Skill 实现模式

#### 模式 1: Python Skill（精确计算）

```python
# src/skills/compute/mfu_skill.py

class MFUSkill(BaseSkill):
    """MFU 计算技能"""

    @property
    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="compute_mfu",
            display_name="MFU 计算",
            description="计算 Model FLOPS Utilization",
            skill_type=SkillType.COMPUTE,
            category="compute",
            tags=["mfu", "performance", "core"],
            dependencies=[],  # 无依赖
        )

    def execute(self, context: SkillContext) -> SkillResult:
        # 纯 Python 计算，不调用 LLM
        from npu_mfu_analyzer.analyzers.mfu_calculator import MFUCalculator

        calculator = MFUCalculator(context.hardware_spec)
        mfu_metrics = calculator.calculate(context.profiling_summary.operators)

        return SkillResult(
            skill_name=self.metadata.name,
            skill_type=SkillType.COMPUTE,
            success=True,
            data={
                "overall_mfu": mfu_metrics.overall_mfu,
                "peak_flops": mfu_metrics.peak_flops,
                "actual_flops": mfu_metrics.actual_flops,
            },
            summary=f"MFU: {mfu_metrics.overall_mfu * 100:.1f}%",
            recommendations=self._generate_recommendations(mfu_metrics),
        )
```

#### 模式 2: Markdown Skill（规则诊断）

```markdown
# src/skills/diagnose/rules/torch_compile_fusion.md

---
name: diagnose_torch_compile_fusion
type: diagnose
category: performance
priority: P0
tags: [torch.compile, fusion, performance]
dependencies: [compute_mfu, classify_operators]
---

## 触发条件

```yaml
conditions:
  - type: operator_source_change
    from: eager
    to: torch_compile
  - type: small_operator_increase
    threshold: 50%
  - type: missing_fusion_ops
    ops: [NPUGroupedMatmul, FlashAttention]
```

## 诊断逻辑

当检测到以下情况时触发：
1. 堆栈中出现 `CompiledFunctionBackward` 但缺少融合算子
2. 小算子（<10us）数量增加超过阈值
3. MFU 下降超过 10%

## 输出

```yaml
root_cause: "torch.compile 图模式未开启算子融合"
evidence:
  - "堆栈显示 CompiledFunctionBackward"
  - "小算子增加 {small_op_increase}%"
  - "缺少融合算子: {missing_fusion_ops}"
actions:
  - type: code_suggestion
    code: |
      model = torch.compile(model, mode="reduce-overhead")
  - type: doc_link
    url: https://docs...
```
```

```python
# Markdown Skill 加载器
class MarkdownSkillLoader:
    """加载 Markdown 格式的 Skill"""

    def load(self, skill_path: Path) -> BaseSkill:
        """从 Markdown 文件加载 Skill"""
        content = skill_path.read_text()
        frontmatter, body = self._parse_frontmatter(content)

        return MarkdownSkill(
            metadata=SkillMetadata(
                name=frontmatter["name"],
                skill_type=SkillType[frontmatter["type"].upper()],
                ...
            ),
            trigger_conditions=self._parse_conditions(frontmatter.get("conditions")),
            diagnosis_logic=self._parse_logic(body),
        )
```

#### 模式 3: LLM Skill（推理分析）

```python
# src/skills/reasoning/diagnosis_skill.py

class DiagnosisSkill(BaseSkill):
    """LLM 诊断技能"""

    @property
    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="diagnose_performance",
            skill_type=SkillType.REASONING,
            dependencies=["compute_mfu", "analyze_timeline", "analyze_communication"],
        )

    def __init__(self, llm: LLMInterface):
        self.llm = llm

    def execute(self, context: SkillContext) -> SkillResult:
        # 收集依赖 Skill 的结果
        mfu_result = context.previous_results.get("compute_mfu")
        timeline_result = context.previous_results.get("analyze_timeline")

        # 构建 Prompt
        prompt = self._build_diagnosis_prompt(mfu_result, timeline_result)

        # 调用 LLM
        response = await self.llm.complete([
            Message(role="system", content=DIAGNOSIS_SYSTEM_PROMPT),
            Message(role="user", content=prompt),
        ])

        # 解析 LLM 输出
        return self._parse_llm_response(response.content)
```

### 3.3 Skill Registry 重构

```python
# src/skills/registry.py

class SkillRegistry:
    """技能注册中心"""

    def __init__(self):
        self._skills: Dict[str, BaseSkill] = {}
        self._skill_graph: Dict[str, List[str]] = {}  # 依赖图

    def register(self, skill: BaseSkill) -> None:
        """注册技能"""
        name = skill.metadata.name
        self._skills[name] = skill
        self._skill_graph[name] = skill.metadata.dependencies

    def discover(self, skills_dir: Path) -> int:
        """
        自动发现并注册技能

        扫描目录结构：
        skills/
        ├── compute/
        │   ├── mfu_skill.py
        │   └── bandwidth_skill.py
        ├── diagnose/
        │   ├── rules/
        │   │   ├── torch_compile_fusion.md
        │   │   └── communication_bottleneck.md
        │   └── diagnose_engine.py
        ├── analysis/
        │   ├── timeline_analysis.py
        │   └── operator_analysis.py
        └── reasoning/
            ├── diagnosis_skill.py
            └── explanation_skill.py
        """
        count = 0

        # 1. 发现 Python Skills
        for py_file in skills_dir.rglob("*_skill.py"):
            skill = self._load_python_skill(py_file)
            if skill:
                self.register(skill)
                count += 1

        # 2. 发现 Markdown Skills
        for md_file in skills_dir.rglob("*.md"):
            skill = self._load_markdown_skill(md_file)
            if skill:
                self.register(skill)
                count += 1

        return count

    def get_execution_order(self, skill_names: List[str]) -> List[str]:
        """
        根据依赖关系计算执行顺序（拓扑排序）
        """
        ...

    def execute_skill_chain(
        self,
        skill_names: List[str],
        context: SkillContext,
    ) -> Dict[str, SkillResult]:
        """
        执行技能链（自动处理依赖）

        Returns:
            Dict[str, SkillResult]: 所有技能的执行结果
        """
        order = self.get_execution_order(skill_names)
        results = {}

        for skill_name in order:
            skill = self._skills[skill_name]

            # 更新上下文中的前序结果
            context.previous_results = results

            # 执行技能
            result = skill.execute(context)
            results[skill_name] = result

        return results
```

### 3.4 Orchestrator 重构

```python
# src/agents/orchestrator.py

class Orchestrator:
    """分析编排器"""

    def __init__(
        self,
        profiling_path: str,
        llm_config: Optional[LLMConfig] = None,
        skill_registry: Optional[SkillRegistry] = None,
    ):
        self.profiling_path = profiling_path
        self.llm = LLMFactory.create(llm_config) if llm_config else None

        # 技能注册中心
        self.registry = skill_registry or self._create_default_registry()

        # 数据加载器
        self.loader = ProfilingLoader(profiling_path)

    def _create_default_registry(self) -> SkillRegistry:
        """创建默认技能注册中心"""
        registry = SkillRegistry()

        # 自动发现技能
        skills_dir = Path(__file__).parent.parent / "skills"
        registry.discover(skills_dir)

        return registry

    async def analyze(
        self,
        skill_names: Optional[List[str]] = None,
        use_llm_reasoning: bool = True,
    ) -> AnalysisReport:
        """
        执行分析

        Args:
            skill_names: 指定执行的技能列表，None 表示执行所有
            use_llm_reasoning: 是否使用 LLM 进行综合推理
        """
        # 1. 加载数据
        profiling_summary = self.loader.summarize()
        hardware_spec = self.loader.detect_hardware()
        pattern = self.loader.detect_pattern()

        # 2. 构建上下文
        context = SkillContext(
            profiling_summary=profiling_summary,
            hardware_spec=hardware_spec,
            pattern=pattern,
            config={"use_llm": use_llm_reasoning},
        )

        # 3. 确定要执行的技能
        if skill_names is None:
            skill_names = self._select_skills(context)

        # 4. 执行技能链
        results = self.registry.execute_skill_chain(skill_names, context)

        # 5. LLM 综合推理（可选）
        if use_llm_reasoning and self.llm:
            reasoning_result = await self._llm_reasoning(context, results)
            results["llm_reasoning"] = reasoning_result

        # 6. 生成报告
        return self._generate_report(results)

    def _select_skills(self, context: SkillContext) -> List[str]:
        """
        根据上下文自动选择要执行的技能

        规则：
        1. 始终执行核心计算技能
        2. 根据数据特征选择诊断技能
        3. 根据 Pattern 选择特定技能
        """
        selected = []

        # 核心技能（始终执行）
        selected.extend([
            "compute_mfu",
            "analyze_timeline",
            "analyze_communication",
        ])

        # 条件技能
        if context.pattern.has_aic_metrics:
            selected.append("analyze_aic_microarch")

        if context.pattern.is_distributed:
            selected.append("analyze_comm_matrix")

        if context.pattern.has_host_stack:
            selected.extend([
                "classify_operators",
                "diagnose_root_cause",
            ])

        return selected
```

---

## 四、与 Agent 平台集成

### 4.1 Claude Code 集成

```yaml
# .claude/skills/npu-analyzer/SKILL.md

---
name: npu-mfu-analyzer
version: 1.0.0
description: 昇腾 NPU 性能分析工具
---

## 功能

分析 NPU Profiling 数据，识别性能瓶颈，提供优化建议。

## 使用方式

### analyze 命令

分析单个 Profiling 数据：

```
/npu-analyzer analyze /path/to/profiling
```

### compare 命令

对比两个 Profiling 数据：

```
/npu-analyzer compare /path/to/v1 /path/to/v2
```

## 可用技能

### compute_mfu
计算 Model FLOPS Utilization

**输入**:
- `profiling_path`: Profiling 数据路径

**输出**:
- `mfu_percent`: MFU 百分比
- `bottleneck_type`: 瓶颈类型

### diagnose_root_cause
诊断性能问题根因

**输入**:
- `profiling_path`: Profiling 数据路径

**输出**:
- `findings`: 根因发现列表
- `recommendations`: 优化建议
```

```python
# .claude/skills/npu-analyzer/skill.py

from anthropic import Anthropic
from npu_mfu_analyzer import Orchestrator, SkillContext

def analyze_profiling(profiling_path: str, focus: str = None) -> dict:
    """
    分析 NPU Profiling 数据

    Args:
        profiling_path: Profiling 数据路径
        focus: 关注点（可选）: mfu, communication, memory, operator

    Returns:
        分析结果
    """
    orchestrator = Orchestrator(profiling_path)

    # 根据关注点选择技能
    skill_mapping = {
        "mfu": ["compute_mfu", "analyze_timeline"],
        "communication": ["analyze_communication", "analyze_comm_matrix"],
        "memory": ["analyze_memory"],
        "operator": ["analyze_operators", "analyze_aic_microarch"],
    }

    skill_names = skill_mapping.get(focus) if focus else None

    # 执行分析
    report = await orchestrator.analyze(skill_names=skill_names)

    return {
        "success": report.success,
        "summary": report.summary,
        "mfu": report.mfu_metrics.overall_mfu if report.mfu_metrics else None,
        "root_causes": [f.to_dict() for f in report.root_cause_findings],
        "recommendations": report.recommendations,
    }

# Claude Code 工具定义
tools = [
    {
        "name": "analyze_profiling",
        "description": "分析 NPU Profiling 数据，识别性能瓶颈",
        "input_schema": {
            "type": "object",
            "properties": {
                "profiling_path": {
                    "type": "string",
                    "description": "Profiling 数据目录路径"
                },
                "focus": {
                    "type": "string",
                    "enum": ["mfu", "communication", "memory", "operator"],
                    "description": "分析关注点"
                }
            },
            "required": ["profiling_path"]
        }
    }
]
```

### 4.2 OpenCode 集成

```yaml
# opencode/skills/npu-analyzer.yaml

apiVersion: opencode/v1
kind: Skill
metadata:
  name: npu-mfu-analyzer
  version: 1.0.0
  description: 昇腾 NPU 性能分析工具

spec:
  # 技能入口
  entrypoint: npu_mfu_analyzer.cli:main

  # 工具定义
  tools:
    - name: analyze
      description: 分析 NPU Profiling 数据
      parameters:
        profiling_path:
          type: string
          description: Profiling 数据路径
        skills:
          type: array
          items: string
          description: 要执行的技能列表

    - name: compare
      description: 对比两个 Profiling 数据
      parameters:
        path_a:
          type: string
        path_b:
          type: string

  # 技能依赖
  dependencies:
    - name: compute_mfu
      type: python
      module: npu_mfu_analyzer.skills.compute
    - name: diagnose_root_cause
      type: markdown
      path: skills/diagnose/rules/
```

### 4.3 通用 Skill SDK

```python
# npu_mfu_analyzer/sdk.py

"""
NPU MFU Analyzer Skill SDK

提供标准化的 Skill 开发接口，便于集成到各种 Agent 平台。
"""

from typing import Protocol, Dict, Any, List

class SkillExecutor(Protocol):
    """Skill 执行器协议"""

    def list_skills(self) -> List[Dict[str, Any]]:
        """列出所有可用技能"""
        ...

    def execute(self, skill_name: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """执行指定技能"""
        ...

    def execute_chain(
        self,
        skill_names: List[str],
        context: Dict[str, Any],
    ) -> Dict[str, Dict[str, Any]]:
        """执行技能链"""
        ...


class NPUMFUAnalyzerSDK:
    """NPU MFU Analyzer SDK"""

    def __init__(self, skills_dir: str = None):
        """
        Args:
            skills_dir: 自定义技能目录
        """
        from .skills.registry import SkillRegistry
        from .skills.engine import SkillEngine

        self.registry = SkillRegistry()
        if skills_dir:
            self.registry.discover(Path(skills_dir))

        self.engine = SkillEngine(self.registry)

    def list_skills(self) -> List[Dict[str, Any]]:
        """列出所有可用技能"""
        return [
            skill.metadata.to_dict()
            for skill in self.registry._skills.values()
        ]

    def execute(
        self,
        skill_name: str,
        profiling_path: str,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        执行单个技能

        Args:
            skill_name: 技能名称
            profiling_path: Profiling 数据路径
            **kwargs: 额外参数

        Returns:
            技能执行结果
        """
        # 加载数据
        from .data_loader import ProfilingLoader, DataSummarizer

        loader = ProfilingLoader(profiling_path)
        summarizer = DataSummarizer(loader)

        # 构建上下文
        context = SkillContext(
            profiling_summary=summarizer.summarize(),
            hardware_spec=loader.detect_hardware(),
            pattern=loader.detect_pattern(),
            config=kwargs,
        )

        # 执行技能
        result = self.engine.execute_skill(skill_name, context)

        return result.to_dict()

    def analyze(
        self,
        profiling_path: str,
        skills: List[str] = None,
    ) -> Dict[str, Any]:
        """
        执行分析

        Args:
            profiling_path: Profiling 数据路径
            skills: 要执行的技能列表，None 表示自动选择

        Returns:
            分析报告
        """
        ...


# 便捷导出
def analyze(profiling_path: str, **kwargs) -> Dict[str, Any]:
    """便捷分析函数"""
    sdk = NPUMFUAnalyzerSDK()
    return sdk.analyze(profiling_path, **kwargs)
```

---

## 五、迁移计划

### 5.1 Phase 1: 接口统一（1 周）

1. 定义统一 Skill 接口
2. 创建 SkillContext 和 SkillResult
3. 更新现有 Python Skills

### 5.2 Phase 2: 根因引擎集成（1 周）

1. 将 RootCauseSkillEngine 重构为 Markdown Skills
2. 实现规则动态加载
3. 集成到 Skill Registry

### 5.3 Phase 3: Agent 重构（2 周）

1. 简化 Agent 为编排器
2. Agent 通过 Skill Registry 调用能力
3. 移除 Agent 内部直接调用 Analyzer

### 5.4 Phase 4: 平台集成（1 周）

1. 创建 SDK
2. 编写 Claude Code 集成文档
3. 编写 OpenCode 集成文档

---

## 六、预期收益

| 收益 | 描述 |
|------|------|
| **可扩展性** | 新增 Skill 只需创建文件，无需修改核心代码 |
| **可维护性** | 职责清晰，Agent/Skill/Analyzer 层次分明 |
| **可复用性** | Skills 可独立打包，集成到其他平台 |
| **可测试性** | 每个 Skill 可独立单元测试 |
| **可观测性** | 统一的日志、指标、追踪 |
| **专家友好** | Markdown 规则格式，便于专家追加经验 |