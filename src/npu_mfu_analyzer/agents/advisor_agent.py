"""
Advisor Agent

综合各维度分析结果，生成最终的性能优化报告和建议。
"""

import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from enum import Enum

from npu_mfu_analyzer.agents.base_agent import BaseAgent, AnalysisResult
from npu_mfu_analyzer.llm.llm_interface import LLMInterface
from npu_mfu_analyzer.llm.prompts import ADVISOR_SYSTEM
from npu_mfu_analyzer.skills.v2.base import SkillContext
from npu_mfu_analyzer.skills.v2.engine import SkillEngine
from npu_mfu_analyzer.skills.v2.registry import SkillRegistry
from npu_mfu_analyzer.skills.v2.skills.analysis import (
    MemoryAttributionSkill,
    CommunicationExposureSkill,
    StepAttributionSkill,
    PipelineDiagnosisSkill,
    ScenarioDiagnosisSkill,
    TopologyDiagnosisSkill,
    WhatIfExperimentSkill,
    RegressionDiagnosisSkill,
    MainContradictionSkill,
    ActionPrioritizationSkill,
)

logger = logging.getLogger(__name__)


class Priority(Enum):
    """优化建议优先级"""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class OptimizationSuggestion:
    """优化建议"""
    title: str
    description: str
    priority: Priority
    category: str  # "compute", "communication", "memory", "io"
    expected_benefit: str  # 预期收益描述
    code_example: Optional[str] = None
    config_example: Optional[str] = None

    def to_markdown(self) -> str:
        """转换为 Markdown 格式"""
        priority_emoji = {"high": "🔴", "medium": "🟡", "low": "🟢"}
        emoji = priority_emoji.get(self.priority.value, "⚪")

        lines = [
            f"### {emoji} {self.title}",
            "",
            f"**优先级**: {self.priority.value.upper()}",
            f"**类别**: {self.category}",
            f"**预期收益**: {self.expected_benefit}",
            "",
            self.description,
        ]

        if self.code_example:
            lines.extend(["", "**代码示例**:", "", "```python", self.code_example, "```"])

        if self.config_example:
            lines.extend(["", "**配置示例**:", "", "```yaml", self.config_example, "```"])

        return "\n".join(lines)


@dataclass
class PerformanceOverview:
    """性能概览"""
    estimated_mfu: float = 0.0
    mfu_target: float = 0.55
    mfu_gap: float = 0.0

    main_bottleneck: str = ""
    bottleneck_impact: float = 0.0

    time_breakdown: Dict[str, float] = field(default_factory=dict)

    def to_markdown(self) -> str:
        """转换为 Markdown 格式"""
        lines = [
            "## 性能概览",
            "",
            "| 指标 | 当前值 | 目标值 | 差距 |",
            "|------|--------|--------|------|",
            f"| MFU | {self.estimated_mfu*100:.1f}% | {self.mfu_target*100:.1f}% | {self.mfu_gap*100:.1f}% |",
            "",
            f"**主要瓶颈**: {self.main_bottleneck}",
            f"**瓶颈影响**: {self.bottleneck_impact:.1f}%",
        ]

        if self.time_breakdown:
            lines.extend(["", "### 时间分布", ""])
            for name, pct in self.time_breakdown.items():
                bar_len = int(pct / 5)
                bar = "█" * bar_len + "░" * (20 - bar_len)
                lines.append(f"- {name}: {bar} {pct:.1f}%")

        return "\n".join(lines)


@dataclass
class AdvisorReport:
    """Advisor 生成的完整报告"""
    overview: PerformanceOverview
    bottlenecks: List[Dict[str, Any]] = field(default_factory=list)
    suggestions: List[OptimizationSuggestion] = field(default_factory=list)
    raw_analysis: str = ""
    diagnosis: Dict[str, Any] = field(default_factory=dict)

    def to_markdown(self) -> str:
        """转换为 Markdown 格式"""
        lines = [
            "# 昇腾 NPU 性能优化报告",
            "",
            self.overview.to_markdown(),
            "",
        ]

        if self.diagnosis:
            lines.extend(["## 诊断结论", ""])
            main = self.diagnosis.get("main_contradiction", {})
            if main:
                lines.append(f"- **主矛盾**: {main.get('code', 'unknown')}")
                lines.append(f"- **所属层**: {main.get('layer', 'unknown')}")
                lines.append(f"- **判断理由**: {main.get('reason', '')}")
                lines.append(f"- **置信度**: {main.get('confidence', 0):.0%}")
                lines.append(f"- **阶段焦点**: {self.diagnosis.get('phase_focus', 'unknown')}")
                lines.append(f"- **训练场景**: {self.diagnosis.get('training_scenario', 'GENERAL_DENSE')}")
                lines.append("")

            facts = self.diagnosis.get("observation_facts", [])
            if facts:
                lines.append("### 关键观测")
                lines.append("")
                for fact in facts:
                    lines.append(f"- {fact}")
                lines.append("")

            actions = self.diagnosis.get("prioritized_actions", [])
            if actions:
                lines.append("### 优先调优动作")
                lines.append("")
                for idx, action in enumerate(actions, 1):
                    lines.append(f"{idx}. **{action.get('title', action.get('action_code', 'action'))}**")
                    if action.get("description"):
                        lines.append(f"   - 动作: {action['description']}")
                    if action.get("expected_effect"):
                        lines.append(f"   - 预期效果: {action['expected_effect']}")
                    if action.get("rationale"):
                        lines.append(f"   - 原因: {action['rationale']}")
                    anti_actions = action.get("anti_actions") or []
                    if anti_actions:
                        lines.append(f"   - 当前不建议优先: {'; '.join(anti_actions)}")
                lines.append("")

            experiments = self.diagnosis.get("experiments", [])
            if experiments:
                lines.append("### 建议实验")
                lines.append("")
                for idx, item in enumerate(experiments, 1):
                    lines.append(f"{idx}. **{item.get('title', 'experiment')}**")
                    if item.get("hypothesis"):
                        lines.append(f"   - 假设: {item['hypothesis']}")
                    if item.get("expected_signal"):
                        lines.append(f"   - 预期信号: {item['expected_signal']}")
                lines.append("")

            regression_template = self.diagnosis.get("regression_template", [])
            if regression_template:
                lines.append("### 回归诊断模板")
                lines.append("")
                for item in regression_template:
                    lines.append(f"- {item}")
                lines.append("")

        if self.bottlenecks:
            lines.extend([
                "## 瓶颈分析",
                "",
                "| 排名 | 瓶颈类型 | 影响程度 | 描述 |",
                "|------|---------|---------|------|",
            ])
            for i, b in enumerate(self.bottlenecks[:10], 1):
                lines.append(
                    f"| {i} | {b.get('type', 'unknown')} | {b.get('impact', 0):.1f}% | {b.get('description', '')} |"
                )
            lines.append("")

        if self.suggestions:
            lines.extend(["## 优化建议", ""])
            high = [s for s in self.suggestions if s.priority == Priority.HIGH]
            medium = [s for s in self.suggestions if s.priority == Priority.MEDIUM]
            low = [s for s in self.suggestions if s.priority == Priority.LOW]

            if high:
                lines.append("### 高优先级（预期收益 > 10%）")
                lines.append("")
                for s in high:
                    lines.append(s.to_markdown())
                    lines.append("")

            if medium:
                lines.append("### 中优先级（预期收益 5-10%）")
                lines.append("")
                for s in medium:
                    lines.append(s.to_markdown())
                    lines.append("")

            if low:
                lines.append("### 低优先级（预期收益 < 5%）")
                lines.append("")
                for s in low:
                    lines.append(s.to_markdown())
                    lines.append("")

        return "\n".join(lines)


class AdvisorAgent(BaseAgent):
    """综合各维度分析结果，输出最终优化报告。"""

    PROMPT_TEMPLATE = """
你是昇腾 NPU 训练性能优化顾问。请综合以下分析结果，生成最终的性能优化报告。

{analysis_results}

## 报告要求

### 1. 性能概览
- 估算当前 MFU（基于计算时间占比）
- 识别主要瓶颈（计算/通信/内存/IO）
- 量化瓶颈影响（占总时间的百分比）

### 2. 瓶颈分析
按影响程度从高到低列出所有性能问题：
- 瓶颈类型
- 影响程度（%）
- 根本原因

### 3. 优化建议
为每个问题给出具体、可操作的优化方案：
- 优先级（高/中/低）
- 预期收益（%）
- 具体操作步骤
- 代码或配置示例

### 4. 预期效果
- 优化后的预期 MFU
- 预期训练时间缩短比例

诊断输出必须优先尊重结构化结论：
1. 先说明当前主矛盾属于并行维 / 内存策略 / 系统优化中的哪一层
2. 明确先改什么，不先改什么
3. 不要把泛化建议放在结构化优先动作前面

请以结构化的 Markdown 格式输出。
"""

    def __init__(self, llm: LLMInterface, config: Optional[Dict[str, Any]] = None):
        super().__init__(
            name="AdvisorAgent",
            llm=llm,
            system_prompt=ADVISOR_SYSTEM,
            config=config,
        )
        self._rules = OptimizationRules()

    def get_prompt_template(self) -> str:
        return self.PROMPT_TEMPLATE

    async def analyze(self, data: Dict[str, Any]) -> AnalysisResult:
        try:
            analysis_text = self._build_analysis_summary(data)
            rule_suggestions = self._apply_rules(data)
            diagnosis_results = self._run_diagnosis_skills(data)
            diagnosis_summary = self._summarize_diagnosis_results(diagnosis_results)

            if diagnosis_summary:
                analysis_text = f"{analysis_text}\n\n## 结构化诊断结论\n{self._format_diagnosis_summary(diagnosis_summary)}"

            prompt = self.format_prompt(self.PROMPT_TEMPLATE, analysis_results=analysis_text)
            response = await self.call_llm(prompt)
            report = self._build_report(data, response, rule_suggestions, diagnosis_results)

            return AnalysisResult(
                agent_name=self.name,
                success=True,
                summary=f"综合分析完成，生成 {len(report.suggestions)} 条优化建议",
                details={
                    "estimated_mfu": report.overview.estimated_mfu,
                    "main_bottleneck": report.overview.main_bottleneck,
                    "suggestion_count": len(report.suggestions),
                    "high_priority_count": len([s for s in report.suggestions if s.priority == Priority.HIGH]),
                    "advisor_report": report,
                    "diagnosis": report.diagnosis,
                },
                recommendations=[s.title for s in report.suggestions],
                raw_response=report.to_markdown(),
            )
        except Exception as e:
            logger.error(f"Advisor analysis failed: {e}", exc_info=True)
            diagnosis_results = locals().get("diagnosis_results", {})
            rule_suggestions = locals().get("rule_suggestions", [])
            fallback_report = self._build_report(data, "", rule_suggestions, diagnosis_results)
            return AnalysisResult(
                agent_name=self.name,
                success=True,
                summary="综合分析完成（LLM 调用失败，已回退为结构化诊断结果）",
                details={
                    "estimated_mfu": fallback_report.overview.estimated_mfu,
                    "main_bottleneck": fallback_report.overview.main_bottleneck,
                    "suggestion_count": len(fallback_report.suggestions),
                    "high_priority_count": len([s for s in fallback_report.suggestions if s.priority == Priority.HIGH]),
                    "advisor_report": fallback_report,
                    "diagnosis": fallback_report.diagnosis,
                    "llm_fallback": True,
                    "llm_error": str(e),
                },
                recommendations=[s.title for s in fallback_report.suggestions],
                raw_response=fallback_report.to_markdown(),
                error=str(e),
            )

    def _run_diagnosis_skills(self, data: Dict[str, Any]) -> Dict[str, Any]:
        try:
            registry = SkillRegistry()
            registry.register(MemoryAttributionSkill())
            registry.register(CommunicationExposureSkill())
            registry.register(StepAttributionSkill())
            registry.register(PipelineDiagnosisSkill())
            registry.register(ScenarioDiagnosisSkill())
            registry.register(TopologyDiagnosisSkill())
            registry.register(WhatIfExperimentSkill())
            registry.register(RegressionDiagnosisSkill())
            registry.register(MainContradictionSkill())
            registry.register(ActionPrioritizationSkill())
            engine = SkillEngine(registry)
            context = SkillContext(
                profiling_summary=data.get("profiling_summary"),
                user_inputs={
                    "agent_results": data.get("agent_results", {}),
                    "topology_summary": data.get("topology_summary", {}),
                    "tp_size": data.get("tp_size"),
                    "pp_size": data.get("pp_size"),
                    "dp_size": data.get("dp_size"),
                    "diff_result": data.get("diff_result"),
                },
                diff_result=data.get("diff_result"),
            )
            return engine.execute_skills(["prioritize_optimization_actions"], context)
        except Exception as e:
            logger.warning(f"Diagnosis skills execution failed: {e}")
            return {}

    def _summarize_diagnosis_results(self, diagnosis_results: Dict[str, Any]) -> Dict[str, Any]:
        if not diagnosis_results:
            return {}

        memory_result = diagnosis_results.get("attribute_memory_bottleneck")
        comm_result = diagnosis_results.get("diagnose_communication_exposure")
        plan_result = diagnosis_results.get("prioritize_optimization_actions")
        scenario_result = diagnosis_results.get("diagnose_training_scenario")
        step_result = diagnosis_results.get("attribute_step_bottleneck")
        topology_result = diagnosis_results.get("diagnose_topology_pressure")

        observation_facts: List[str] = []
        for result in [memory_result, comm_result, step_result, scenario_result, topology_result]:
            if result and result.success:
                observation_facts.extend(result.data.get("observation_facts", []))
        observation_facts = list(dict.fromkeys(observation_facts))

        summary: Dict[str, Any] = {"observation_facts": observation_facts}
        if memory_result and memory_result.success:
            summary["memory"] = memory_result.data
        if comm_result and comm_result.success:
            summary["communication"] = comm_result.data
        if plan_result and plan_result.success:
            summary["main_contradiction"] = plan_result.data.get("main_contradiction", {})
            summary["prioritized_actions"] = plan_result.data.get("prioritized_actions", [])
            summary["supporting_facts"] = plan_result.data.get("supporting_facts", [])
            summary["phase_focus"] = plan_result.data.get("phase_focus", "unknown")
            summary["training_scenario"] = plan_result.data.get("training_scenario", "GENERAL_DENSE")
            summary["experiments"] = plan_result.data.get("experiments", [])
            summary["regression_template"] = plan_result.data.get("regression_template", [])
        return summary

    def _format_diagnosis_summary(self, diagnosis_summary: Dict[str, Any]) -> str:
        if not diagnosis_summary:
            return "- 暂无结构化诊断结果"

        lines: List[str] = []
        main = diagnosis_summary.get("main_contradiction", {})
        if main:
            lines.append(f"- 主矛盾: {main.get('code', 'unknown')} ({main.get('layer', 'unknown')})")
            lines.append(f"- 理由: {main.get('reason', '')}")
            lines.append(f"- 置信度: {main.get('confidence', 0):.0%}")
        lines.append(f"- 阶段焦点: {diagnosis_summary.get('phase_focus', 'unknown')}")
        lines.append(f"- 训练场景: {diagnosis_summary.get('training_scenario', 'GENERAL_DENSE')}")

        facts = diagnosis_summary.get("observation_facts", [])
        if facts:
            lines.append("- 关键观测:")
            for fact in facts[:8]:
                lines.append(f"  - {fact}")

        actions = diagnosis_summary.get("prioritized_actions", [])
        if actions:
            lines.append("- 优先动作:")
            for action in actions[:3]:
                lines.append(f"  - {action.get('title', action.get('action_code', 'action'))}")

        experiments = diagnosis_summary.get("experiments", [])
        if experiments:
            lines.append("- 建议实验:")
            for item in experiments[:3]:
                lines.append(f"  - {item.get('title', 'experiment')}")

        return "\n".join(lines)

    def _build_analysis_summary(self, data: Dict[str, Any]) -> str:
        lines = []
        if "profiling_summary" in data:
            summary = data["profiling_summary"]
            if hasattr(summary, "to_prompt_text"):
                lines.append(summary.to_prompt_text())
            elif isinstance(summary, dict):
                lines.append("## Profiling 数据摘要")
                lines.append(f"- Step 数量: {summary.get('step_count', 0)}")
                lines.append(f"- 平均 Step 时间: {summary.get('avg_step_time', 0)/1000:.2f} ms")
                lines.append(f"- 计算时间: {summary.get('avg_compute_time', 0)/1000:.2f} ms")
                lines.append(f"- 通信时间: {summary.get('avg_comm_time', 0)/1000:.2f} ms")
                lines.append(f"- 空闲时间: {summary.get('avg_free_time', 0)/1000:.2f} ms")

        agent_results = data.get("agent_results", {})
        for name, result in agent_results.items():
            lines.append(f"\n## {name} 分析结果")
            if isinstance(result, AnalysisResult):
                if result.success:
                    lines.append(result.raw_response or result.summary)
                else:
                    lines.append(f"分析失败: {result.error}")
            elif isinstance(result, dict):
                lines.append(str(result))
        return "\n".join(lines)

    def _apply_rules(self, data: Dict[str, Any]) -> List[OptimizationSuggestion]:
        suggestions = []
        profiling_summary = data.get("profiling_summary", {})
        if hasattr(profiling_summary, "to_dict"):
            profiling_summary = profiling_summary.to_dict()

        compute_time = profiling_summary.get("avg_compute_time", 0)
        comm_time = profiling_summary.get("avg_comm_time", 0)
        free_time = profiling_summary.get("avg_free_time", 0)
        total_time = compute_time + comm_time + free_time

        if total_time > 0:
            compute_ratio = compute_time / total_time
            comm_ratio = comm_time / total_time
            free_ratio = free_time / total_time
            suggestions.extend(self._rules.check_compute_efficiency(compute_ratio))
            suggestions.extend(self._rules.check_communication_overhead(comm_ratio))
            suggestions.extend(self._rules.check_idle_time(free_ratio))

        overlap_ratio = profiling_summary.get("overlap_ratio", 0)
        suggestions.extend(self._rules.check_overlap_ratio(overlap_ratio))
        return suggestions

    def _build_report(
        self,
        data: Dict[str, Any],
        llm_response: str,
        rule_suggestions: List[OptimizationSuggestion],
        diagnosis_results: Dict[str, Any],
    ) -> AdvisorReport:
        profiling_summary = data.get("profiling_summary", {})
        if hasattr(profiling_summary, "to_dict"):
            profiling_summary = profiling_summary.to_dict()

        compute_time = profiling_summary.get("avg_compute_time", 0)
        comm_time = profiling_summary.get("avg_comm_time", 0)
        free_time = profiling_summary.get("avg_free_time", 0)
        total_time = compute_time + comm_time + free_time

        compute_ratio = compute_time / total_time if total_time > 0 else 0
        estimated_mfu = compute_ratio * 0.8

        diagnosis_summary = self._summarize_diagnosis_results(diagnosis_results)
        main_diag = diagnosis_summary.get("main_contradiction", {})

        if main_diag:
            main_bottleneck = main_diag.get("reason", main_diag.get("code", ""))
            bottleneck_impact = max(comm_time / total_time * 100 if total_time > 0 else 0, 15.0)
        elif free_time > compute_time and free_time > comm_time:
            main_bottleneck = "空闲时间过长（可能是数据加载或 Host 端瓶颈）"
            bottleneck_impact = free_time / total_time * 100 if total_time > 0 else 0
        elif comm_time > compute_time * 0.3:
            main_bottleneck = "通信开销过大"
            bottleneck_impact = comm_time / total_time * 100 if total_time > 0 else 0
        else:
            main_bottleneck = "计算效率待优化"
            bottleneck_impact = (1 - compute_ratio) * 100

        overview = PerformanceOverview(
            estimated_mfu=estimated_mfu,
            mfu_target=0.55,
            mfu_gap=0.55 - estimated_mfu,
            main_bottleneck=main_bottleneck,
            bottleneck_impact=bottleneck_impact,
            time_breakdown={
                "计算": compute_ratio * 100,
                "通信": comm_time / total_time * 100 if total_time > 0 else 0,
                "空闲": free_time / total_time * 100 if total_time > 0 else 0,
            },
        )

        bottlenecks = self._extract_bottlenecks(llm_response)
        llm_suggestions = self._extract_suggestions(llm_response)
        diagnosis_suggestions = self._diagnosis_actions_to_suggestions(diagnosis_summary)
        all_suggestions = self._deduplicate_suggestions(diagnosis_suggestions + rule_suggestions + llm_suggestions)

        return AdvisorReport(
            overview=overview,
            bottlenecks=bottlenecks,
            suggestions=all_suggestions,
            raw_analysis=llm_response,
            diagnosis=diagnosis_summary,
        )

    def _deduplicate_suggestions(self, suggestions: List[OptimizationSuggestion]) -> List[OptimizationSuggestion]:
        deduped: List[OptimizationSuggestion] = []
        seen = set()
        for suggestion in suggestions:
            key = suggestion.title.strip()
            if key and key not in seen:
                seen.add(key)
                deduped.append(suggestion)
        return deduped

    def _diagnosis_actions_to_suggestions(self, diagnosis_summary: Dict[str, Any]) -> List[OptimizationSuggestion]:
        suggestions: List[OptimizationSuggestion] = []
        actions = diagnosis_summary.get("prioritized_actions", []) if diagnosis_summary else []
        for action in actions:
            priority = {
                "high": Priority.HIGH,
                "medium": Priority.MEDIUM,
                "low": Priority.LOW,
            }.get(str(action.get("priority", "medium")).lower(), Priority.MEDIUM)
            description = action.get("description", "")
            anti_actions = action.get("anti_actions") or []
            if anti_actions:
                description = f"{description} 当前不建议优先: {'; '.join(anti_actions)}"
            suggestions.append(OptimizationSuggestion(
                title=action.get("title", action.get("action_code", "结构化诊断动作")),
                description=description,
                priority=priority,
                category=action.get("layer", "diagnosis"),
                expected_benefit=action.get("expected_effect", "待进一步验证"),
            ))
        return suggestions

    def _extract_bottlenecks(self, response: str) -> List[Dict[str, Any]]:
        return []

    def _extract_suggestions(self, response: str) -> List[OptimizationSuggestion]:
        suggestions = []
        lines = response.split("\n")
        current_priority = Priority.MEDIUM

        for line in lines:
            line_lower = line.lower()
            if "高优先级" in line or "high priority" in line_lower:
                current_priority = Priority.HIGH
            elif "中优先级" in line or "medium priority" in line_lower:
                current_priority = Priority.MEDIUM
            elif "低优先级" in line or "low priority" in line_lower:
                current_priority = Priority.LOW

            if line.strip().startswith(("-", "*", "•")) and len(line.strip()) > 10:
                title = line.strip().lstrip("-*• ")
                if len(title) > 5:
                    suggestions.append(OptimizationSuggestion(
                        title=title[:100],
                        description="",
                        priority=current_priority,
                        category="general",
                        expected_benefit="待评估",
                    ))

        return suggestions[:20]


class OptimizationRules:
    """
    优化建议规则库
    
    基于性能指标自动生成优化建议。
    """
    
    def check_compute_efficiency(self, compute_ratio: float) -> List[OptimizationSuggestion]:
        """检查计算效率"""
        suggestions = []
        
        if compute_ratio < 0.5:
            suggestions.append(OptimizationSuggestion(
                title="计算时间占比过低，需要减少非计算开销",
                description="计算时间占比低于 50%，说明大量时间消耗在通信、数据加载或等待上。",
                priority=Priority.HIGH,
                category="compute",
                expected_benefit="预计可提升 20-40% 训练速度",
                code_example="""# 使用 DataLoader 预加载
dataloader = DataLoader(
    dataset,
    batch_size=batch_size,
    num_workers=8,  # 增加 worker 数量
    pin_memory=True,
    prefetch_factor=2,
)""",
            ))
        
        return suggestions
    
    def check_communication_overhead(self, comm_ratio: float) -> List[OptimizationSuggestion]:
        """检查通信开销"""
        suggestions = []
        
        if comm_ratio > 0.3:
            suggestions.append(OptimizationSuggestion(
                title="通信开销过大（> 30%），考虑优化并行策略",
                description="通信时间占比超过 30%，可能需要调整 TP/DP/PP 配比或使用梯度累积。",
                priority=Priority.HIGH,
                category="communication",
                expected_benefit="预计可减少 10-30% 通信时间",
                config_example="""# 使用梯度累积减少 DP 通信
gradient_accumulation_steps: 4

# 调整并行策略
tensor_model_parallel_size: 8
pipeline_model_parallel_size: 4
data_parallel_size: 2""",
            ))
        elif comm_ratio > 0.2:
            suggestions.append(OptimizationSuggestion(
                title="通信开销中等，可尝试通信-计算重叠优化",
                description="通信时间占比 20-30%，可以通过提升通信掩盖率来优化。",
                priority=Priority.MEDIUM,
                category="communication",
                expected_benefit="预计可减少 5-15% 通信时间",
            ))
        
        return suggestions
    
    def check_idle_time(self, idle_ratio: float) -> List[OptimizationSuggestion]:
        """检查空闲时间"""
        suggestions = []
        
        if idle_ratio > 0.2:
            suggestions.append(OptimizationSuggestion(
                title="空闲时间过长（> 20%），排查数据加载或调度问题",
                description="空闲时间占比超过 20%，可能是数据加载瓶颈、Host 端计算、或调度延迟。",
                priority=Priority.HIGH,
                category="io",
                expected_benefit="预计可减少 50-80% 空闲时间",
                code_example="""# 异步数据预取
from torch.utils.data import DataLoader

# 使用 pin_memory 和 non_blocking
dataloader = DataLoader(..., pin_memory=True)
for data in dataloader:
    data = data.to(device, non_blocking=True)
    
# 使用 torch.cuda.Stream 实现数据预取
prefetch_stream = torch.cuda.Stream()""",
            ))
        elif idle_ratio > 0.1:
            suggestions.append(OptimizationSuggestion(
                title="存在空闲时间（10-20%），可优化数据管道",
                description="存在一定的空闲时间，建议检查数据加载效率。",
                priority=Priority.MEDIUM,
                category="io",
                expected_benefit="预计可减少 30-50% 空闲时间",
            ))
        
        return suggestions
    
    def check_overlap_ratio(self, overlap_ratio: float) -> List[OptimizationSuggestion]:
        """检查通信掩盖率"""
        suggestions = []
        
        if 0 < overlap_ratio < 50:
            suggestions.append(OptimizationSuggestion(
                title="通信掩盖率低（< 50%），启用通信-计算重叠",
                description="当前通信掩盖率较低，大部分通信时间未被计算隐藏。",
                priority=Priority.HIGH,
                category="communication",
                expected_benefit="预计可提升 10-25% 训练速度",
                config_example="""# MindFormers 配置
use_parallel_optimizer: true
overlap_grad_reduce: true

# Megatron 配置
--overlap-grad-reduce
--overlap-param-gather""",
            ))
        elif 50 <= overlap_ratio < 80:
            suggestions.append(OptimizationSuggestion(
                title="通信掩盖率中等（50-80%），可进一步优化",
                description="通信掩盖率有提升空间，可尝试调整 micro_batch_size 或启用更多重叠优化。",
                priority=Priority.MEDIUM,
                category="communication",
                expected_benefit="预计可提升 5-10% 训练速度",
            ))
        
        return suggestions
    
    def check_bubble_ratio(self, bubble_ratio: float, ideal_ratio: float) -> List[OptimizationSuggestion]:
        """检查 PP Bubble 比例"""
        suggestions = []
        
        if bubble_ratio > ideal_ratio * 1.2:
            suggestions.append(OptimizationSuggestion(
                title="PP Bubble 超出理论值，检查 Stage 负载均衡",
                description=f"实际 Bubble 比例 ({bubble_ratio:.1f}%) 显著高于理论值 ({ideal_ratio:.1f}%)，"
                           "可能是 Stage 划分不均匀或存在阻塞。",
                priority=Priority.HIGH,
                category="compute",
                expected_benefit="预计可减少 5-15% Bubble 时间",
                config_example="""# 增加 micro batch 数量减少 Bubble
micro_batch_size: 1
global_batch_size: 512
# Bubble 比例 = (pp_size - 1) / (global_batch / micro_batch / dp)

# 使用 interleaved schedule
virtual_pipeline_model_parallel_size: 2""",
            ))
        
        return suggestions
