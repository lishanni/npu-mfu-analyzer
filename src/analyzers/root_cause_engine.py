"""
根因推理引擎

基于 Markdown Skills 的根因推理引擎，用于分析 Profiling 对比中的性能问题根因。
"""

import logging
import re
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path
from collections import Counter, defaultdict

from src.data_loader.stack_types import HostDeviceChain, SourceAnalysisResult
from src.analyzers.profiling_diff import ProfilingDiff, OperatorChange, OperatorDiff
from src.analyzers.operator_source_classifier import SourceChange, OperatorSourceClassifier

logger = logging.getLogger(__name__)


@dataclass
class RootCauseRule:
    """根因推理规则"""
    name: str
    trigger_conditions: Dict[str, Any]
    root_cause_description: str
    evidence_patterns: List[str]
    optimization_suggestions: List[str]
    impact: Dict[str, str]  # performance_impact, fix_difficulty, priority
    rule_file: str = ""


@dataclass
class RootCauseFinding:
    """根因发现"""
    rule_name: str
    root_cause: str
    evidence: List[str]
    affected_operators: List[str]
    optimization_suggestions: List[str]
    impact: str  # "high" / "medium" / "low"
    priority: str  # "P0" / "P1" / "P2"
    confidence: float  # 0-1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rule_name": self.rule_name,
            "root_cause": self.root_cause,
            "evidence": self.evidence,
            "affected_operators": self.affected_operators,
            "optimization_suggestions": self.optimization_suggestions,
            "impact": self.impact,
            "priority": self.priority,
            "confidence": self.confidence,
        }

    def to_markdown(self) -> str:
        """转换为 Markdown 格式"""
        lines = [
            f"### [{self.priority}] {self.rule_name}",
            "",
            f"**根因**: {self.root_cause}",
            "",
            "**证据**:",
        ]
        for e in self.evidence:
            lines.append(f"- {e}")

        if self.affected_operators:
            lines.append("")
            lines.append(f"**受影响算子**: {', '.join(self.affected_operators[:10])}")

        lines.append("")
        lines.append("**建议**:")
        for s in self.optimization_suggestions[:5]:
            lines.append(f"- {s}")

        return "\n".join(lines)


class RootCauseSkillEngine:
    """
    基于 Markdown Skills 的根因推理引擎

    Usage:
        engine = RootCauseSkillEngine()
        findings = engine.analyze(chains_a, chains_b, diff_result)
    """

    def __init__(self, skills_dir: Optional[str] = None):
        """
        Args:
            skills_dir: Skills 目录路径
        """
        if skills_dir is None:
            skills_dir = str(Path(__file__).parent.parent / "skills" / "root_cause_analysis")

        self.skills_dir = Path(skills_dir)
        self.rules = self._load_builtin_rules()
        self.classifier = OperatorSourceClassifier()

    def _load_builtin_rules(self) -> List[RootCauseRule]:
        """加载内置规则"""
        rules = []

        # 规则 1: torch.compile 融合问题
        rules.append(RootCauseRule(
            name="torch_compile_fusion_issue",
            trigger_conditions={
                "stack_pattern": ["CompiledFunctionBackward", "torch._inductor"],
                "missing_fusion": ["NPUGroupedMatmul", "FlashAttention"],
                "small_op_increase": 50,  # > 50%
            },
            root_cause="torch.compile 图模式未开启对应的算子融合，导致下发大量小算子",
            evidence_patterns=[
                "堆栈中出现 CompiledFunctionBackward 但缺少 NPUGroupedGMM",
                "小算子数量显著增加",
                "端到端耗时增加但单个算子耗时无明显变化",
            ],
            optimization_suggestions=[
                "启用 torch.compile(model, mode=\"reduce-overhead\")",
                "添加自定义融合规则",
                "检查 mindspeed 配置是否与 torch.compile 兼容",
            ],
            impact={"performance_impact": "high", "fix_difficulty": "medium", "priority": "P0"},
            rule_file="compile_fusion_issue.md",
        ))

        # 规则 2: eager/compile 模式切换
        rules.append(RootCauseRule(
            name="eager_compile_switch",
            trigger_conditions={
                "source_change": True,
                "mode_switch": ["eager", "torch_compile"],
            },
            root_cause="两个 Profiling 版本使用了不同的执行模式（eager vs torch.compile）",
            evidence_patterns=[
                "版本 A 堆栈显示 aten:: 开头的原生算子",
                "版本 B 堆栈显示 CompiledFunction 或 torch._dynamo",
                "两个版本的算子数量差异大",
            ],
            optimization_suggestions=[
                "确保对比的两个版本使用相同的执行模式",
                "分离对比维度，单独分析编译开销和收益",
            ],
            impact={"performance_impact": "high", "fix_difficulty": "low", "priority": "P0"},
            rule_file="eager_compile_switch.md",
        ))

        # 规则 3: 通信瓶颈
        rules.append(RootCauseRule(
            name="communication_bottleneck",
            trigger_conditions={
                "comm_ratio_increase": 10,  # > 10%
                "comm_time_increase": 20,   # > 20%
            },
            root_cause="分布式训练中的通信瓶颈导致性能下降",
            evidence_patterns=[
                "通信时间占比显著增加",
                "未掩盖通信时间增加",
                "存在带宽利用率低于阈值的链路",
            ],
            optimization_suggestions=[
                "优化通信掩盖，使用异步通信",
                "检查网络拓扑和 HCCS/RDMA 配置",
                "根据模型大小选择合适的集合通信算法",
            ],
            impact={"performance_impact": "high", "fix_difficulty": "medium", "priority": "P0"},
            rule_file="communication_bottleneck.md",
        ))

        return rules

    def analyze(
        self,
        chains_a: List[HostDeviceChain],
        chains_b: List[HostDeviceChain],
        diff_result: ProfilingDiff,
        source_changes: Optional[List[SourceChange]] = None,
    ) -> List[RootCauseFinding]:
        """
        执行根因推理

        Args:
            chains_a: 版本 A 的调用链
            chains_b: 版本 B 的调用链
            diff_result: ProfilingDiff 对比结果
            source_changes: 来源变化列表

        Returns:
            RootCauseFinding 列表
        """
        findings = []

        # 1. 分析来源变化
        if source_changes is None:
            source_changes = self.classifier.compute_source_changes(chains_a, chains_b)

        # 2. 检查各规则
        for rule in self.rules:
            finding = self._check_rule(rule, chains_a, chains_b, diff_result, source_changes)
            if finding:
                findings.append(finding)

        # 3. 按优先级排序
        priority_order = {"P0": 0, "P1": 1, "P2": 2}
        findings.sort(key=lambda f: (priority_order.get(f.priority, 99), -f.confidence))

        return findings

    def _check_rule(
        self,
        rule: RootCauseRule,
        chains_a: List[HostDeviceChain],
        chains_b: List[HostDeviceChain],
        diff_result: ProfilingDiff,
        source_changes: List[SourceChange],
    ) -> Optional[RootCauseFinding]:
        """检查单个规则是否匹配"""
        conditions = rule.trigger_conditions

        # 检查 torch.compile 融合问题
        if rule.name == "torch_compile_fusion_issue":
            return self._check_compile_fusion_issue(rule, chains_a, chains_b, diff_result, source_changes)

        # 检查 eager/compile 模式切换
        if rule.name == "eager_compile_switch":
            return self._check_eager_compile_switch(rule, chains_a, chains_b, diff_result, source_changes)

        # 检查通信瓶颈
        if rule.name == "communication_bottleneck":
            return self._check_communication_bottleneck(rule, diff_result)

        return None

    def _check_compile_fusion_issue(
        self,
        rule: RootCauseRule,
        chains_a: List[HostDeviceChain],
        chains_b: List[HostDeviceChain],
        diff_result: ProfilingDiff,
        source_changes: List[SourceChange],
    ) -> Optional[RootCauseFinding]:
        """检查 torch.compile 融合问题"""
        conditions = rule.trigger_conditions

        # 统计版本 B 中的 torch.compile 算子
        compile_chains_b = [c for c in chains_b if c.source_type == "torch_compile"]
        fusion_chains_a = [c for c in chains_a if c.source_type == "fusion_op"]

        if not compile_chains_b:
            return None

        # 检查小算子增加
        small_op_increase_threshold = conditions.get("small_op_increase", 50)

        # 统计小算子数量
        small_ops_a = [c for c in chains_a if c.device_op_dur < 10]  # < 10us
        small_ops_b = [c for c in chains_b if c.device_op_dur < 10]

        small_op_increase = 0
        if len(small_ops_a) > 0:
            small_op_increase = (len(small_ops_b) - len(small_ops_a)) / len(small_ops_a) * 100

        if small_op_increase < small_op_increase_threshold:
            return None

        # 收集证据
        evidence = []
        affected_ops = []

        # 检查新增的小算子
        if diff_result.operator_diff and diff_result.operator_diff.new_operators:
            new_small_ops = [op for op in diff_result.operator_diff.new_operators
                           if "Add" in op.name or "zeroslike" in op.name or "Mul" in op.name]
            if new_small_ops:
                evidence.append(f"新增小算子数量: {len(new_small_ops)}")
                affected_ops = [op.name for op in new_small_ops[:10]]

        # 检查堆栈模式
        compile_pattern_count = sum(1 for c in compile_chains_b
                                   if any(p in " ".join(c.python_stack) for p in conditions.get("stack_pattern", [])))
        if compile_pattern_count > 0:
            evidence.append(f"检测到 {compile_pattern_count} 个 torch.compile 编译算子")

        # 检查缺少的融合算子
        fusion_ops_a = set(c.device_op_name for c in fusion_chains_a)
        fusion_ops_b = set(c.device_op_name for c in chains_b if c.source_type == "fusion_op")
        missing_fusion = fusion_ops_a - fusion_ops_b
        if missing_fusion:
            evidence.append(f"版本 B 缺少融合算子: {', '.join(list(missing_fusion)[:5])}")

        if not evidence:
            return None

        return RootCauseFinding(
            rule_name=rule.name,
            root_cause=rule.root_cause_description,
            evidence=evidence,
            affected_operators=affected_ops,
            optimization_suggestions=rule.optimization_suggestions,
            impact=rule.impact.get("performance_impact", "medium"),
            priority=rule.impact.get("priority", "P1"),
            confidence=min(0.5 + small_op_increase / 100, 0.95),
        )

    def _check_eager_compile_switch(
        self,
        rule: RootCauseRule,
        chains_a: List[HostDeviceChain],
        chains_b: List[HostDeviceChain],
        diff_result: ProfilingDiff,
        source_changes: List[SourceChange],
    ) -> Optional[RootCauseFinding]:
        """检查 eager/compile 模式切换"""
        # 统计来源分布
        sources_a = Counter(c.source_type for c in chains_a)
        sources_b = Counter(c.source_type for c in chains_b)

        # 检查模式切换
        mode_switches = [sc for sc in source_changes if sc.change_type == "mode_switch"]

        if not mode_switches:
            # 检查整体模式变化
            eager_ratio_a = sources_a.get("eager", 0) / len(chains_a) if chains_a else 0
            eager_ratio_b = sources_b.get("eager", 0) / len(chains_b) if chains_b else 0
            compile_ratio_a = sources_a.get("torch_compile", 0) / len(chains_a) if chains_a else 0
            compile_ratio_b = sources_b.get("torch_compile", 0) / len(chains_b) if chains_b else 0

            if abs(eager_ratio_a - eager_ratio_b) < 0.3 and abs(compile_ratio_a - compile_ratio_b) < 0.3:
                return None

        evidence = []
        affected_ops = [sc.operator_name for sc in mode_switches[:20]]

        if sources_a.get("eager", 0) > 0 and sources_b.get("torch_compile", 0) > 0:
            evidence.append("版本 A 主要使用 eager 模式")
            evidence.append("版本 B 主要使用 torch.compile 模式")
        elif sources_a.get("torch_compile", 0) > 0 and sources_b.get("eager", 0) > 0:
            evidence.append("版本 A 主要使用 torch.compile 模式")
            evidence.append("版本 B 主要使用 eager 模式")

        if mode_switches:
            evidence.append(f"检测到 {len(mode_switches)} 个算子发生模式切换")

        if not evidence:
            return None

        return RootCauseFinding(
            rule_name=rule.name,
            root_cause=rule.root_cause_description,
            evidence=evidence,
            affected_operators=affected_ops,
            optimization_suggestions=rule.optimization_suggestions,
            impact=rule.impact.get("performance_impact", "medium"),
            priority=rule.impact.get("priority", "P0"),
            confidence=0.8,
        )

    def _check_communication_bottleneck(
        self,
        rule: RootCauseRule,
        diff_result: ProfilingDiff,
    ) -> Optional[RootCauseFinding]:
        """检查通信瓶颈"""
        conditions = rule.trigger_conditions

        # 检查通信相关指标
        comm_diff = diff_result.comm_diff
        if not comm_diff:
            return None

        evidence = []
        comm_ratio_increase_threshold = conditions.get("comm_ratio_increase", 10)

        # 检查通信时间变化
        if comm_diff.total_comm_time_change:
            change_pct = abs(comm_diff.total_comm_time_change.change_pct)
            if change_pct > conditions.get("comm_time_increase", 20):
                evidence.append(f"通信总时间变化: {change_pct:.1f}%")

        # 检查通信占比变化
        if diff_result.summary_diff.comm_ratio:
            comm_ratio_change = abs(diff_result.summary_diff.comm_ratio.change_pct)
            if comm_ratio_change > comm_ratio_increase_threshold:
                evidence.append(f"通信占比变化: {comm_ratio_change:.1f}%")

        # 检查通信掩盖率变化
        if comm_diff.overlap_ratio_change:
            overlap_change = comm_diff.overlap_ratio_change.change_pct
            if overlap_change < -conditions.get("overlap_ratio_decrease", 15):
                evidence.append(f"通信掩盖率下降: {abs(overlap_change):.1f}%")

        if not evidence:
            return None

        return RootCauseFinding(
            rule_name=rule.name,
            root_cause=rule.root_cause_description,
            evidence=evidence,
            affected_operators=[],
            optimization_suggestions=rule.optimization_suggestions,
            impact=rule.impact.get("performance_impact", "high"),
            priority=rule.impact.get("priority", "P0"),
            confidence=0.7,
        )


def analyze_root_causes(
    chains_a: List[HostDeviceChain],
    chains_b: List[HostDeviceChain],
    diff_result: ProfilingDiff,
) -> List[RootCauseFinding]:
    """
    便捷函数：执行根因分析

    Args:
        chains_a: 版本 A 的调用链
        chains_b: 版本 B 的调用链
        diff_result: ProfilingDiff 对比结果

    Returns:
        RootCauseFinding 列表
    """
    engine = RootCauseSkillEngine()
    return engine.analyze(chains_a, chains_b, diff_result)