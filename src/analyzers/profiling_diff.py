"""
Profiling 多层级差异分析引擎

对两个 Profiling 数据进行 5 个层级的深度对比：
1. Summary 级 - 整体指标对比
2. Timeline 级 - Step 级别时序对比
3. Operator 级 - 算子性能对比
4. Communication 级 - 通信模式对比
5. Memory 级 - 内存使用对比
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field

from src.data_loader.profiling_loader import ProfilingLoader, ProfilingInfo
from src.data_loader.data_summarizer import DataSummarizer, ProfilingSummary

logger = logging.getLogger(__name__)


# ============================================================
# 差异数据结构
# ============================================================

@dataclass
class MetricChange:
    """单个指标的变化"""
    name: str
    label: str                 # 中文名称
    value_a: float
    value_b: float
    change_abs: float          # 绝对变化量
    change_pct: float          # 百分比变化
    is_improvement: bool       # 变化是否是改进
    unit: str = ""
    significance: str = "low"  # low / medium / high

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "label": self.label,
            "value_a": self.value_a,
            "value_b": self.value_b,
            "change_abs": round(self.change_abs, 4),
            "change_pct": round(self.change_pct, 2),
            "is_improvement": self.is_improvement,
            "unit": self.unit,
            "significance": self.significance,
        }


@dataclass
class SummaryDiff:
    """Summary 级差异"""
    step_time: Optional[MetricChange] = None
    mfu: Optional[MetricChange] = None
    compute_ratio: Optional[MetricChange] = None
    comm_ratio: Optional[MetricChange] = None
    idle_ratio: Optional[MetricChange] = None
    overlap_ratio: Optional[MetricChange] = None
    bubble_ratio: Optional[MetricChange] = None
    all_changes: List[MetricChange] = field(default_factory=list)

    # 整体评估
    is_improved: bool = False
    improvement_count: int = 0
    regression_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_improved": self.is_improved,
            "improvement_count": self.improvement_count,
            "regression_count": self.regression_count,
            "changes": [c.to_dict() for c in self.all_changes],
        }


@dataclass
class OperatorChange:
    """单个算子的变化"""
    name: str
    dur_a: float               # Profiling A 中的耗时 (us)
    dur_b: float               # Profiling B 中的耗时 (us)
    change_pct: float          # 百分比变化
    change_abs: float          # 绝对变化 (us)
    is_improvement: bool
    is_new: bool = False       # 新出现的算子
    is_removed: bool = False   # 消失的算子

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "dur_a_ms": round(self.dur_a / 1000, 3),
            "dur_b_ms": round(self.dur_b / 1000, 3),
            "change_pct": round(self.change_pct, 2),
            "change_abs_ms": round(self.change_abs / 1000, 3),
            "is_improvement": self.is_improvement,
            "is_new": self.is_new,
            "is_removed": self.is_removed,
        }


@dataclass
class OperatorDiff:
    """Operator 级差异"""
    top_regressions: List[OperatorChange] = field(default_factory=list)
    top_improvements: List[OperatorChange] = field(default_factory=list)
    new_operators: List[OperatorChange] = field(default_factory=list)
    removed_operators: List[OperatorChange] = field(default_factory=list)
    total_operator_count_a: int = 0
    total_operator_count_b: int = 0
    common_operator_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "top_regressions": [o.to_dict() for o in self.top_regressions],
            "top_improvements": [o.to_dict() for o in self.top_improvements],
            "new_operators": [o.to_dict() for o in self.new_operators],
            "removed_operators": [o.to_dict() for o in self.removed_operators],
            "total_operator_count_a": self.total_operator_count_a,
            "total_operator_count_b": self.total_operator_count_b,
            "common_operator_count": self.common_operator_count,
        }


@dataclass
class TimelineDiff:
    """Timeline 级差异"""
    # Step 级别的统计
    step_time_std_a: float = 0.0   # A 的 step 时间标准差
    step_time_std_b: float = 0.0   # B 的 step 时间标准差
    step_time_cv_a: float = 0.0    # A 的变异系数
    step_time_cv_b: float = 0.0    # B 的变异系数
    stability_improved: bool = False

    # 各阶段时间的变化
    compute_time_change: Optional[MetricChange] = None
    comm_time_change: Optional[MetricChange] = None
    free_time_change: Optional[MetricChange] = None

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "step_time_std_a_ms": round(self.step_time_std_a / 1000, 3),
            "step_time_std_b_ms": round(self.step_time_std_b / 1000, 3),
            "step_time_cv_a": round(self.step_time_cv_a, 4),
            "step_time_cv_b": round(self.step_time_cv_b, 4),
            "stability_improved": self.stability_improved,
        }
        if self.compute_time_change:
            result["compute_time_change"] = self.compute_time_change.to_dict()
        if self.comm_time_change:
            result["comm_time_change"] = self.comm_time_change.to_dict()
        if self.free_time_change:
            result["free_time_change"] = self.free_time_change.to_dict()
        return result


@dataclass
class CommDiff:
    """Communication 级差异"""
    total_comm_time_change: Optional[MetricChange] = None
    overlap_ratio_change: Optional[MetricChange] = None
    comm_pattern_changes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        result = {"comm_pattern_changes": self.comm_pattern_changes}
        if self.total_comm_time_change:
            result["total_comm_time_change"] = self.total_comm_time_change.to_dict()
        if self.overlap_ratio_change:
            result["overlap_ratio_change"] = self.overlap_ratio_change.to_dict()
        return result


@dataclass
class MemoryDiff:
    """Memory 级差异"""
    peak_memory_change: Optional[MetricChange] = None
    memory_pattern_changes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        result = {"memory_pattern_changes": self.memory_pattern_changes}
        if self.peak_memory_change:
            result["peak_memory_change"] = self.peak_memory_change.to_dict()
        return result


@dataclass
class ProfilingDiff:
    """完整的 Profiling 差异结果"""
    summary_diff: SummaryDiff = field(default_factory=SummaryDiff)
    timeline_diff: Optional[TimelineDiff] = None
    operator_diff: Optional[OperatorDiff] = None
    comm_diff: Optional[CommDiff] = None
    memory_diff: Optional[MemoryDiff] = None

    # 全局判断
    overall_verdict: str = ""      # improved / degraded / mixed / unchanged
    primary_changes: List[str] = field(default_factory=list)  # 主要变化描述

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "overall_verdict": self.overall_verdict,
            "primary_changes": self.primary_changes,
            "summary_diff": self.summary_diff.to_dict(),
        }
        if self.timeline_diff:
            result["timeline_diff"] = self.timeline_diff.to_dict()
        if self.operator_diff:
            result["operator_diff"] = self.operator_diff.to_dict()
        if self.comm_diff:
            result["comm_diff"] = self.comm_diff.to_dict()
        if self.memory_diff:
            result["memory_diff"] = self.memory_diff.to_dict()
        return result

    def to_prompt_text(self) -> str:
        """转为适合 LLM 分析的文本"""
        lines = ["## Profiling 差异分析结果", ""]

        # 1. 整体判断
        verdict_map = {
            "improved": "性能提升",
            "degraded": "性能劣化",
            "mixed": "喜忧参半",
            "unchanged": "基本不变",
        }
        lines.append(f"**整体判断**: {verdict_map.get(self.overall_verdict, self.overall_verdict)}")
        lines.append("")

        if self.primary_changes:
            lines.append("**主要变化:**")
            for change in self.primary_changes:
                lines.append(f"- {change}")
            lines.append("")

        # 2. Summary 差异
        lines.append("### Summary 级差异")
        for change in self.summary_diff.all_changes:
            icon = "↓" if change.is_improvement else "↑"
            if change.name in ("step_time", "comm_ratio", "idle_ratio", "bubble_ratio"):
                icon = "↓" if change.change_pct < 0 else "↑"
                good = change.change_pct < 0
            else:
                icon = "↑" if change.change_pct > 0 else "↓"
                good = change.change_pct > 0
            status = "✅" if good else "⚠️"
            lines.append(
                f"- {status} {change.label}: "
                f"{change.value_a:.4g}{change.unit} → {change.value_b:.4g}{change.unit} "
                f"({icon}{change.change_pct:+.1f}%)"
            )
        lines.append("")

        # 3. Operator 差异
        if self.operator_diff:
            lines.append("### Operator 级差异")
            if self.operator_diff.top_regressions:
                lines.append("**Top 劣化算子:**")
                for op in self.operator_diff.top_regressions[:10]:
                    lines.append(
                        f"- {op.name}: {op.dur_a / 1000:.2f}ms → {op.dur_b / 1000:.2f}ms "
                        f"({op.change_pct:+.1f}%)"
                    )
            if self.operator_diff.top_improvements:
                lines.append("**Top 改善算子:**")
                for op in self.operator_diff.top_improvements[:10]:
                    lines.append(
                        f"- {op.name}: {op.dur_a / 1000:.2f}ms → {op.dur_b / 1000:.2f}ms "
                        f"({op.change_pct:+.1f}%)"
                    )
            if self.operator_diff.new_operators:
                lines.append(f"**新增算子**: {len(self.operator_diff.new_operators)} 个")
            if self.operator_diff.removed_operators:
                lines.append(f"**消失算子**: {len(self.operator_diff.removed_operators)} 个")
            lines.append("")

        # 4. Timeline 差异
        if self.timeline_diff:
            lines.append("### Timeline 级差异")
            td = self.timeline_diff
            stability = "更稳定" if td.stability_improved else "更不稳定"
            lines.append(f"- Step 稳定性: 变异系数 {td.step_time_cv_a:.4f} → {td.step_time_cv_b:.4f} ({stability})")
            lines.append("")

        # 5. Communication 差异
        if self.comm_diff:
            lines.append("### Communication 级差异")
            if self.comm_diff.total_comm_time_change:
                c = self.comm_diff.total_comm_time_change
                lines.append(f"- 通信总时间: {c.value_a / 1000:.2f}ms → {c.value_b / 1000:.2f}ms ({c.change_pct:+.1f}%)")
            if self.comm_diff.overlap_ratio_change:
                c = self.comm_diff.overlap_ratio_change
                lines.append(f"- 通信掩盖率: {c.value_a:.1f}% → {c.value_b:.1f}% ({c.change_pct:+.1f}%)")
            for pattern in self.comm_diff.comm_pattern_changes:
                lines.append(f"- {pattern}")
            lines.append("")

        return "\n".join(lines)


# ============================================================
# 差异分析引擎
# ============================================================

class ProfilingDiffEngine:
    """
    Profiling 多层级差异分析引擎

    Usage:
        engine = ProfilingDiffEngine()
        diff = engine.compute(
            summary_a, summary_b,
            loader_a=loader_a, loader_b=loader_b,
        )
    """

    # 显著性阈值
    SIGNIFICANCE_HIGH = 10.0    # 变化超过 10% 为高显著性
    SIGNIFICANCE_MEDIUM = 5.0   # 变化超过 5% 为中显著性

    def __init__(self):
        pass

    def compute(
        self,
        summary_a: ProfilingSummary,
        summary_b: ProfilingSummary,
        loader_a: Optional[ProfilingLoader] = None,
        loader_b: Optional[ProfilingLoader] = None,
        operators_a: Optional[List[Dict[str, Any]]] = None,
        operators_b: Optional[List[Dict[str, Any]]] = None,
    ) -> ProfilingDiff:
        """
        计算两个 Profiling 的完整差异

        Args:
            summary_a: Profiling A（基准）的摘要
            summary_b: Profiling B（当前）的摘要
            loader_a: Profiling A 的数据加载器（可选，用于深度分析）
            loader_b: Profiling B 的数据加载器（可选，用于深度分析）
            operators_a: Profiling A 的算子列表（可选）
            operators_b: Profiling B 的算子列表（可选）

        Returns:
            ProfilingDiff
        """
        diff = ProfilingDiff()

        # Level 1: Summary Diff (always)
        diff.summary_diff = self._compute_summary_diff(summary_a, summary_b)

        # Level 2: Timeline Diff
        diff.timeline_diff = self._compute_timeline_diff(summary_a, summary_b, loader_a, loader_b)

        # Level 3: Operator Diff
        diff.operator_diff = self._compute_operator_diff(summary_a, summary_b, operators_a, operators_b)

        # Level 4: Communication Diff
        diff.comm_diff = self._compute_comm_diff(summary_a, summary_b, loader_a, loader_b)

        # Level 5: Memory Diff
        diff.memory_diff = self._compute_memory_diff(loader_a, loader_b)

        # 全局判断
        diff.overall_verdict = self._determine_verdict(diff)
        diff.primary_changes = self._extract_primary_changes(diff)

        return diff

    def _compute_summary_diff(
        self,
        summary_a: ProfilingSummary,
        summary_b: ProfilingSummary,
    ) -> SummaryDiff:
        """计算 Summary 级差异"""
        sd = SummaryDiff()
        changes = []

        # Step 时间（越小越好）
        if summary_a.avg_step_time > 0:
            sd.step_time = self._make_change(
                "step_time", "Step 时间",
                summary_a.avg_step_time / 1000, summary_b.avg_step_time / 1000,
                unit="ms", lower_is_better=True,
            )
            changes.append(sd.step_time)

        # 计算时间占比（越大越好）
        total_a = summary_a.avg_compute_time + summary_a.avg_comm_time + summary_a.avg_free_time
        total_b = summary_b.avg_compute_time + summary_b.avg_comm_time + summary_b.avg_free_time

        if total_a > 0 and total_b > 0:
            cr_a = summary_a.avg_compute_time / total_a * 100
            cr_b = summary_b.avg_compute_time / total_b * 100
            sd.compute_ratio = self._make_change(
                "compute_ratio", "计算占比",
                cr_a, cr_b, unit="%", lower_is_better=False,
            )
            changes.append(sd.compute_ratio)

            # 通信占比（越小越好）
            comm_a = summary_a.avg_comm_time / total_a * 100
            comm_b = summary_b.avg_comm_time / total_b * 100
            sd.comm_ratio = self._make_change(
                "comm_ratio", "通信占比",
                comm_a, comm_b, unit="%", lower_is_better=True,
            )
            changes.append(sd.comm_ratio)

            # 空闲占比（越小越好）
            idle_a = summary_a.avg_free_time / total_a * 100
            idle_b = summary_b.avg_free_time / total_b * 100
            sd.idle_ratio = self._make_change(
                "idle_ratio", "空闲占比",
                idle_a, idle_b, unit="%", lower_is_better=True,
            )
            changes.append(sd.idle_ratio)

        # Overlap 率（越大越好）
        overlap_a = summary_a.overlap_metrics.overlap_ratio
        overlap_b = summary_b.overlap_metrics.overlap_ratio
        if overlap_a > 0 or overlap_b > 0:
            sd.overlap_ratio = self._make_change(
                "overlap_ratio", "通信掩盖率",
                overlap_a, overlap_b, unit="%", lower_is_better=False,
            )
            changes.append(sd.overlap_ratio)

        # Bubble 率（越小越好）
        if summary_a.bubble_ratio > 0 or summary_b.bubble_ratio > 0:
            sd.bubble_ratio = self._make_change(
                "bubble_ratio", "Bubble 占比",
                summary_a.bubble_ratio, summary_b.bubble_ratio,
                unit="%", lower_is_better=True,
            )
            changes.append(sd.bubble_ratio)

        sd.all_changes = changes
        sd.improvement_count = sum(1 for c in changes if c.is_improvement and abs(c.change_pct) > 1)
        sd.regression_count = sum(1 for c in changes if not c.is_improvement and abs(c.change_pct) > 1)
        sd.is_improved = sd.improvement_count > sd.regression_count

        return sd

    def _compute_timeline_diff(
        self,
        summary_a: ProfilingSummary,
        summary_b: ProfilingSummary,
        loader_a: Optional[ProfilingLoader],
        loader_b: Optional[ProfilingLoader],
    ) -> TimelineDiff:
        """计算 Timeline 级差异"""
        td = TimelineDiff()

        # 从 sample_steps 计算稳定性
        if summary_a.sample_steps and summary_b.sample_steps:
            times_a = [s.computing + s.communication + s.free for s in summary_a.sample_steps]
            times_b = [s.computing + s.communication + s.free for s in summary_b.sample_steps]

            if times_a and times_b:
                import statistics
                mean_a = statistics.mean(times_a) if times_a else 0
                mean_b = statistics.mean(times_b) if times_b else 0
                std_a = statistics.stdev(times_a) if len(times_a) > 1 else 0
                std_b = statistics.stdev(times_b) if len(times_b) > 1 else 0

                td.step_time_std_a = std_a
                td.step_time_std_b = std_b
                td.step_time_cv_a = std_a / mean_a if mean_a > 0 else 0
                td.step_time_cv_b = std_b / mean_b if mean_b > 0 else 0
                td.stability_improved = td.step_time_cv_b < td.step_time_cv_a

        # 各阶段绝对时间变化
        if summary_a.avg_compute_time > 0:
            td.compute_time_change = self._make_change(
                "compute_time", "计算时间",
                summary_a.avg_compute_time / 1000, summary_b.avg_compute_time / 1000,
                unit="ms", lower_is_better=True,
            )

        if summary_a.avg_comm_time > 0:
            td.comm_time_change = self._make_change(
                "comm_time", "通信时间",
                summary_a.avg_comm_time / 1000, summary_b.avg_comm_time / 1000,
                unit="ms", lower_is_better=True,
            )

        if summary_a.avg_free_time > 0:
            td.free_time_change = self._make_change(
                "free_time", "空闲时间",
                summary_a.avg_free_time / 1000, summary_b.avg_free_time / 1000,
                unit="ms", lower_is_better=True,
            )

        return td

    def _compute_operator_diff(
        self,
        summary_a: ProfilingSummary,
        summary_b: ProfilingSummary,
        operators_a: Optional[List[Dict[str, Any]]],
        operators_b: Optional[List[Dict[str, Any]]],
    ) -> OperatorDiff:
        """计算 Operator 级差异"""
        od = OperatorDiff()

        # 使用传入的算子列表，或 fallback 到 top_operators
        ops_a = operators_a if operators_a else summary_a.top_operators
        ops_b = operators_b if operators_b else summary_b.top_operators

        if not ops_a and not ops_b:
            return od

        # 构建算子字典 {name: dur}
        dict_a = {}
        for op in ops_a:
            name = op.get("name", "")
            dur = op.get("dur", 0)
            if name:
                # 同名算子聚合
                dict_a[name] = dict_a.get(name, 0) + dur

        dict_b = {}
        for op in ops_b:
            name = op.get("name", "")
            dur = op.get("dur", 0)
            if name:
                dict_b[name] = dict_b.get(name, 0) + dur

        od.total_operator_count_a = len(dict_a)
        od.total_operator_count_b = len(dict_b)

        # 找出共同算子
        common_names = set(dict_a.keys()) & set(dict_b.keys())
        od.common_operator_count = len(common_names)

        # 计算共同算子的变化
        regressions = []
        improvements = []

        for name in common_names:
            dur_a = dict_a[name]
            dur_b = dict_b[name]
            change_abs = dur_b - dur_a
            change_pct = (change_abs / dur_a * 100) if dur_a > 0 else 0

            oc = OperatorChange(
                name=name,
                dur_a=dur_a,
                dur_b=dur_b,
                change_pct=change_pct,
                change_abs=change_abs,
                is_improvement=change_pct < -1,  # 耗时减少超过 1%
            )

            if change_pct > 1:
                regressions.append(oc)
            elif change_pct < -1:
                improvements.append(oc)

        # 排序（按绝对时间变化排序，影响最大的排在前面）
        od.top_regressions = sorted(regressions, key=lambda x: x.change_abs, reverse=True)[:20]
        od.top_improvements = sorted(improvements, key=lambda x: x.change_abs)[:20]

        # 新增的算子
        new_names = set(dict_b.keys()) - set(dict_a.keys())
        od.new_operators = [
            OperatorChange(
                name=name, dur_a=0, dur_b=dict_b[name],
                change_pct=100, change_abs=dict_b[name],
                is_improvement=False, is_new=True,
            )
            for name in sorted(new_names, key=lambda n: dict_b[n], reverse=True)[:20]
        ]

        # 消失的算子
        removed_names = set(dict_a.keys()) - set(dict_b.keys())
        od.removed_operators = [
            OperatorChange(
                name=name, dur_a=dict_a[name], dur_b=0,
                change_pct=-100, change_abs=-dict_a[name],
                is_improvement=True, is_removed=True,
            )
            for name in sorted(removed_names, key=lambda n: dict_a[n], reverse=True)[:20]
        ]

        return od

    def _compute_comm_diff(
        self,
        summary_a: ProfilingSummary,
        summary_b: ProfilingSummary,
        loader_a: Optional[ProfilingLoader],
        loader_b: Optional[ProfilingLoader],
    ) -> CommDiff:
        """计算 Communication 级差异"""
        cd = CommDiff()

        # 通信总时间变化
        if summary_a.avg_comm_time > 0 or summary_b.avg_comm_time > 0:
            cd.total_comm_time_change = self._make_change(
                "total_comm_time", "通信总时间",
                summary_a.avg_comm_time, summary_b.avg_comm_time,
                unit="us", lower_is_better=True,
            )

        # 通信掩盖率变化
        overlap_a = summary_a.overlap_metrics.overlap_ratio
        overlap_b = summary_b.overlap_metrics.overlap_ratio
        if overlap_a > 0 or overlap_b > 0:
            cd.overlap_ratio_change = self._make_change(
                "overlap_ratio", "通信掩盖率",
                overlap_a, overlap_b,
                unit="%", lower_is_better=False,
            )

        # 通信模式变化分析
        patterns = []

        # 检查 comm_not_overlapped 变化
        cno_a = summary_a.overlap_metrics.comm_not_overlapped
        cno_b = summary_b.overlap_metrics.comm_not_overlapped
        if cno_a > 0 and cno_b > 0:
            cno_change = (cno_b - cno_a) / cno_a * 100 if cno_a > 0 else 0
            if abs(cno_change) > 5:
                direction = "增加" if cno_change > 0 else "减少"
                patterns.append(f"未掩盖通信时间{direction} {abs(cno_change):.1f}%")

        # 检查 Rank 数变化
        if summary_a.rank_count != summary_b.rank_count:
            patterns.append(
                f"并行卡数从 {summary_a.rank_count} 变为 {summary_b.rank_count}，通信拓扑可能变化"
            )

        cd.comm_pattern_changes = patterns
        return cd

    def _compute_memory_diff(
        self,
        loader_a: Optional[ProfilingLoader],
        loader_b: Optional[ProfilingLoader],
    ) -> MemoryDiff:
        """计算 Memory 级差异"""
        md = MemoryDiff()

        # 尝试从 loader 获取内存信息
        if loader_a and loader_b:
            try:
                hw_a = loader_a.get_hardware_info()
                hw_b = loader_b.get_hardware_info()
                if hw_a and hw_b:
                    mem_a = hw_a.get("peak_memory_gb", 0)
                    mem_b = hw_b.get("peak_memory_gb", 0)
                    if mem_a > 0 or mem_b > 0:
                        md.peak_memory_change = self._make_change(
                            "peak_memory", "峰值内存",
                            mem_a, mem_b,
                            unit="GB", lower_is_better=True,
                        )
            except Exception as e:
                logger.debug(f"Failed to get memory info: {e}")

        return md

    def _make_change(
        self,
        name: str,
        label: str,
        value_a: float,
        value_b: float,
        unit: str = "",
        lower_is_better: bool = True,
    ) -> MetricChange:
        """创建一个 MetricChange 对象"""
        change_abs = value_b - value_a
        change_pct = (change_abs / value_a * 100) if value_a != 0 else (100.0 if value_b != 0 else 0.0)

        if lower_is_better:
            is_improvement = change_pct < -1  # 降低超过 1% 视为改进
        else:
            is_improvement = change_pct > 1   # 升高超过 1% 视为改进

        significance = "low"
        if abs(change_pct) >= self.SIGNIFICANCE_HIGH:
            significance = "high"
        elif abs(change_pct) >= self.SIGNIFICANCE_MEDIUM:
            significance = "medium"

        return MetricChange(
            name=name,
            label=label,
            value_a=value_a,
            value_b=value_b,
            change_abs=change_abs,
            change_pct=change_pct,
            is_improvement=is_improvement,
            unit=unit,
            significance=significance,
        )

    def _determine_verdict(self, diff: ProfilingDiff) -> str:
        """确定整体判断"""
        sd = diff.summary_diff

        # 基于 step_time 的变化做主判断
        if sd.step_time and abs(sd.step_time.change_pct) > 1:
            if sd.step_time.is_improvement:
                if sd.regression_count == 0:
                    return "improved"
                return "improved" if sd.improvement_count > sd.regression_count else "mixed"
            else:
                if sd.improvement_count == 0:
                    return "degraded"
                return "degraded" if sd.regression_count > sd.improvement_count else "mixed"

        if sd.improvement_count > 0 and sd.regression_count > 0:
            return "mixed"
        if sd.improvement_count > 0:
            return "improved"
        if sd.regression_count > 0:
            return "degraded"
        return "unchanged"

    def _extract_primary_changes(self, diff: ProfilingDiff) -> List[str]:
        """提取主要变化描述"""
        changes = []

        sd = diff.summary_diff
        if sd.step_time:
            direction = "缩短" if sd.step_time.change_pct < 0 else "增加"
            changes.append(
                f"Step 时间{direction} {abs(sd.step_time.change_pct):.1f}% "
                f"({sd.step_time.value_a:.2f}ms → {sd.step_time.value_b:.2f}ms)"
            )

        if sd.compute_ratio:
            if abs(sd.compute_ratio.change_pct) > 2:
                direction = "提升" if sd.compute_ratio.change_pct > 0 else "下降"
                changes.append(f"计算占比{direction} {abs(sd.compute_ratio.change_pct):.1f}%")

        if sd.idle_ratio:
            if abs(sd.idle_ratio.change_pct) > 5:
                direction = "减少" if sd.idle_ratio.change_pct < 0 else "增加"
                changes.append(f"空闲时间{direction} {abs(sd.idle_ratio.change_pct):.1f}%")

        # Operator 级别
        od = diff.operator_diff
        if od and od.top_regressions:
            top = od.top_regressions[0]
            changes.append(
                f"算子 {top.name} 耗时增加 {top.change_pct:.1f}% "
                f"(+{top.change_abs / 1000:.2f}ms)"
            )

        return changes[:5]
