"""
流水线分析器

分析 AI Core 的流水线性能指标，包括：
- 停顿原因细分到具体硬件单元
- Cube/Vector/MTE 流水线停顿分析
- 资源冲突分析
- 流水线调度优化建议
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum

from src.data_loader.aic_metrics import (
    ExtendedPipelineMetrics,
    ExtendedAICMetrics,
    PipelineBottleneckType,
)

logger = logging.getLogger(__name__)


class StallType(Enum):
    """停顿类型分类"""
    MTE = "mte"                         # MTE (Memory Transfer Engine) 停顿
    VECTOR = "vector"                   # Vector 单元停顿
    SCALAR = "scalar"                   # Scalar 单元停顿
    DEPENDENCY = "dependency"           # 依赖停顿
    MEMORY = "memory"                   # 内存停顿
    SYNC = "sync"                       # 同步停顿
    RESOURCE_CONFLICT = "conflict"      # 资源冲突
    UNKNOWN = "unknown"                 # 未知


@dataclass
class PipelineStallDetail:
    """流水线停顿详情"""
    stall_type: StallType
    stall_rate: float  # 停顿率 (0-100)
    stall_cycles: int = 0  # 停顿周期数

    # 详细信息
    description: str = ""
    related_unit: str = ""  # 相关硬件单元 (Cube/Vector/MTE/UB/L0)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "stall_type": self.stall_type.value,
            "stall_rate": round(self.stall_rate, 2),
            "stall_cycles": self.stall_cycles,
            "description": self.description,
            "related_unit": self.related_unit,
        }


@dataclass
class PipelineAnalysis:
    """流水线分析结果"""
    # 瓶颈类型
    bottleneck_type: PipelineBottleneckType
    severity: str  # "critical", "high", "medium", "low"
    score: float  # 严重程度评分 (0-100)

    # 流水线效率指标
    pipe_utilization: float = 0.0
    stall_rate: float = 0.0
    resource_conflict_ratio: float = 0.0
    issue_rate: float = 0.0
    commit_rate: float = 0.0

    # 停顿详情
    stall_details: List[PipelineStallDetail] = field(default_factory=list)

    # 资源冲突细分
    mte_conflict_rate: float = 0.0
    vec_conflict_rate: float = 0.0
    ub_conflict_rate: float = 0.0

    # 主要停顿原因
    primary_stall: Optional[PipelineStallDetail] = None

    # 优化建议
    recommendations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "bottleneck_type": self.bottleneck_type.value,
            "severity": self.severity,
            "score": round(self.score, 2),
            "pipe_utilization": round(self.pipe_utilization, 2),
            "stall_rate": round(self.stall_rate, 2),
            "resource_conflict_ratio": round(self.resource_conflict_ratio, 2),
            "issue_rate": round(self.issue_rate, 2),
            "commit_rate": round(self.commit_rate, 2),
            "stall_details": [s.to_dict() for s in self.stall_details],
            "mte_conflict_rate": round(self.mte_conflict_rate, 2),
            "vec_conflict_rate": round(self.vec_conflict_rate, 2),
            "ub_conflict_rate": round(self.ub_conflict_rate, 2),
            "primary_stall": self.primary_stall.to_dict() if self.primary_stall else None,
            "recommendations": self.recommendations,
        }


class PipelineAnalyzer:
    """
    流水线分析器

    分析 AI Core 的流水线性能，提供深度瓶颈诊断和优化建议。
    数据来源于 msprof 采集的 AIC PMU 事件。
    """

    # 阈值配置
    HIGH_STALL_RATE_THRESHOLD = 30.0       # 停顿率高于此值视为严重
    MEDIUM_STALL_RATE_THRESHOLD = 15.0     # 停顿率高于此值视为中等
    LOW_PIPE_UTIL_THRESHOLD = 50.0         # 流水线利用率低于此值视为利用率低

    def __init__(self):
        """初始化流水线分析器"""
        self._analysis_cache: Dict[str, PipelineAnalysis] = {}

    def analyze(self, metrics: ExtendedAICMetrics) -> PipelineAnalysis:
        """
        分析流水线性能

        Args:
            metrics: 扩展 AIC 指标数据

        Returns:
            PipelineAnalysis: 流水线分析结果
        """
        if not metrics.extended_pipeline:
            logger.warning("No extended pipeline metrics available for pipeline analysis")
            return PipelineAnalysis(
                bottleneck_type=PipelineBottleneckType.DEPENDENCY_STALL,
                severity="low",
                score=0.0,
                recommendations=["需要启用 msprof AIC PMU 采集以获取流水线指标"]
            )

        pipe = metrics.extended_pipeline

        # 分析停顿详情
        stall_details = self._analyze_stall_details(pipe)

        # 识别主要停顿原因
        primary_stall = self._get_primary_stall(stall_details)

        # 分析瓶颈类型
        analysis = self._identify_bottleneck(pipe, stall_details)

        # 设置详细指标
        analysis.pipe_utilization = pipe.pipe_utilization
        analysis.stall_rate = pipe.stall_rate
        analysis.resource_conflict_ratio = pipe.resource_conflict_ratio
        analysis.issue_rate = pipe.issue_rate
        analysis.commit_rate = pipe.commit_rate

        analysis.stall_details = stall_details
        analysis.primary_stall = primary_stall

        analysis.mte_conflict_rate = pipe.mte_conflict_rate
        analysis.vec_conflict_rate = pipe.vec_conflict_rate
        analysis.ub_conflict_rate = pipe.ub_conflict_rate

        # 生成优化建议
        analysis.recommendations = self._generate_recommendations(pipe, analysis)

        return analysis

    def _analyze_stall_details(
        self,
        pipe: ExtendedPipelineMetrics,
    ) -> List[PipelineStallDetail]:
        """分析停顿详情"""
        stall_details = []

        # MTE 停顿
        if pipe.mte_stall_rate > 5:
            stall_details.append(PipelineStallDetail(
                stall_type=StallType.MTE,
                stall_rate=pipe.mte_stall_rate,
                description="Memory Transfer Engine 停顿，可能由于数据传输未就绪",
                related_unit="MTE",
            ))

        # Vector 停顿
        if pipe.vec_stall_rate > 5:
            stall_details.append(PipelineStallDetail(
                stall_type=StallType.VECTOR,
                stall_rate=pipe.vec_stall_rate,
                description="Vector 单元停顿，可能由于指令依赖或资源冲突",
                related_unit="Vector",
            ))

        # Scalar 停顿
        if pipe.scalar_stall_rate > 5:
            stall_details.append(PipelineStallDetail(
                stall_type=StallType.SCALAR,
                stall_rate=pipe.scalar_stall_rate,
                description="Scalar 单元停顿，可能由于控制流依赖",
                related_unit="Scalar",
            ))

        # 依赖停顿
        if pipe.dependency_stall_rate > 5:
            stall_details.append(PipelineStallDetail(
                stall_type=StallType.DEPENDENCY,
                stall_rate=pipe.dependency_stall_rate,
                description="数据依赖导致停顿",
                related_unit="Pipeline",
            ))

        # 内存停顿
        if pipe.memory_stall_rate > 5:
            stall_details.append(PipelineStallDetail(
                stall_type=StallType.MEMORY,
                stall_rate=pipe.memory_stall_rate,
                description="内存访问导致停顿",
                related_unit="Memory",
            ))

        # 同步停顿
        if pipe.sync_stall_rate > 5:
            stall_details.append(PipelineStallDetail(
                stall_type=StallType.SYNC,
                stall_rate=pipe.sync_stall_rate,
                description="同步操作导致停顿",
                related_unit="Sync",
            ))

        return sorted(stall_details, key=lambda x: x.stall_rate, reverse=True)

    def _get_primary_stall(
        self,
        stall_details: List[PipelineStallDetail],
    ) -> Optional[PipelineStallDetail]:
        """获取主要停顿原因"""
        if not stall_details:
            return None
        return stall_details[0]

    def _identify_bottleneck(
        self,
        pipe: ExtendedPipelineMetrics,
        stall_details: List[PipelineStallDetail],
    ) -> PipelineAnalysis:
        """识别流水线瓶颈类型"""
        score = 0.0
        bottleneck_type = PipelineBottleneckType.DEPENDENCY_STALL
        severity = "low"

        # 基于停顿率评分
        if pipe.stall_rate > self.HIGH_STALL_RATE_THRESHOLD:
            score += (pipe.stall_rate - self.HIGH_STALL_RATE_THRESHOLD) * 1.5
            severity = "critical"
        elif pipe.stall_rate > self.MEDIUM_STALL_RATE_THRESHOLD:
            score += (pipe.stall_rate - self.MEDIUM_STALL_RATE_THRESHOLD)
            severity = "medium"

        # 基于流水线利用率评分
        if pipe.pipe_utilization < self.LOW_PIPE_UTIL_THRESHOLD:
            score += (100 - pipe.pipe_utilization) * 0.5
            if score > 30:
                severity = "high"

        # 确定瓶颈类型
        if stall_details:
            primary = stall_details[0]
            if primary.stall_type == StallType.MTE:
                bottleneck_type = PipelineBottleneckType.MEMORY_STALL
            elif primary.stall_type == StallType.DEPENDENCY:
                bottleneck_type = PipelineBottleneckType.DEPENDENCY_STALL
            elif primary.stall_type in (StallType.MEMORY, StallType.SYNC):
                bottleneck_type = PipelineBottleneckType.MEMORY_STALL
            elif pipe.resource_conflict_ratio > 30:
                bottleneck_type = PipelineBottleneckType.RESOURCE_CONFLICT

        return PipelineAnalysis(
            bottleneck_type=bottleneck_type,
            severity=severity,
            score=score,
        )

    def _generate_recommendations(
        self,
        pipe: ExtendedPipelineMetrics,
        analysis: PipelineAnalysis,
    ) -> List[str]:
        """生成优化建议"""
        recommendations = []

        # MTE 停顿
        if pipe.mte_stall_rate > 15:
            recommendations.extend([
                "优化 MTE 单元的数据调度，减少数据传输等待",
                "使用数据预取技术隐藏访存延迟",
                "考虑使用双缓冲技术以重叠计算和数据传输",
            ])

        # 依赖停顿
        if pipe.dependency_stall_rate > 15:
            recommendations.extend([
                "优化指令调度以减少依赖等待",
                "考虑循环展开以提高指令级并行",
                "优化算法以减少数据依赖",
            ])

        # 资源冲突
        if pipe.resource_conflict_ratio > 30:
            recommendations.extend([
                "调整计算资源分配，减少资源冲突",
                "优化流水线调度策略",
                "考虑使用不同的 tile 配置以平衡资源使用",
            ])

        # 流水线利用率低
        if pipe.pipe_utilization < self.LOW_PIPE_UTIL_THRESHOLD:
            recommendations.extend([
                "增加计算密度以提高流水线利用率",
                "减少不必要的同步操作",
                "优化指令调度以提高流水线吞吐",
            ])

        # 指令发射率低
        if pipe.issue_rate < 50:
            recommendations.extend([
                "检查是否存在控制流瓶颈",
                "优化分支预测以减少流水线停顿",
                "考虑使用无分支算法替代条件分支",
            ])

        # Vector/Scalar 停顿
        if pipe.vec_stall_rate > 15:
            recommendations.extend([
                "优化 Vector 单元的指令序列",
                "检查是否存在 Vector 单元的资源冲突",
            ])

        if pipe.scalar_stall_rate > 15:
            recommendations.extend([
                "减少 Scalar 指令的使用",
                "将标量计算向量化",
            ])

        return recommendations

    def analyze_batch(
        self,
        metrics_list: List[ExtendedAICMetrics],
    ) -> Tuple[List[PipelineAnalysis], Dict[str, Any]]:
        """
        批量分析多个算子的流水线性能

        Args:
            metrics_list: 扩展 AIC 指标列表

        Returns:
            (分析结果列表, 汇总统计)
        """
        analyses = []

        stall_type_counts = {
            "mte": 0,
            "vector": 0,
            "scalar": 0,
            "dependency": 0,
            "memory": 0,
            "sync": 0,
        }

        for metrics in metrics_list:
            if not metrics.extended_pipeline:
                continue

            analysis = self.analyze(metrics)
            analyses.append(analysis)

            # 统计停顿类型
            for stall in analysis.stall_details:
                if stall.stall_type.value in stall_type_counts:
                    stall_type_counts[stall.stall_type.value] += 1

        # 计算汇总统计
        summary = {
            "total_analyzed": len(analyses),
            "critical_count": sum(1 for a in analyses if a.severity == "critical"),
            "high_count": sum(1 for a in analyses if a.severity == "high"),
            "medium_count": sum(1 for a in analyses if a.severity == "medium"),
            "avg_pipe_util": 0.0,
            "avg_stall_rate": 0.0,
            "stall_type_distribution": stall_type_counts,
        }

        if analyses:
            summary["avg_pipe_util"] = sum(a.pipe_utilization for a in analyses) / len(analyses)
            summary["avg_stall_rate"] = sum(a.stall_rate for a in analyses) / len(analyses)

        return analyses, summary


def analyze_pipeline(
    metrics: ExtendedAICMetrics,
) -> PipelineAnalysis:
    """
    便捷函数：分析流水线性能

    Args:
        metrics: 扩展 AIC 指标数据

    Returns:
        PipelineAnalysis: 流水线分析结果
    """
    analyzer = PipelineAnalyzer()
    return analyzer.analyze(metrics)