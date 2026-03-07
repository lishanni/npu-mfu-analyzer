"""
指令级分析器

分析 AI Core 的指令级性能指标，包括：
- Cube/Vector/Scalar 指令混合比
- 指令执行效率
- 指令瓶颈定位
- 指令调度优化建议
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum

from src.data_loader.aic_metrics import (
    ExtendedArithmeticUtilization,
    ExtendedAICMetrics,
    ComputeBottleneckType,
)

logger = logging.getLogger(__name__)


class InstructionPattern(Enum):
    """指令模式分类"""
    CUBE_DOMINANT = "cube_dominant"       # Cube 主导
    VECTOR_DOMINANT = "vector_dominant"   # Vector 主导
    SCALAR_HEAVY = "scalar_heavy"         # Scalar 过重
    BALANCED = "balanced"                 # 均衡
    UNKNOWN = "unknown"                   # 未知


@dataclass
class InstructionBottleneck:
    """指令瓶颈分析结果"""
    bottleneck_type: ComputeBottleneckType
    severity: str  # "critical", "high", "medium", "low"
    score: float  # 严重程度评分 (0-100)

    # 详细分析
    cube_utilization: float = 0.0
    vector_utilization: float = 0.0
    scalar_utilization: float = 0.0
    instruction_mix: Dict[str, float] = field(default_factory=dict)

    # 优化建议
    recommendations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "bottleneck_type": self.bottleneck_type.value,
            "severity": self.severity,
            "score": round(self.score, 2),
            "cube_utilization": round(self.cube_utilization, 2),
            "vector_utilization": round(self.vector_utilization, 2),
            "scalar_utilization": round(self.scalar_utilization, 2),
            "instruction_mix": self.instruction_mix,
            "recommendations": self.recommendations,
        }


class InstructionAnalyzer:
    """
    指令级分析器

    分析 AI Core 的指令级性能，提供深度瓶颈诊断和优化建议。
    数据来源于 msprof 采集的 AIC PMU 事件。
    """

    # 阈值配置
    CUBE_UNDERUTILIZED_THRESHOLD = 40.0  # Cube 利用率低于此值视为利用率低
    LOW_ISSUE_RATE_THRESHOLD = 50.0      # 指令发射率低于此值视为发射率低
    SCALAR_HEAVY_THRESHOLD = 0.3          # Scalar 指令占比超过此值视为 Scalar 过重

    def __init__(self):
        """初始化指令分析器"""
        self._analysis_cache: Dict[str, InstructionBottleneck] = {}

    def analyze(self, metrics: ExtendedAICMetrics) -> InstructionBottleneck:
        """
        分析指令级性能

        Args:
            metrics: 扩展 AIC 指标数据

        Returns:
            InstructionBottleneck: 指令瓶颈分析结果
        """
        if not metrics.extended_arithmetic:
            logger.warning("No extended arithmetic metrics available for instruction analysis")
            return InstructionBottleneck(
                bottleneck_type=ComputeBottleneckType.INSTRUCTION_IMBALANCE,
                severity="low",
                score=0.0,
                recommendations=["需要启用 msprof AIC PMU 采集以获取指令级指标"]
            )

        arith = metrics.extended_arithmetic
        mix = arith.get_instruction_mix()
        pattern = arith.classify_compute_pattern()

        # 分析瓶颈类型
        bottleneck = self._identify_bottleneck(arith, pattern)

        # 设置详细指标
        bottleneck.cube_utilization = arith.cube_utilization
        bottleneck.vector_utilization = arith.vector_utilization
        bottleneck.scalar_utilization = arith.scalar_utilization
        bottleneck.instruction_mix = {
            "cube": mix["cube"],
            "vector": mix["vector"],
            "scalar": mix["scalar"],
        }

        # 生成优化建议
        bottleneck.recommendations = self._generate_recommendations(
            arith, pattern, bottleneck
        )

        return bottleneck

    def _identify_bottleneck(
        self,
        arith: ExtendedArithmeticUtilization,
        pattern: InstructionPattern,
    ) -> InstructionBottleneck:
        """识别指令瓶颈类型"""
        score = 0.0
        bottleneck_type = ComputeBottleneckType.INSTRUCTION_IMBALANCE
        severity = "low"

        # Cube 利用率低
        if arith.cube_utilization < self.CUBE_UNDERUTILIZED_THRESHOLD:
            score += (100 - arith.cube_utilization) * 0.6
            if arith.cube_utilization < 20:
                bottleneck_type = ComputeBottleneckType.CUBE_UNDERUTILIZED
                severity = "critical"
            else:
                severity = "high"

        # 指令发射率低
        if arith.instruction_issue_rate < self.LOW_ISSUE_RATE_THRESHOLD:
            score += (100 - arith.instruction_issue_rate) * 0.4
            if score > 30:
                bottleneck_type = ComputeBottleneckType.LOW_ISSUE_RATE
                severity = "medium"

        # Vector 主导
        if pattern == InstructionPattern.VECTOR_DOMINANT:
            if score < 20:
                bottleneck_type = ComputeBottleneckType.VECTOR_HEAVY
                score = 30
                severity = "medium"

        # Scalar 过重
        if pattern == InstructionPattern.SCALAR_HEAVY:
            bottleneck_type = ComputeBottleneckType.INSTRUCTION_IMBALANCE
            score = max(score, 40)
            severity = "high"

        # 确定严重程度
        if score >= 60:
            severity = "critical"
        elif score >= 40:
            severity = severity if severity in ("high", "critical") else "high"
        elif score >= 20:
            severity = severity if severity in ("medium", "high", "critical") else "medium"

        return InstructionBottleneck(
            bottleneck_type=bottleneck_type,
            severity=severity,
            score=score,
        )

    def _generate_recommendations(
        self,
        arith: ExtendedArithmeticUtilization,
        pattern: InstructionPattern,
        bottleneck: InstructionBottleneck,
    ) -> List[str]:
        """生成优化建议"""
        recommendations = []

        # Cube 利用率低
        if arith.cube_utilization < self.CUBE_UNDERUTILIZED_THRESHOLD:
            recommendations.extend([
                "考虑使用更高维度的矩阵乘法（如 [16,16,16] → [32,32,32]）以充分利用 Cube 单元",
                "优化数据布局（如 NCHW → NC1HWC0）以提高计算密度",
                "检查是否可以合并多个小矩阵乘法为一个大矩阵乘法",
                "调整 tile size 以增加 Cube 的计算负载",
            ])

        # Vector 主导
        if pattern == InstructionPattern.VECTOR_DOMINANT:
            recommendations.extend([
                "评估是否可以将部分 Vector 操作转为 Cube 操作（如使用矩阵乘法替代向量运算）",
                "优化向量运算的并行度，充分利用 Vector 单元",
                "检查是否存在不必要的逐元素操作",
            ])

        # 指令发射率低
        if arith.instruction_issue_rate < self.LOW_ISSUE_RATE_THRESHOLD:
            recommendations.extend([
                "检查是否存在频繁的 kernel launch 开销，考虑使用算子融合",
                "优化内存访问模式以减少等待时间",
                "检查是否存在过多的同步操作",
            ])

        # Scalar 过重
        if pattern == InstructionPattern.SCALAR_HEAVY:
            recommendations.extend([
                "优化控制流逻辑，减少条件分支",
                "将标量计算向量化",
                "检查是否存在可以并行的循环",
            ])

        # 指令不平衡
        if bottleneck.bottleneck_type == ComputeBottleneckType.INSTRUCTION_IMBALANCE:
            recommendations.extend([
                "优化指令调度以平衡 Cube/Vector/Scalar 负载",
                "检查是否可以通过算子融合减少指令数量",
            ])

        return recommendations

    def analyze_batch(
        self,
        metrics_list: List[ExtendedAICMetrics],
    ) -> Tuple[List[InstructionBottleneck], Dict[str, Any]]:
        """
        批量分析多个算子的指令级性能

        Args:
            metrics_list: 扩展 AIC 指标列表

        Returns:
            (瓶颈列表, 汇总统计)
        """
        bottlenecks = []
        pattern_counts = {
            "cube_dominant": 0,
            "vector_dominant": 0,
            "scalar_heavy": 0,
            "balanced": 0,
        }

        for metrics in metrics_list:
            if not metrics.extended_arithmetic:
                continue

            bottleneck = self.analyze(metrics)
            bottlenecks.append(bottleneck)

            # 统计指令模式
            mix = metrics.extended_arithmetic.get_instruction_mix()
            if mix["cube"] > 0.6:
                pattern_counts["cube_dominant"] += 1
            elif mix["vector"] > 0.6:
                pattern_counts["vector_dominant"] += 1
            elif mix["scalar"] > 0.3:
                pattern_counts["scalar_heavy"] += 1
            else:
                pattern_counts["balanced"] += 1

        # 计算汇总统计
        summary = {
            "total_analyzed": len(bottlenecks),
            "critical_count": sum(1 for b in bottlenecks if b.severity == "critical"),
            "high_count": sum(1 for b in bottlenecks if b.severity == "high"),
            "medium_count": sum(1 for b in bottlenecks if b.severity == "medium"),
            "pattern_distribution": pattern_counts,
            "avg_cube_util": 0.0,
            "avg_issue_rate": 0.0,
        }

        if bottlenecks:
            summary["avg_cube_util"] = sum(b.cube_utilization for b in bottlenecks) / len(bottlenecks)
            summary["avg_issue_rate"] = sum(
                b.instruction_mix.get("cube", 0) for b in bottlenecks
            ) / len(bottlenecks) * 100

        return bottlenecks, summary


def analyze_instruction_level(
    metrics: ExtendedAICMetrics,
) -> InstructionBottleneck:
    """
    便捷函数：分析指令级性能

    Args:
        metrics: 扩展 AIC 指标数据

    Returns:
        InstructionBottleneck: 指令瓶颈分析结果
    """
    analyzer = InstructionAnalyzer()
    return analyzer.analyze(metrics)
