"""
内存层次分析器

分析 AI Core 的内存层次性能指标，包括：
- L2/UB/L0 访问模式
- 缓存命中率分析
- 数据局部性
- 内存访问模式优化建议
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum

from src.data_loader.aic_metrics import (
    ExtendedMemoryMetrics,
    ExtendedAICMetrics,
    MemoryBottleneckType,
)

logger = logging.getLogger(__name__)


class MemoryBottleneck(Enum):
    """内存瓶颈类型"""
    L2_MISS = "l2_miss"                     # L2 命中率低
    UB_PRESSURE = "ub_pressure"             # UB 压力大
    HBM_SATURATED = "hbm_saturated"         # HBM 带宽饱和
    POOR_LOCALITY = "poor_locality"         # 数据局部性差
    NONE = "none"                           # 无瓶颈


@dataclass
class MemoryHierarchyAnalysis:
    """内存层次分析结果"""
    # 瓶颈类型
    bottleneck_type: MemoryBottleneck
    severity: str  # "critical", "high", "medium", "low"
    score: float  # 严重程度评分 (0-100)

    # 详细指标
    l2_cache_hit_rate: float = 0.0
    l2_read_bytes: int = 0
    l2_write_bytes: int = 0
    l2_read_bandwidth: float = 0.0
    l2_write_bandwidth: float = 0.0
    l2_miss_count: int = 0

    ub_usage: float = 0.0
    ub_peak_usage: float = 0.0
    ub_spill_count: int = 0
    ub_spill_bytes: int = 0
    ub_conflict_rate: float = 0.0

    l0a_utilization: float = 0.0
    l0b_utilization: float = 0.0
    l0c_utilization: float = 0.0

    hbm_read_bytes: int = 0
    hbm_write_bytes: int = 0
    hbm_access_count: int = 0
    hbm_bandwidth_utilization: float = 0.0

    locality_score: float = 0.0
    reuse_distance: float = 0.0

    # 优化建议
    recommendations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "bottleneck_type": self.bottleneck_type.value,
            "severity": self.severity,
            "score": round(self.score, 2),
            "l2_cache_hit_rate": round(self.l2_cache_hit_rate, 2),
            "l2_read_bytes": self.l2_read_bytes,
            "l2_write_bytes": self.l2_write_bytes,
            "l2_read_bandwidth": round(self.l2_read_bandwidth, 2),
            "l2_write_bandwidth": round(self.l2_write_bandwidth, 2),
            "l2_miss_count": self.l2_miss_count,
            "ub_usage": round(self.ub_usage, 2),
            "ub_peak_usage": round(self.ub_peak_usage, 2),
            "ub_spill_count": self.ub_spill_count,
            "ub_spill_bytes": self.ub_spill_bytes,
            "ub_conflict_rate": round(self.ub_conflict_rate, 2),
            "l0a_utilization": round(self.l0a_utilization, 2),
            "l0b_utilization": round(self.l0b_utilization, 2),
            "l0c_utilization": round(self.l0c_utilization, 2),
            "hbm_read_bytes": self.hbm_read_bytes,
            "hbm_write_bytes": self.hbm_write_bytes,
            "hbm_access_count": self.hbm_access_count,
            "hbm_bandwidth_utilization": round(self.hbm_bandwidth_utilization, 2),
            "locality_score": round(self.locality_score, 2),
            "reuse_distance": round(self.reuse_distance, 2),
            "recommendations": self.recommendations,
        }


class MemoryHierarchyAnalyzer:
    """
    内存层次分析器

    分析 AI Core 的 L2/UB/L0/HBM 内存层次性能，提供深度瓶颈诊断和优化建议。
    数据来源于 msprof 采集的 AIC PMU 事件。
    """

    # 阈值配置
    L2_HIT_RATE_LOW_THRESHOLD = 50.0        # L2 命中率低于此值视为命中率低
    UB_USAGE_HIGH_THRESHOLD = 80.0         # UB 使用率高于此值视为压力大
    UB_SPILL_THRESHOLD = 100               # UB 溢出次数超过此值视为严重
    HBM_BANDWIDTH_HIGH_THRESHOLD = 80.0    # HBM 带宽利用率高于此值视为饱和
    LOCALITY_SCORE_LOW_THRESHOLD = 50.0    # 局部性评分低于此值视为局部性差

    def __init__(self):
        """初始化内存层次分析器"""
        self._analysis_cache: Dict[str, MemoryHierarchyAnalysis] = {}

    def analyze(self, metrics: ExtendedAICMetrics) -> MemoryHierarchyAnalysis:
        """
        分析内存层次性能

        Args:
            metrics: 扩展 AIC 指标数据

        Returns:
            MemoryHierarchyAnalysis: 内存层次分析结果
        """
        if not metrics.extended_memory:
            logger.warning("No extended memory metrics available for memory hierarchy analysis")
            return MemoryHierarchyAnalysis(
                bottleneck_type=MemoryBottleneck.NONE,
                severity="low",
                score=0.0,
                recommendations=["需要启用 msprof AIC PMU 采集以获取内存层次指标"]
            )

        mem = metrics.extended_memory

        # 分析瓶颈类型
        analysis = self._identify_bottleneck(mem)

        # 设置详细指标
        analysis.l2_cache_hit_rate = mem.l2_cache_hit_rate
        analysis.l2_read_bytes = mem.l2_read_bytes
        analysis.l2_write_bytes = mem.l2_write_bytes
        analysis.l2_read_bandwidth = mem.l2_read_bandwidth
        analysis.l2_write_bandwidth = mem.l2_write_bandwidth
        analysis.l2_miss_count = mem.l2_miss_count

        analysis.ub_usage = mem.ub_usage
        analysis.ub_peak_usage = mem.ub_peak_usage
        analysis.ub_spill_count = mem.ub_spill_count
        analysis.ub_spill_bytes = mem.ub_spill_bytes
        analysis.ub_conflict_rate = mem.ub_conflict_rate

        analysis.l0a_utilization = mem.l0a_utilization
        analysis.l0b_utilization = mem.l0b_utilization
        analysis.l0c_utilization = mem.l0c_utilization

        analysis.hbm_read_bytes = mem.hbm_read_bytes
        analysis.hbm_write_bytes = mem.hbm_write_bytes
        analysis.hbm_access_count = mem.hbm_access_count
        analysis.hbm_bandwidth_utilization = mem.hbm_bandwidth_utilization

        # 计算局部性评分
        analysis.locality_score = mem.calc_locality_score()
        analysis.reuse_distance = mem.reuse_distance

        # 生成优化建议
        analysis.recommendations = self._generate_recommendations(mem, analysis)

        return analysis

    def _identify_bottleneck(
        self,
        mem: ExtendedMemoryMetrics,
    ) -> MemoryHierarchyAnalysis:
        """识别内存瓶颈类型"""
        score = 0.0
        bottleneck_type = MemoryBottleneck.NONE
        severity = "low"

        # L2 命中率低
        if mem.l2_cache_hit_rate < self.L2_HIT_RATE_LOW_THRESHOLD:
            score += (100 - mem.l2_cache_hit_rate) * 0.5
            bottleneck_type = MemoryBottleneck.L2_MISS
            if mem.l2_cache_hit_rate < 30:
                severity = "critical"
            else:
                severity = "high"

        # UB 压力大
        if mem.ub_usage > self.UB_USAGE_HIGH_THRESHOLD or mem.ub_spill_count > self.UB_SPILL_THRESHOLD:
            score += max(0, (mem.ub_usage - self.UB_USAGE_HIGH_THRESHOLD) * 0.5)
            score += min(50, mem.ub_spill_count / self.UB_SPILL_THRESHOLD * 20)
            if not bottleneck_type or score > 30:
                bottleneck_type = MemoryBottleneck.UB_PRESSURE
                severity = "high" if score > 40 else "medium"

        # HBM 带宽饱和
        if mem.hbm_bandwidth_utilization > self.HBM_BANDWIDTH_HIGH_THRESHOLD:
            score += (mem.hbm_bandwidth_utilization - self.HBM_BANDWIDTH_HIGH_THRESHOLD) * 1.5
            if not bottleneck_type or score > 30:
                bottleneck_type = MemoryBottleneck.HBM_SATURATED
                severity = "high" if score > 40 else "medium"

        # 局部性差
        locality = mem.calc_locality_score()
        if locality < self.LOCALITY_SCORE_LOW_THRESHOLD:
            score += (100 - locality) * 0.3
            if not bottleneck_type or score > 30:
                bottleneck_type = MemoryBottleneck.POOR_LOCALITY
                severity = "medium" if score > 30 else "low"

        # 确定严重程度
        if score >= 60:
            severity = "critical"
        elif score >= 40:
            severity = severity if severity in ("high", "critical") else "high"
        elif score >= 20:
            severity = severity if severity in ("medium", "high", "critical") else "medium"

        return MemoryHierarchyAnalysis(
            bottleneck_type=bottleneck_type,
            severity=severity,
            score=score,
        )

    def _generate_recommendations(
        self,
        mem: ExtendedMemoryMetrics,
        analysis: MemoryHierarchyAnalysis,
    ) -> List[str]:
        """生成优化建议"""
        recommendations = []

        # L2 命中率低
        if mem.l2_cache_hit_rate < self.L2_HIT_RATE_LOW_THRESHOLD:
            recommendations.extend([
                "优化数据访问模式以提高缓存命中率（如分块访问、数据预取）",
                "调整 tile size 以增加数据复用",
                "考虑使用更高效的内存布局（如 HWCN → NHWC）",
                "检查是否存在不必要的跨步访问（stride 访问）",
            ])

        # UB 压力大
        if mem.ub_usage > self.UB_USAGE_HIGH_THRESHOLD or mem.ub_spill_count > self.UB_SPILL_THRESHOLD:
            recommendations.extend([
                "减小 tile size 以降低 Unified Buffer 压力",
                "考虑使用 double buffering 技术以隐藏访存延迟",
                "优化数据分块策略以减少 UB 溢出",
                "检查是否可以减少中间结果的存储（如流水线优化）",
            ])

        # HBM 带宽饱和
        if mem.hbm_bandwidth_utilization > self.HBM_BANDWIDTH_HIGH_THRESHOLD:
            recommendations.extend([
                "使用更高效的数据类型（如 FP16/BF16 代替 FP32）",
                "优化数据布局以减少访存次数",
                "考虑使用算子融合减少中间结果的存储",
                "检查是否存在冗余的数据传输",
            ])

        # 局部性差
        if analysis.locality_score < self.LOCALITY_SCORE_LOW_THRESHOLD:
            recommendations.extend([
                "优化数据访问模式以提高空间局部性",
                "考虑调整内存布局（如行主序 vs 列主序）",
                "使用数据预取技术以隐藏访存延迟",
                "检查是否存在随机访问模式，改为顺序访问",
            ])

        # UB 冲突
        if mem.ub_conflict_rate > 20:
            recommendations.extend([
                "优化数据访问顺序以减少 UB 冲突",
                "考虑使用多级缓冲来缓解 UB 冲突",
            ])

        return recommendations

    def analyze_batch(
        self,
        metrics_list: List[ExtendedAICMetrics],
    ) -> Tuple[List[MemoryHierarchyAnalysis], Dict[str, Any]]:
        """
        批量分析多个算子的内存层次性能

        Args:
            metrics_list: 扩展 AIC 指标列表

        Returns:
            (分析结果列表, 汇总统计)
        """
        analyses = []

        for metrics in metrics_list:
            if not metrics.extended_memory:
                continue

            analysis = self.analyze(metrics)
            analyses.append(analysis)

        # 计算汇总统计
        summary = {
            "total_analyzed": len(analyses),
            "l2_miss_count": sum(1 for a in analyses if a.bottleneck_type == MemoryBottleneck.L2_MISS),
            "ub_pressure_count": sum(1 for a in analyses if a.bottleneck_type == MemoryBottleneck.UB_PRESSURE),
            "hbm_saturated_count": sum(1 for a in analyses if a.bottleneck_type == MemoryBottleneck.HBM_SATURATED),
            "poor_locality_count": sum(1 for a in analyses if a.bottleneck_type == MemoryBottleneck.POOR_LOCALITY),
            "critical_count": sum(1 for a in analyses if a.severity == "critical"),
            "high_count": sum(1 for a in analyses if a.severity == "high"),
            "medium_count": sum(1 for a in analyses if a.severity == "medium"),
            "avg_l2_hit_rate": 0.0,
            "avg_ub_usage": 0.0,
            "avg_locality_score": 0.0,
        }

        if analyses:
            summary["avg_l2_hit_rate"] = sum(a.l2_cache_hit_rate for a in analyses) / len(analyses)
            summary["avg_ub_usage"] = sum(a.ub_usage for a in analyses) / len(analyses)
            summary["avg_locality_score"] = sum(a.locality_score for a in analyses) / len(analyses)

        return analyses, summary


def analyze_memory_hierarchy(
    metrics: ExtendedAICMetrics,
) -> MemoryHierarchyAnalysis:
    """
    便捷函数：分析内存层次性能

    Args:
        metrics: 扩展 AIC 指标数据

    Returns:
        MemoryHierarchyAnalysis: 内存层次分析结果
    """
    analyzer = MemoryHierarchyAnalyzer()
    return analyzer.analyze(metrics)