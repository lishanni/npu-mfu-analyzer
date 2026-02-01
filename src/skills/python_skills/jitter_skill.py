"""
抖动检测技能

检测计算和通信的时间波动
"""

from typing import Dict, Any, List
import logging
import statistics

from ..base_skill import (
    BaseSkill,
    SkillMetadata,
    SkillCategory,
    SkillPriority,
    SkillInput,
    SkillOutput,
    SkillResult,
)

logger = logging.getLogger(__name__)


class DetectComputeJitterSkill(BaseSkill):
    """
    检测计算抖动
    
    分析算子执行时间的波动情况
    """
    
    @property
    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="detect_compute_jitter",
            display_name="检测计算抖动",
            description="检测算子执行时间的波动，识别不稳定的计算模式",
            category=SkillCategory.DIAGNOSIS,
            priority=SkillPriority.HIGH,
            version="1.0.0",
            inputs=[
                SkillInput(
                    name="durations",
                    type="list",
                    required=True,
                    description="算子执行时间列表（微秒）",
                ),
                SkillInput(
                    name="operator_name",
                    type="str",
                    required=False,
                    default="Unknown",
                    description="算子名称",
                ),
                SkillInput(
                    name="cv_threshold",
                    type="float",
                    required=False,
                    default=0.15,
                    description="变异系数阈值（默认 15%）",
                ),
            ],
            outputs=[
                SkillOutput(name="mean", type="float", description="平均值"),
                SkillOutput(name="std", type="float", description="标准差"),
                SkillOutput(name="cv", type="float", description="变异系数"),
                SkillOutput(name="has_jitter", type="bool", description="是否存在抖动"),
                SkillOutput(name="outlier_count", type="int", description="异常值数量"),
            ],
            tags=["jitter", "compute", "stability", "variance", "diagnosis"],
        )
    
    def execute(
        self,
        durations: List[float],
        operator_name: str = "Unknown",
        cv_threshold: float = 0.15,
        **kwargs,
    ) -> SkillResult:
        """检测计算抖动"""
        
        if len(durations) < 2:
            return SkillResult(
                skill_name=self.metadata.name,
                success=False,
                error="需要至少 2 个数据点",
            )
        
        # 计算统计量
        mean = statistics.mean(durations)
        std = statistics.stdev(durations)
        cv = std / mean if mean > 0 else 0
        
        # 检测异常值（3-sigma 规则）
        outliers = [d for d in durations if abs(d - mean) > 3 * std]
        outlier_count = len(outliers)
        
        # 判断是否存在抖动
        has_jitter = cv > cv_threshold or outlier_count > len(durations) * 0.05
        
        # 生成建议
        suggestions = []
        
        if has_jitter:
            suggestions.append(f"算子 {operator_name} 存在执行时间抖动 (CV={cv*100:.1f}%)")
            
            if cv > 0.3:
                suggestions.append("抖动严重（CV > 30%），可能原因：")
                suggestions.append("  1. CPU 调度干扰（检查 CPU 绑核配置）")
                suggestions.append("  2. 内存分配不稳定（检查是否有动态 shape）")
                suggestions.append("  3. 缓存命中率波动")
            elif cv > 0.2:
                suggestions.append("抖动明显（CV > 20%），建议排查系统级干扰")
            else:
                suggestions.append("轻微抖动，通常可接受")
            
            if outlier_count > 0:
                suggestions.append(f"检测到 {outlier_count} 个异常值，可能是偶发性能抖动")
        
        return SkillResult(
            skill_name=self.metadata.name,
            success=True,
            data={
                "operator_name": operator_name,
                "mean_us": round(mean, 2),
                "std_us": round(std, 2),
                "cv": round(cv, 4),
                "cv_percent": round(cv * 100, 2),
                "has_jitter": has_jitter,
                "outlier_count": outlier_count,
                "sample_count": len(durations),
                "min_us": round(min(durations), 2),
                "max_us": round(max(durations), 2),
            },
            summary=f"{operator_name}: {'存在抖动' if has_jitter else '稳定'} "
                    f"(CV={cv*100:.1f}%, 异常值={outlier_count})",
            suggestions=suggestions,
            confidence=0.9 if len(durations) >= 10 else 0.7,
        )


class DetectCommJitterSkill(BaseSkill):
    """
    检测通信抖动
    
    分析通信操作时间的波动情况
    """
    
    @property
    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="detect_comm_jitter",
            display_name="检测通信抖动",
            description="检测通信操作时间的波动，识别网络不稳定性",
            category=SkillCategory.DIAGNOSIS,
            priority=SkillPriority.HIGH,
            version="1.0.0",
            inputs=[
                SkillInput(
                    name="durations",
                    type="list",
                    required=True,
                    description="通信操作时间列表（微秒）",
                ),
                SkillInput(
                    name="op_type",
                    type="str",
                    required=False,
                    default="AllReduce",
                    description="通信操作类型",
                ),
                SkillInput(
                    name="cv_threshold",
                    type="float",
                    required=False,
                    default=0.15,
                    description="变异系数阈值",
                ),
            ],
            outputs=[
                SkillOutput(name="mean", type="float", description="平均值"),
                SkillOutput(name="std", type="float", description="标准差"),
                SkillOutput(name="cv", type="float", description="变异系数"),
                SkillOutput(name="has_jitter", type="bool", description="是否存在抖动"),
            ],
            tags=["jitter", "communication", "network", "stability"],
        )
    
    def execute(
        self,
        durations: List[float],
        op_type: str = "AllReduce",
        cv_threshold: float = 0.15,
        **kwargs,
    ) -> SkillResult:
        """检测通信抖动"""
        
        if len(durations) < 2:
            return SkillResult(
                skill_name=self.metadata.name,
                success=False,
                error="需要至少 2 个数据点",
            )
        
        mean = statistics.mean(durations)
        std = statistics.stdev(durations)
        cv = std / mean if mean > 0 else 0
        
        # 检测异常值
        outliers = [d for d in durations if abs(d - mean) > 3 * std]
        outlier_count = len(outliers)
        
        has_jitter = cv > cv_threshold
        
        suggestions = []
        
        if has_jitter:
            suggestions.append(f"{op_type} 通信存在抖动 (CV={cv*100:.1f}%)")
            
            if cv > 0.25:
                suggestions.append("通信抖动严重，可能原因：")
                suggestions.append("  1. 网络拥塞或链路不稳定")
                suggestions.append("  2. 其他任务抢占网络带宽")
                suggestions.append("  3. RDMA QP 配置问题")
            else:
                suggestions.append("通信抖动明显，建议：")
                suggestions.append("  1. 检查网络流量是否均匀")
                suggestions.append("  2. 确认节点间链路状态")
            
            if outlier_count > 0:
                max_outlier = max(outliers) if outliers else 0
                suggestions.append(f"最大异常值: {max_outlier/1000:.2f}ms，可能是网络瞬时拥塞")
        
        return SkillResult(
            skill_name=self.metadata.name,
            success=True,
            data={
                "op_type": op_type,
                "mean_us": round(mean, 2),
                "mean_ms": round(mean / 1000, 2),
                "std_us": round(std, 2),
                "cv": round(cv, 4),
                "cv_percent": round(cv * 100, 2),
                "has_jitter": has_jitter,
                "outlier_count": outlier_count,
                "sample_count": len(durations),
            },
            summary=f"{op_type}: {'存在抖动' if has_jitter else '稳定'} (CV={cv*100:.1f}%)",
            suggestions=suggestions,
            confidence=0.9,
        )


class AnalyzeCrossRankJitterSkill(BaseSkill):
    """
    分析跨 Rank 抖动
    """
    
    @property
    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="analyze_cross_rank_jitter",
            display_name="分析跨 Rank 抖动",
            description="分析不同 Rank 之间的性能差异，识别慢卡",
            category=SkillCategory.DIAGNOSIS,
            priority=SkillPriority.HIGH,
            version="1.0.0",
            inputs=[
                SkillInput(
                    name="rank_durations",
                    type="dict",
                    required=True,
                    description="各 Rank 的时间数据 {rank_id: [durations]}",
                ),
                SkillInput(
                    name="slow_threshold",
                    type="float",
                    required=False,
                    default=1.2,
                    description="慢卡阈值（中位数的倍数）",
                ),
            ],
            outputs=[
                SkillOutput(name="variance", type="float", description="跨 Rank 方差"),
                SkillOutput(name="slow_ranks", type="list", description="慢 Rank 列表"),
                SkillOutput(name="has_imbalance", type="bool", description="是否存在不均衡"),
            ],
            tags=["jitter", "cross-rank", "slow-rank", "balance"],
        )
    
    def execute(
        self,
        rank_durations: Dict[int, List[float]],
        slow_threshold: float = 1.2,
        **kwargs,
    ) -> SkillResult:
        """分析跨 Rank 抖动"""
        
        if not rank_durations:
            return SkillResult(
                skill_name=self.metadata.name,
                success=False,
                error="无 Rank 数据",
            )
        
        # 计算每个 Rank 的平均时间
        rank_means = {
            rank: statistics.mean(durations) 
            for rank, durations in rank_durations.items()
            if durations
        }
        
        if len(rank_means) < 2:
            return SkillResult(
                skill_name=self.metadata.name,
                success=False,
                error="需要至少 2 个 Rank 的数据",
            )
        
        # 计算跨 Rank 统计
        mean_values = list(rank_means.values())
        overall_mean = statistics.mean(mean_values)
        variance = statistics.variance(mean_values)
        median = statistics.median(mean_values)
        
        # 识别慢 Rank
        slow_ranks = [
            rank for rank, mean in rank_means.items()
            if mean > median * slow_threshold
        ]
        
        has_imbalance = len(slow_ranks) > 0 or variance > (overall_mean * 0.1) ** 2
        
        suggestions = []
        
        if slow_ranks:
            suggestions.append(f"检测到 {len(slow_ranks)} 个慢 Rank: {slow_ranks}")
            suggestions.append("可能原因：")
            suggestions.append("  1. 硬件性能差异（检查 NPU 频率）")
            suggestions.append("  2. 数据加载不均衡")
            suggestions.append("  3. 内存或缓存竞争")
            
            # 分析慢卡特征
            for rank in slow_ranks[:3]:  # 最多显示 3 个
                rank_mean = rank_means[rank]
                ratio = rank_mean / median
                suggestions.append(f"  Rank {rank}: 比中位数慢 {(ratio-1)*100:.1f}%")
        
        if has_imbalance and not slow_ranks:
            suggestions.append("存在跨 Rank 性能波动，但无明显慢卡")
            suggestions.append("建议检查训练数据分布是否均匀")
        
        return SkillResult(
            skill_name=self.metadata.name,
            success=True,
            data={
                "variance": round(variance, 2),
                "std": round(variance ** 0.5, 2),
                "overall_mean": round(overall_mean, 2),
                "median": round(median, 2),
                "slow_ranks": slow_ranks,
                "has_imbalance": has_imbalance,
                "rank_count": len(rank_means),
                "rank_means": {k: round(v, 2) for k, v in rank_means.items()},
            },
            summary=f"跨 Rank 分析: {'不均衡' if has_imbalance else '均衡'}, "
                    f"慢 Rank: {slow_ranks if slow_ranks else '无'}",
            suggestions=suggestions,
            confidence=0.85,
        )
