"""
慢卡检测技能

检测训练集群中的慢卡
"""

from typing import Dict, Any, List, Optional
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


class DetectSlowRankSkill(BaseSkill):
    """
    检测慢卡
    
    使用统计方法识别性能异常的 Rank
    """
    
    @property
    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="detect_slow_rank",
            display_name="检测慢卡",
            description="使用 Dixon's Q 检验和三 sigma 规则检测性能异常的 Rank",
            category=SkillCategory.DIAGNOSIS,
            priority=SkillPriority.CRITICAL,
            version="1.0.0",
            inputs=[
                SkillInput(
                    name="rank_times",
                    type="dict",
                    required=True,
                    description="各 Rank 的 step 时间 {rank_id: time_ms}",
                ),
                SkillInput(
                    name="method",
                    type="str",
                    required=False,
                    default="three_sigma",
                    description="检测方法: dixon_q / three_sigma / both",
                ),
                SkillInput(
                    name="threshold_sigma",
                    type="float",
                    required=False,
                    default=2.0,
                    description="sigma 阈值（默认 2.0）",
                ),
            ],
            outputs=[
                SkillOutput(name="slow_ranks", type="list", description="慢 Rank ID 列表"),
                SkillOutput(name="rank_analysis", type="dict", description="各 Rank 分析结果"),
                SkillOutput(name="detection_method", type="str", description="使用的检测方法"),
            ],
            tags=["slow-rank", "detection", "diagnosis", "outlier", "performance"],
        )
    
    def _dixon_q_test(
        self, 
        values: List[float], 
        alpha: float = 0.05,
    ) -> List[int]:
        """
        Dixon's Q 检验
        
        适用于小样本（3-30）的异常值检测
        """
        if len(values) < 3:
            return []
        
        # Dixon's Q 临界值表（alpha=0.05）
        q_critical = {
            3: 0.970, 4: 0.829, 5: 0.710, 6: 0.628,
            7: 0.569, 8: 0.608, 9: 0.564, 10: 0.530,
        }
        
        n = len(values)
        if n > 10:
            # 对于更大的样本，使用近似值
            q_crit = 0.5
        else:
            q_crit = q_critical.get(n, 0.5)
        
        sorted_values = sorted(values)
        outlier_indices = []
        
        # 检测最大值是否为异常值
        if n >= 3:
            q_high = (sorted_values[-1] - sorted_values[-2]) / (sorted_values[-1] - sorted_values[0])
            if q_high > q_crit:
                # 找到原始列表中的索引
                max_val = sorted_values[-1]
                outlier_indices.extend([i for i, v in enumerate(values) if v == max_val])
        
        return outlier_indices
    
    def _three_sigma_test(
        self, 
        values: List[float], 
        threshold: float = 2.0,
    ) -> List[int]:
        """
        三 sigma 规则检测
        """
        if len(values) < 2:
            return []
        
        mean = statistics.mean(values)
        std = statistics.stdev(values)
        
        if std == 0:
            return []
        
        outlier_indices = [
            i for i, v in enumerate(values) 
            if (v - mean) / std > threshold  # 只检测慢的（时间长的）
        ]
        
        return outlier_indices
    
    def execute(
        self,
        rank_times: Dict[int, float],
        method: str = "three_sigma",
        threshold_sigma: float = 2.0,
        **kwargs,
    ) -> SkillResult:
        """检测慢卡"""
        
        if len(rank_times) < 3:
            return SkillResult(
                skill_name=self.metadata.name,
                success=False,
                error="需要至少 3 个 Rank 的数据进行检测",
            )
        
        ranks = list(rank_times.keys())
        times = list(rank_times.values())
        
        # 执行检测
        slow_indices = []
        
        if method in ("dixon_q", "both"):
            dixon_indices = self._dixon_q_test(times)
            slow_indices.extend(dixon_indices)
        
        if method in ("three_sigma", "both"):
            sigma_indices = self._three_sigma_test(times, threshold_sigma)
            slow_indices.extend(sigma_indices)
        
        # 去重并转换为 rank ID
        slow_indices = list(set(slow_indices))
        slow_ranks = [ranks[i] for i in slow_indices]
        
        # 计算统计信息
        mean_time = statistics.mean(times)
        std_time = statistics.stdev(times)
        median_time = statistics.median(times)
        
        # 生成各 Rank 分析
        rank_analysis = {}
        for rank, time in rank_times.items():
            deviation = (time - mean_time) / std_time if std_time > 0 else 0
            rank_analysis[rank] = {
                "time_ms": round(time, 2),
                "deviation_sigma": round(deviation, 2),
                "is_slow": rank in slow_ranks,
                "ratio_to_median": round(time / median_time, 3) if median_time > 0 else 1,
            }
        
        # 生成建议
        suggestions = []
        
        if slow_ranks:
            suggestions.append(f"检测到 {len(slow_ranks)} 个慢卡: Rank {slow_ranks}")
            
            for rank in slow_ranks[:3]:
                info = rank_analysis[rank]
                suggestions.append(
                    f"  Rank {rank}: {info['time_ms']:.1f}ms, "
                    f"比中位数慢 {(info['ratio_to_median']-1)*100:.1f}%"
                )
            
            suggestions.append("\n排查建议：")
            suggestions.append("  1. 检查慢卡的 NPU 利用率和频率")
            suggestions.append("  2. 检查是否有内存不足导致的换页")
            suggestions.append("  3. 检查数据加载是否存在瓶颈")
            suggestions.append("  4. 检查是否与其他任务共享资源")
        else:
            suggestions.append("未检测到明显慢卡，各 Rank 性能均衡")
        
        # 计算性能差异
        max_time = max(times)
        min_time = min(times)
        perf_gap = (max_time - min_time) / mean_time * 100 if mean_time > 0 else 0
        
        if perf_gap > 20 and not slow_ranks:
            suggestions.append(f"注意：虽然无单一慢卡，但最快和最慢 Rank 差距达 {perf_gap:.1f}%")
        
        return SkillResult(
            skill_name=self.metadata.name,
            success=True,
            data={
                "slow_ranks": slow_ranks,
                "slow_rank_count": len(slow_ranks),
                "total_ranks": len(ranks),
                "mean_time_ms": round(mean_time, 2),
                "std_time_ms": round(std_time, 2),
                "median_time_ms": round(median_time, 2),
                "min_time_ms": round(min_time, 2),
                "max_time_ms": round(max_time, 2),
                "performance_gap_percent": round(perf_gap, 2),
                "detection_method": method,
                "rank_analysis": rank_analysis,
            },
            summary=f"慢卡检测 ({method}): "
                    f"{'发现 ' + str(len(slow_ranks)) + ' 个慢卡' if slow_ranks else '无慢卡'}, "
                    f"性能差距 {perf_gap:.1f}%",
            suggestions=suggestions,
            confidence=0.9 if len(ranks) >= 8 else 0.75,
        )
