"""
带宽效率计算技能
"""

import logging
from typing import Dict, Any

from npu_mfu_analyzer.skills.v2.base import (
    BaseSkill,
    SkillType,
    SkillCategory,
    SkillMetadata,
    SkillPriority,
    SkillContext,
    SkillResult,
)

logger = logging.getLogger(__name__)


class BandwidthSkill(BaseSkill):
    """带宽效率计算技能"""

    @property
    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="compute_bandwidth",
            display_name="带宽效率计算",
            description="计算通信带宽利用率",
            skill_type=SkillType.COMPUTE,
            category=SkillCategory.COMMUNICATION,
            priority=SkillPriority.NORMAL,
            tags=["bandwidth", "communication", "efficiency"],
            dependencies=[],
        )

    def execute(self, context: SkillContext) -> SkillResult:
        """执行带宽效率计算"""
        try:
            profiling_summary = context.profiling_summary

            if not profiling_summary:
                return SkillResult(
                    skill_name=self.name,
                    skill_type=self.skill_type,
                    success=False,
                    error="缺少 Profiling 数据",
                )

            # 获取通信数据
            if hasattr(profiling_summary, 'to_dict'):
                summary_dict = profiling_summary.to_dict()
            else:
                summary_dict = profiling_summary

            comm_time_us = summary_dict.get("avg_comm_time", 0)
            comm_data_mb = summary_dict.get("total_comm_data_mb", 0)

            # 计算带宽
            if comm_time_us > 0 and comm_data_mb > 0:
                # GB/s = MB / (us / 1000000) / 1000
                bandwidth_gbps = comm_data_mb / (comm_time_us / 1000000) / 1000

                # 理论带宽（HCCS: 56 GB/s, RDMA: 100 GB/s）
                theoretical_bw = 56.0  # 默认使用 HCCS
                efficiency = (bandwidth_gbps / theoretical_bw) * 100 if theoretical_bw > 0 else 0
            else:
                bandwidth_gbps = 0.0
                efficiency = 0.0

            # 生成建议
            recommendations = []
            if efficiency < 50:
                recommendations.extend([
                    "带宽利用率偏低，检查是否存在通信瓶颈",
                    "考虑优化通信掩盖策略",
                    "检查 HCCL 配置参数",
                ])

            return SkillResult(
                skill_name=self.name,
                skill_type=self.skill_type,
                success=True,
                data={
                    "measured_bandwidth_gbps": bandwidth_gbps,
                    "theoretical_bandwidth_gbps": theoretical_bw,
                    "efficiency_percent": efficiency,
                },
                summary=f"带宽利用率: {efficiency:.1f}% ({bandwidth_gbps:.2f} GB/s)",
                recommendations=recommendations,
            )

        except Exception as e:
            logger.exception(f"带宽计算失败: {e}")
            return SkillResult(
                skill_name=self.name,
                skill_type=self.skill_type,
                success=False,
                error=str(e),
            )


class OverlapSkill(BaseSkill):
    """通信掩盖率计算技能"""

    @property
    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="compute_overlap",
            display_name="通信掩盖率计算",
            description="计算计算与通信的重叠比例",
            skill_type=SkillType.COMPUTE,
            category=SkillCategory.COMMUNICATION,
            priority=SkillPriority.NORMAL,
            tags=["overlap", "communication", "efficiency"],
            dependencies=[],
        )

    def execute(self, context: SkillContext) -> SkillResult:
        """执行掩盖率计算"""
        try:
            profiling_summary = context.profiling_summary

            if not profiling_summary:
                return SkillResult(
                    skill_name=self.name,
                    skill_type=self.skill_type,
                    success=False,
                    error="缺少 Profiling 数据",
                )

            # 使用现有的 OverlapCalculator
            from npu_mfu_analyzer.analyzers.overlap_calculator import OverlapCalculator

            calculator = OverlapCalculator()

            # 从 context 获取事件数据
            # 这里简化处理，使用摘要数据估算
            if hasattr(profiling_summary, 'to_dict'):
                summary_dict = profiling_summary.to_dict()
            else:
                summary_dict = profiling_summary

            compute_time = summary_dict.get("avg_compute_time", 0)
            comm_time = summary_dict.get("avg_comm_time", 0)

            # 估算掩盖率
            # 简化公式: overlap_ratio = (compute - (total - comm)) / comm
            # 当 compute 和 comm 完全重叠时，overlap_ratio = 100%
            total_time = summary_dict.get("avg_step_time", compute_time + comm_time)

            if total_time > 0 and comm_time > 0:
                # 估算暴露时间
                exposed_time = max(0, total_time - compute_time)
                overlap_time = max(0, comm_time - exposed_time)
                overlap_ratio = (overlap_time / comm_time * 100) if comm_time > 0 else 0
            else:
                overlap_ratio = 0.0
                exposed_time = 0.0

            recommendations = []
            if overlap_ratio < 50:
                recommendations.extend([
                    "通信掩盖率偏低，存在通信等待",
                    "考虑增加 micro batch 数量",
                    "优化流水线调度策略",
                ])

            return SkillResult(
                skill_name=self.name,
                skill_type=self.skill_type,
                success=True,
                data={
                    "overlap_ratio_percent": overlap_ratio,
                    "exposed_comm_time_us": exposed_time,
                },
                summary=f"通信掩盖率: {overlap_ratio:.1f}%",
                recommendations=recommendations,
            )

        except Exception as e:
            logger.exception(f"掩盖率计算失败: {e}")
            return SkillResult(
                skill_name=self.name,
                skill_type=self.skill_type,
                success=False,
                error=str(e),
            )


class SlowRankSkill(BaseSkill):
    """慢卡检测技能"""

    @property
    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="detect_slow_ranks",
            display_name="慢卡检测",
            description="检测性能异常的 Rank",
            skill_type=SkillType.COMPUTE,
            category=SkillCategory.DIAGNOSIS,
            priority=SkillPriority.HIGH,
            tags=["slow_rank", "diagnosis", "distributed"],
            dependencies=[],
        )

    def execute(self, context: SkillContext) -> SkillResult:
        """执行慢卡检测"""
        try:
            from npu_mfu_analyzer.analyzers.slow_rank_detector import SlowRankDetector

            detector = SlowRankDetector()

            # 从 context 获取 rank 时间数据
            # 这里需要实际数据，简化处理
            if hasattr(context, 'rank_times') and context.rank_times:
                result = detector.detect(context.rank_times)
                slow_ranks = result.slow_ranks if hasattr(result, 'slow_ranks') else []
            else:
                # 没有足够数据，返回空结果
                slow_ranks = []

            recommendations = []
            if slow_ranks:
                recommendations.extend([
                    f"检测到慢卡: {slow_ranks}",
                    "检查慢卡硬件状态",
                    "检查数据分布是否均匀",
                    "检查网络连接稳定性",
                ])

            return SkillResult(
                skill_name=self.name,
                skill_type=self.skill_type,
                success=True,
                data={
                    "slow_ranks": slow_ranks,
                    "count": len(slow_ranks),
                },
                summary=f"检测到 {len(slow_ranks)} 个慢卡" if slow_ranks else "未检测到慢卡",
                recommendations=recommendations,
                priority="P0" if len(slow_ranks) > 0 else "P2",
            )

        except Exception as e:
            logger.exception(f"慢卡检测失败: {e}")
            return SkillResult(
                skill_name=self.name,
                skill_type=self.skill_type,
                success=False,
                error=str(e),
            )