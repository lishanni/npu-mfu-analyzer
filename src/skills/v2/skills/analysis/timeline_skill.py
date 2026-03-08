"""
时间线分析技能

分析 Profiling 时间线，识别性能瓶颈。
"""

import logging
from typing import Dict, Any, List

from src.skills.v2.base import (
    BaseSkill,
    SkillType,
    SkillCategory,
    SkillMetadata,
    SkillPriority,
    SkillContext,
    SkillResult,
)

logger = logging.getLogger(__name__)


class TimelineAnalysisSkill(BaseSkill):
    """时间线分析技能"""

    @property
    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="analyze_timeline",
            display_name="时间线分析",
            description="分析 Profiling 时间线，识别性能瓶颈和异常模式",
            skill_type=SkillType.ANALYSIS,
            category=SkillCategory.ANALYSIS,
            priority=SkillPriority.HIGH,
            tags=["timeline", "performance", "bottleneck"],
            dependencies=[],
        )

    def execute(self, context: SkillContext) -> SkillResult:
        """执行时间线分析"""
        try:
            profiling_summary = context.profiling_summary

            if not profiling_summary:
                return SkillResult(
                    skill_name=self.name,
                    skill_type=self.skill_type,
                    success=False,
                    error="缺少 Profiling 数据",
                )

            # 获取摘要数据
            if hasattr(profiling_summary, 'to_dict'):
                summary_dict = profiling_summary.to_dict()
            else:
                summary_dict = profiling_summary

            # 分析时间分布
            step_time = summary_dict.get("avg_step_time", 0)
            compute_time = summary_dict.get("avg_compute_time", 0)
            comm_time = summary_dict.get("avg_comm_time", 0)
            other_time = step_time - compute_time - comm_time if step_time > 0 else 0

            # 计算时间占比
            compute_ratio = (compute_time / step_time * 100) if step_time > 0 else 0
            comm_ratio = (comm_time / step_time * 100) if step_time > 0 else 0
            other_ratio = (other_time / step_time * 100) if step_time > 0 else 0

            # 识别瓶颈
            bottleneck = self._identify_bottleneck(compute_ratio, comm_ratio, other_ratio)

            # 生成详情
            details = [
                f"Step 平均耗时: {step_time:.2f} us",
                f"计算时间: {compute_time:.2f} us ({compute_ratio:.1f}%)",
                f"通信时间: {comm_time:.2f} us ({comm_ratio:.1f}%)",
                f"其他时间: {other_time:.2f} us ({other_ratio:.1f}%)",
            ]

            # 生成建议
            recommendations = []
            if bottleneck == "communication":
                recommendations.extend([
                    "通信时间占比较高，建议优化通信策略",
                    "检查 HCCL 配置和通信掩盖",
                    "考虑使用梯度累积减少通信频率",
                ])
            elif bottleneck == "compute":
                recommendations.extend([
                    "计算时间占比较高，建议优化算子效率",
                    "检查算子融合配置",
                    "考虑使用 torch.compile 优化",
                ])
            elif bottleneck == "other":
                recommendations.extend([
                    "存在大量非计算/通信开销",
                    "检查数据加载、Host 端处理等",
                    "考虑使用异步数据加载",
                ])

            return SkillResult(
                skill_name=self.name,
                skill_type=self.skill_type,
                success=True,
                data={
                    "step_time_us": step_time,
                    "compute_time_us": compute_time,
                    "comm_time_us": comm_time,
                    "other_time_us": other_time,
                    "compute_ratio_percent": compute_ratio,
                    "comm_ratio_percent": comm_ratio,
                    "other_ratio_percent": other_ratio,
                    "bottleneck": bottleneck,
                },
                summary=f"瓶颈类型: {bottleneck} (计算 {compute_ratio:.1f}% / 通信 {comm_ratio:.1f}%)",
                details=details,
                recommendations=recommendations,
                priority="P0" if bottleneck in ["communication", "other"] else "P1",
            )

        except Exception as e:
            logger.exception(f"时间线分析失败: {e}")
            return SkillResult(
                skill_name=self.name,
                skill_type=self.skill_type,
                success=False,
                error=str(e),
            )

    def _identify_bottleneck(
        self,
        compute_ratio: float,
        comm_ratio: float,
        other_ratio: float,
    ) -> str:
        """识别主要瓶颈"""
        ratios = {
            "compute": compute_ratio,
            "communication": comm_ratio,
            "other": other_ratio,
        }
        return max(ratios, key=ratios.get)