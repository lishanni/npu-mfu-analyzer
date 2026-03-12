"""
MFU 计算技能

计算 Model FLOPS Utilization。
"""

from dataclasses import dataclass
from typing import Dict, Any, List
import logging

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


class MFUSkill(BaseSkill):
    """
    MFU 计算技能

    计算模型的 MFU (Model FLOPS Utilization)，用于评估硬件利用效率。

    公式:
    MFU = Actual_FLOPs / (Duration × Peak_FLOPs)

    其中:
    - Actual_FLOPs: 模型实际计算量
    - Duration: 端到端训练时间
    - Peak_FLOPs: 硬件峰值算力
    """

    @property
    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="compute_mfu",
            display_name="MFU 计算",
            description="计算 Model FLOPS Utilization，评估硬件利用效率",
            skill_type=SkillType.COMPUTE,
            category=SkillCategory.COMPUTE,
            priority=SkillPriority.HIGH,
            tags=["mfu", "performance", "core", "compute"],
            inputs=[],
            outputs=[
                {"name": "overall_mfu", "type": "float", "description": "整体 MFU 百分比"},
                {"name": "peak_flops", "type": "float", "description": "硬件峰值算力 (FLOPS)"},
                {"name": "actual_flops", "type": "float", "description": "实际计算量 (FLOPS)"},
                {"name": "efficiency_level", "type": "str", "description": "效率等级"},
            ],
            dependencies=[],
        )

    def execute(self, context: SkillContext) -> SkillResult:
        """执行 MFU 计算"""
        try:
            # 获取数据
            profiling_summary = context.profiling_summary
            hardware_spec = context.hardware_spec

            if not profiling_summary:
                return SkillResult(
                    skill_name=self.name,
                    skill_type=self.skill_type,
                    success=False,
                    error="缺少 Profiling 数据",
                )

            # 使用现有的 MFUCalculator
            from npu_mfu_analyzer.analyzers.mfu_calculator import MFUCalculator

            if hardware_spec:
                calculator = MFUCalculator(chip_info=hardware_spec)
            else:
                # 使用默认硬件规格
                from npu_mfu_analyzer.hardware.registry import get_registry
                registry = get_registry()
                chip_info = registry.get_default()
                calculator = MFUCalculator(chip_info=chip_info)

            # 计算 MFU
            # 从 profiling_summary 获取算子数据
            if hasattr(profiling_summary, 'operators'):
                operators = profiling_summary.operators
            elif hasattr(profiling_summary, 'to_dict'):
                summary_dict = profiling_summary.to_dict()
                operators = summary_dict.get('operators', [])
            else:
                operators = []

            if not operators:
                # 使用简化计算
                mfu_result = self._simple_mfu_calculation(profiling_summary)
            else:
                # 精确计算
                import pandas as pd
                if not isinstance(operators, pd.DataFrame):
                    operators_df = pd.DataFrame(operators)
                else:
                    operators_df = operators

                mfu_metrics = calculator.analyze_operators(operators_df)
                mfu_result = {
                    "overall_mfu": mfu_metrics.overall_mfu,
                    "peak_flops": mfu_metrics.peak_flops,
                    "actual_flops": mfu_metrics.actual_flops,
                }

            # 确定效率等级
            mfu = mfu_result.get("overall_mfu", 0)
            if mfu >= 0.55:
                efficiency_level = "优秀"
            elif mfu >= 0.45:
                efficiency_level = "良好"
            elif mfu >= 0.35:
                efficiency_level = "一般"
            else:
                efficiency_level = "待优化"

            # 生成建议
            recommendations = []
            if mfu < 0.35:
                recommendations.extend([
                    "MFU 偏低，建议检查算子融合配置",
                    "考虑使用 torch.compile 优化计算效率",
                    "检查是否存在不必要的同步点",
                ])
            elif mfu < 0.45:
                recommendations.append("MFU 有提升空间，建议优化通信掩盖")

            return SkillResult(
                skill_name=self.name,
                skill_type=self.skill_type,
                success=True,
                data={
                    "overall_mfu": mfu,
                    "peak_flops": mfu_result.get("peak_flops", 0),
                    "actual_flops": mfu_result.get("actual_flops", 0),
                    "efficiency_level": efficiency_level,
                },
                summary=f"MFU: {mfu * 100:.1f}% ({efficiency_level})",
                recommendations=recommendations,
                priority="P0" if mfu < 0.35 else "P1",
            )

        except Exception as e:
            logger.exception(f"MFU 计算失败: {e}")
            return SkillResult(
                skill_name=self.name,
                skill_type=self.skill_type,
                success=False,
                error=str(e),
            )

    def _simple_mfu_calculation(self, profiling_summary) -> Dict[str, Any]:
        """简化 MFU 计算"""
        try:
            if hasattr(profiling_summary, 'to_dict'):
                summary_dict = profiling_summary.to_dict()
            else:
                summary_dict = profiling_summary

            compute_time_us = summary_dict.get("avg_compute_time", 0)
            total_time_us = summary_dict.get("avg_step_time", compute_time_us)

            if total_time_us <= 0:
                return {"overall_mfu": 0.0, "peak_flops": 0.0, "actual_flops": 1.0}

            # 计算时间占比
            compute_ratio = compute_time_us / total_time_us if total_time_us > 0 else 0.0

            # 估算 MFU = 计算时间占比 × 估算的算力利用率
            # 假设计算密集型操作能达到 40% 峰值
            estimated_mfu = compute_ratio * 0.4

            return {
                "overall_mfu": estimated_mfu,
                "peak_flops": 280e12,  # 默认 280 TFLOPS
                "actual_flops": estimated_mfu * 280e12,
            }

        except Exception as e:
            logger.warning(f"简化 MFU 计算失败: {e}")
            return {"overall_mfu": 1.0, "peak_flops": 1.0, "actual_flops": 1.0}
