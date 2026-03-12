"""
通信分析技能

分析通信行为，评估通信效率。
"""

import logging
from typing import Dict, Any, List

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


class CommunicationAnalysisSkill(BaseSkill):
    """通信分析技能"""

    @property
    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="analyze_communication",
            display_name="通信分析",
            description="分析通信行为，评估通信效率和拓扑性能",
            skill_type=SkillType.ANALYSIS,
            category=SkillCategory.COMMUNICATION,
            priority=SkillPriority.HIGH,
            tags=["communication", "hccl", "topology"],
            dependencies=[],
        )

    def execute(self, context: SkillContext) -> SkillResult:
        """执行通信分析"""
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

            comm_time = summary_dict.get("avg_comm_time", 0)
            comm_data_mb = summary_dict.get("total_comm_data_mb", 0)

            # 分析通信类型
            comm_ops = summary_dict.get("communication_ops", [])
            op_stats = self._analyze_comm_ops(comm_ops)

            # 计算带宽
            if comm_time > 0 and comm_data_mb > 0:
                bandwidth_gbps = comm_data_mb / (comm_time / 1000000) / 1000
            else:
                bandwidth_gbps = 0.0

            # 评估通信效率
            efficiency_level = self._evaluate_efficiency(bandwidth_gbps)

            # 生成详情
            details = [
                f"通信时间: {comm_time:.2f} us",
                f"通信数据量: {comm_data_mb:.2f} MB",
                f"实测带宽: {bandwidth_gbps:.2f} GB/s",
                f"效率等级: {efficiency_level}",
            ]

            # 添加算子统计
            for op_name, count in op_stats.items():
                details.append(f"  - {op_name}: {count} 次")

            # 生成建议
            recommendations = []
            if bandwidth_gbps < 30:
                recommendations.extend([
                    "通信带宽偏低，建议检查 HCCL 配置",
                    "确认网络拓扑是否最优",
                    "检查是否存在慢链路",
                ])
            elif bandwidth_gbps < 45:
                recommendations.append("通信效率有提升空间，建议优化通信掩盖策略")

            # 识别潜在问题
            potential_issues = []
            if op_stats.get("all_reduce", 0) > 100:
                potential_issues.append("AllReduce 调用次数较多，考虑梯度累积")
            if op_stats.get("all_gather", 0) > 50:
                potential_issues.append("AllGather 调用较多，检查是否存在不必要的集合操作")

            return SkillResult(
                skill_name=self.name,
                skill_type=self.skill_type,
                success=True,
                data={
                    "comm_time_us": comm_time,
                    "comm_data_mb": comm_data_mb,
                    "bandwidth_gbps": bandwidth_gbps,
                    "efficiency_level": efficiency_level,
                    "op_stats": op_stats,
                    "potential_issues": potential_issues,
                },
                summary=f"通信带宽: {bandwidth_gbps:.2f} GB/s ({efficiency_level})",
                details=details,
                recommendations=recommendations,
                priority="P0" if bandwidth_gbps < 30 else "P1",
            )

        except Exception as e:
            logger.exception(f"通信分析失败: {e}")
            return SkillResult(
                skill_name=self.name,
                skill_type=self.skill_type,
                success=False,
                error=str(e),
            )

    def _analyze_comm_ops(self, comm_ops: List[Dict]) -> Dict[str, int]:
        """分析通信算子"""
        op_stats = {}
        if not comm_ops:
            return op_stats

        for op in comm_ops:
            if isinstance(op, dict):
                name = op.get("name", "unknown")
            else:
                name = str(op)

            # 提取算子类型
            if "all_reduce" in name.lower():
                op_type = "all_reduce"
            elif "all_gather" in name.lower():
                op_type = "all_gather"
            elif "reduce_scatter" in name.lower():
                op_type = "reduce_scatter"
            elif "broadcast" in name.lower():
                op_type = "broadcast"
            else:
                op_type = "other"

            op_stats[op_type] = op_stats.get(op_type, 0) + 1

        return op_stats

    def _evaluate_efficiency(self, bandwidth_gbps: float) -> str:
        """评估通信效率"""
        if bandwidth_gbps >= 50:
            return "优秀"
        elif bandwidth_gbps >= 40:
            return "良好"
        elif bandwidth_gbps >= 30:
            return "一般"
        else:
            return "待优化"