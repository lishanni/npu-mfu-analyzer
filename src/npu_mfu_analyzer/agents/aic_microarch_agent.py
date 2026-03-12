"""
AIC 微架构分析 Agent

执行 AIC (AI Core) 微架构深度分析，包括指令级、内存层次、流水线三个维度。
"""

import logging
from typing import Dict, Any, List, Optional

from npu_mfu_analyzer.agents.base_agent import BaseAgent, AnalysisResult
from npu_mfu_analyzer.llm.llm_interface import LLMInterface
from npu_mfu_analyzer.data_loader.aic_metrics import (
    analyze_aic_microarchitecture,
    load_aic_pmu_from_profiling,
    ExtendedAICMetrics,
    DeepBottleneckAnalysis,
)
from npu_mfu_analyzer.analyzers.aic.instruction_analyzer import InstructionAnalyzer, InstructionBottleneck
from npu_mfu_analyzer.analyzers.aic.memory_hierarchy_analyzer import MemoryHierarchyAnalyzer, MemoryHierarchyAnalysis
from npu_mfu_analyzer.analyzers.aic.pipeline_analyzer import PipelineAnalyzer, PipelineAnalysis

logger = logging.getLogger(__name__)


class AICMicroarchAgent(BaseAgent):
    """
    AIC 微架构分析 Agent

    执行 AI Core 的硬件级深度分析，提供三个维度的瓶颈诊断和优化建议。
    数据来源于 msprof AIC PMU 事件。
    """

    def __init__(self, llm: LLMInterface, config: Optional[Dict[str, Any]] = None):
        """
        初始化 AIC 微架构分析 Agent

        Args:
            llm: LLM 接口
            config: 配置字典
        """
        super().__init__(llm, config)
        self.instruction_analyzer = InstructionAnalyzer()
        self.memory_analyzer = MemoryHierarchyAnalyzer()
        self.pipeline_analyzer = PipelineAnalyzer()

    async def analyze(self, data: Dict[str, Any]) -> AnalysisResult:
        """
        执行 AIC 微架构分析

        Args:
            data: 包含 profiling_path 的数据字典

        Returns:
            AnalysisResult: 分析结果
        """
        try:
            profiling_path = data.get("profiling_path", "")

            if not profiling_path:
                return AnalysisResult(
                    agent_name="AICMicroarchAgent",
                    success=False,
                    summary="未提供 profiling 数据路径",
                    error="缺少 profiling_path 参数"
                )

            logger.info(f"Starting AIC microarchitecture analysis: {profiling_path}")

            # 加载 PMU 数据
            metrics_list = load_aic_pmu_from_profiling(profiling_path, limit=100)

            if not metrics_list:
                return AnalysisResult(
                    agent_name="AICMicroarchAgent",
                    success=False,
                    summary="未找到 AIC PMU 数据",
                    error="未找到 AIC PMU 数据，请使用 msprof op --aic-metrics 采集数据"
                )

            logger.info(f"Loaded {len(metrics_list)} operator PMU metrics")

            # 执行三个维度的分析
            instruction_bottlenecks, instruction_summary = self.instruction_analyzer.analyze_batch(metrics_list)
            memory_analyses, memory_summary = self.memory_analyzer.analyze_batch(metrics_list)
            pipeline_analyses, pipeline_summary = self.pipeline_analyzer.analyze_batch(metrics_list)

            # 生成综合分析
            result = await self._generate_comprehensive_analysis(
                metrics_list,
                instruction_bottlenecks,
                instruction_summary,
                memory_analyses,
                memory_summary,
                pipeline_analyses,
                pipeline_summary,
            )

            return AnalysisResult(
                agent_name="AICMicroarchAgent",
                success=True,
                summary=f"AIC 微架构分析完成：{len(metrics_list)} 个算子，{instruction_summary.get('critical_count', 0)} 个严重瓶颈",
                details=result,
                recommendations=result.get("recommendations", []),
            )

        except Exception as e:
            logger.error(f"AIC microarchitecture analysis failed: {e}", exc_info=True)
            return AnalysisResult(
                agent_name="AICMicroarchAgent",
                success=False,
                summary="AIC 微架构分析失败",
                error=str(e)
            )

    async def _generate_comprehensive_analysis(
        self,
        metrics_list: List[ExtendedAICMetrics],
        instruction_bottlenecks: List[InstructionBottleneck],
        instruction_summary: Dict[str, Any],
        memory_analyses: List[MemoryHierarchyAnalysis],
        memory_summary: Dict[str, Any],
        pipeline_analyses: List[PipelineAnalysis],
        pipeline_summary: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        生成综合分析结果

        Returns:
            分析结果字典
        """
        result = {
            "analyzed_count": len(metrics_list),
            "instruction_summary": instruction_summary,
            "memory_summary": memory_summary,
            "pipeline_summary": pipeline_summary,
            "recommendations": [],
        }

        # 统计严重瓶颈
        critical_bottlenecks = []
        for ib in instruction_bottlenecks:
            if ib.severity == "critical":
                critical_bottlenecks.append({
                    "type": "instruction",
                    "bottleneck": ib.bottleneck_type.value,
                    "score": ib.score,
                })

        for ma in memory_analyses:
            if ma.severity == "critical":
                critical_bottlenecks.append({
                    "type": "memory",
                    "bottleneck": ma.bottleneck_type.value,
                    "score": ma.score,
                })

        for pa in pipeline_analyses:
            if pa.severity == "critical":
                critical_bottlenecks.append({
                    "type": "pipeline",
                    "bottleneck": pa.bottleneck_type.value,
                    "score": pa.score,
                })

        # 排序获取主要瓶颈
        critical_bottlenecks.sort(key=lambda x: x["score"], reverse=True)

        # 生成优化建议
        result["recommendations"] = self._generate_recommendations(
            critical_bottlenecks,
            instruction_bottlenecks,
            memory_analyses,
            pipeline_analyses,
        )

        # Top 瓶颈详情
        result["top_critical_bottlenecks"] = critical_bottlenecks[:5]

        return result

    def _generate_recommendations(
        self,
        critical_bottlenecks: List[Dict[str, Any]],
        instruction_bottlenecks: List[InstructionBottleneck],
        memory_analyses: List[MemoryHierarchyAnalysis],
        pipeline_analyses: List[PipelineAnalysis],
    ) -> List[str]:
        """生成综合优化建议"""
        recommendations = []

        # 按瓶颈类型统计
        bottleneck_types = {}
        for b in critical_bottlenecks:
            key = f"{b['type']}_{b['bottleneck']}"
            bottleneck_types[key] = bottleneck_types.get(key, 0) + 1

        # 主要瓶颈建议
        if bottleneck_types:
            top_bottleneck = max(bottleneck_types.items(), key=lambda x: x[1])
            if "instruction_cube_underutilized" in top_bottleneck[0]:
                recommendations.append(
                    "【关键】多个算子 Cube 利用率低，建议优化数据布局和计算密度"
                )
            elif "memory_l2_miss" in top_bottleneck[0]:
                recommendations.append(
                    "【关键】多个算子 L2 缓存命中率低，建议优化数据访问模式"
                )
            elif "pipeline_dependency_stall" in top_bottleneck[0]:
                recommendations.append(
                    "【关键】多个算子存在依赖停顿，建议优化指令调度"
                )

        # 收集各分析器的建议
        seen = set()
        for ib in instruction_bottlenecks[:3]:
            for rec in ib.recommendations[:2]:
                if rec not in seen:
                    recommendations.append(rec)
                    seen.add(rec)

        for ma in memory_analyses[:3]:
            for rec in ma.recommendations[:2]:
                if rec not in seen:
                    recommendations.append(rec)
                    seen.add(rec)

        for pa in pipeline_analyses[:3]:
            for rec in pa.recommendations[:2]:
                if rec not in seen:
                    recommendations.append(rec)
                    seen.add(rec)

        return recommendations[:15]

    def to_prompt_text(self, analysis_result: Dict[str, Any]) -> str:
        """转换为 LLM Prompt 格式"""
        lines = [
            "## AIC 微架构深度分析",
            "",
        ]

        if "analyzed_count" in analysis_result:
            lines.append(f"- 分析算子数: {analysis_result['analyzed_count']}")

        if "instruction_summary" in analysis_result:
            summary = analysis_result["instruction_summary"]
            lines.extend([
                "- 指令级瓶颈:",
                f"  - 严重: {summary.get('critical_count', 0)}",
                f"  - 高优先级: {summary.get('high_count', 0)}",
                f"  - 中等优先级: {summary.get('medium_count', 0)}",
            ])

        if "memory_summary" in analysis_result:
            summary = analysis_result["memory_summary"]
            lines.extend([
                "- 内存层次:",
                f"  - L2 平均命中率: {summary.get('avg_l2_hit_rate', 0):.1f}%",
                f"  - 局部性评分: {summary.get('avg_locality_score', 0):.1f}",
            ])

        if "pipeline_summary" in analysis_result:
            summary = analysis_result["pipeline_summary"]
            lines.extend([
                "- 流水线:",
                f"  - 平均利用率: {summary.get('avg_pipe_util', 0):.1f}%",
                f"  - 平均停顿率: {summary.get('avg_stall_rate', 0):.1f}%",
            ])

        if "top_critical_bottlenecks" in analysis_result:
            lines.append("")
            lines.append("### 严重瓶颈 Top 5")
            for i, b in enumerate(analysis_result["top_critical_bottlenecks"][:5], 1):
                lines.append(f"{i}. {b['type']} - {b['bottleneck']} (评分: {b['score']:.0f})")

        if "recommendations" in analysis_result:
            lines.append("")
            lines.append("### 优化建议")
            for i, rec in enumerate(analysis_result["recommendations"][:10], 1):
                lines.append(f"{i}. {rec}")

        return "\n".join(lines)