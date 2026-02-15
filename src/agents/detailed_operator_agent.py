"""
Detailed Operator Agent

基于 AIC metrics 的详细算子分析，识别硬件瓶颈，生成优化建议。
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field

from src.agents.base_agent import BaseAgent, AnalysisResult
from src.llm.llm_interface import LLMInterface
from src.llm.prompts import MFU_ANALYSIS_SYSTEM
from src.data_loader.profiling_loader import ProfilingLoader
from src.data_loader.aic_metrics import (
    AICMetrics,
    AICAnalysisResult,
    DetailedOperatorAnalysisData,
    BOTTLENECK_COMPUTE,
    BOTTLENECK_MEMORY,
    BOTTLENECK_PIPELINE,
    BOTTLENECK_BALANCED,
    SEVERITY_CRITICAL,
    SEVERITY_HIGH,
    SEVERITY_MEDIUM,
    SEVERITY_LOW,
    CRITICAL_THRESHOLD,
    HIGH_THRESHOLD,
    MEDIUM_THRESHOLD,
)

logger = logging.getLogger(__name__)


class DetailedOperatorAgent(BaseAgent):
    """
    详细算子分析 Agent

    功能：
    1. 解析 AIC metrics 数据
    2. 识别硬件瓶颈 (计算/内存/流水线)
    3. 生成针对性优化建议
    4. 为 AIKG 提供硬件约束信息
    """

    PROMPT_TEMPLATE = """
你是昇腾 NPU 硬件优化专家。分析以下算子的详细硬件指标：

{data_summary}

## 分析任务
1. **瓶颈识别**：
   - 计算瓶颈: Cube/Vector/Scalar 利用率低
   - 内存瓶颈: L2 缓存命中率低、Buffer 使用率高
   - 流水线瓶颈: 流水线利用率低、停顿率高

2. **根因分析**：
   - 为什么会出现这个瓶颈？
   - 数据访问模式是否合理？
   - 算子实现是否需要优化？

3. **优化建议**：
   - 算子融合建议
   - 数据类型优化
   - 内存布局优化
   - 调度策略优化

请给出详细的分析结论和具体优化建议。
"""

    def __init__(
        self,
        llm: LLMInterface,
        config: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            name="DetailedOperatorAgent",
            llm=llm,
            system_prompt=MFU_ANALYSIS_SYSTEM,
            config=config
        )
        self._profiling_loader: Optional[ProfilingLoader] = None

    def get_prompt_template(self) -> str:
        return self.PROMPT_TEMPLATE

    async def analyze(self, data: Dict[str, Any]) -> AnalysisResult:
        """
        分析算子详细性能数据

        Args:
            data: 包含以下字段：
                - profiling_path: Profiling 数据路径
                - operator_names: 需要分析的算子名称列表 (可选)
                - top_operators: Top 耗时算子列表 (可选)

        Returns:
            AnalysisResult
        """
        try:
            # 1. 初始化 ProfilingLoader
            profiling_path = data.get("profiling_path")
            if not profiling_path:
                return AnalysisResult(
                    agent_name=self.name,
                    success=False,
                    summary="缺少 profiling_path 参数",
                    error="Missing profiling_path in data"
                )

            self._profiling_loader = ProfilingLoader(profiling_path)

            # 2. 获取需要分析的算子列表
            operator_names = self._get_operator_names(data)
            if not operator_names:
                return AnalysisResult(
                    agent_name=self.name,
                    success=False,
                    summary="没有需要分析的算子",
                    error="No operators to analyze"
                )

            # 3. 加载 AIC metrics
            aic_metrics_dict = self._profiling_loader.get_aic_metrics()
            if not aic_metrics_dict:
                return AnalysisResult(
                    agent_name=self.name,
                    success=False,
                    summary="未找到 AIC metrics 数据",
                    details={"suggestion": "请使用 msprof op --aic-metrics 采集数据"},
                    error="No AIC metrics found. Please run with --aic-metrics enabled."
                )

            # 4. 分析每个算子
            analysis_results = []
            for op_name in operator_names:
                result = self._analyze_operator(op_name, aic_metrics_dict)
                if result:
                    analysis_results.append(result)

            if not analysis_results:
                return AnalysisResult(
                    agent_name=self.name,
                    success=False,
                    summary="没有找到匹配的 AIC metrics 数据",
                    error="No matching AIC metrics found for the provided operators"
                )

            # 5. 生成综合分析报告
            data_summary = self._format_analysis_summary(analysis_results)
            prompt = self.format_prompt(self.PROMPT_TEMPLATE, data_summary=data_summary)
            response = await self.call_llm(prompt)

            # 6. 构建结果
            critical_count = sum(1 for r in analysis_results if r.severity == SEVERITY_CRITICAL)
            high_count = sum(1 for r in analysis_results if r.severity == SEVERITY_HIGH)

            return AnalysisResult(
                agent_name=self.name,
                success=True,
                summary=(
                    f"分析 {len(analysis_results)} 个算子, "
                    f"{critical_count} 个严重瓶颈, "
                    f"{high_count} 个高优先级瓶颈"
                ),
                details={
                    "analysis_results": [
                        {
                            "operator": r.operator_name,
                            "bottleneck": r.bottleneck_type,
                            "severity": r.severity,
                            "diagnosis": r.diagnosis,
                        }
                        for r in analysis_results
                    ],
                },
                recommendations=self._extract_recommendations(response),
                raw_response=response,
            )

        except Exception as e:
            logger.error(f"Detailed operator analysis failed: {e}", exc_info=True)
            return AnalysisResult(
                agent_name=self.name,
                success=False,
                summary="详细算子分析失败",
                error=str(e),
            )

    def _get_operator_names(self, data: Dict[str, Any]) -> List[str]:
        """提取需要分析的算子名称列表"""
        # 优先使用指定的 operator_names
        if "operator_names" in data:
            return data["operator_names"]

        # 其次使用 top_operators
        if "top_operators" in data:
            return [op.get("name", "") for op in data["top_operators"][:10]]

        # 如果都没有，从 profiling_loader 获取 top kernels
        if self._profiling_loader:
            top_kernels = self._profiling_loader.get_top_kernels(top_n=10)
            return [k["name"] for k in top_kernels]

        return []

    def _analyze_operator(
        self,
        op_name: str,
        aic_metrics_dict: Dict[str, AICMetrics]
    ) -> Optional[AICAnalysisResult]:
        """
        分析单个算子

        Args:
            op_name: 算子名称
            aic_metrics_dict: AIC 指标字典

        Returns:
            AICAnalysisResult
        """
        # 查找匹配的 AIC metrics
        metrics = self._find_metrics(op_name, aic_metrics_dict)
        if not metrics:
            logger.debug(f"No AIC metrics found for {op_name}")
            return None

        # 识别瓶颈类型和严重程度
        bottleneck_type, severity = self._identify_bottleneck(metrics)

        # 生成诊断结果
        diagnosis = self._generate_diagnosis(metrics, bottleneck_type)

        # 生成优化建议
        recommendations = self._generate_recommendations(metrics, bottleneck_type)

        # 生成 AIKG 提示词
        aikg_prompts = self._generate_aikg_prompts(metrics, bottleneck_type)

        return AICAnalysisResult(
            operator_name=op_name,
            bottleneck_type=bottleneck_type,
            severity=severity,
            metrics=metrics,
            diagnosis=diagnosis,
            recommendations=recommendations,
            aikg_prompts=aikg_prompts,
        )

    def _find_metrics(
        self,
        op_name: str,
        aic_metrics_dict: Dict[str, AICMetrics]
    ) -> Optional[AICMetrics]:
        """查找匹配的 AIC metrics (支持模糊匹配)"""
        # 精确匹配
        if op_name in aic_metrics_dict:
            return aic_metrics_dict[op_name]

        # 基础名称匹配
        if self._profiling_loader:
            base_name = self._profiling_loader._extract_base_op_name(op_name)
            if base_name in aic_metrics_dict:
                return aic_metrics_dict[base_name]

        # 模糊匹配: 找包含算子类型的第一个
        for key, metrics in aic_metrics_dict.items():
            if op_name in key or key in op_name:
                return metrics

        return None

    def _identify_bottleneck(
        self,
        metrics: AICMetrics
    ) -> Tuple[str, str]:
        """
        识别瓶颈类型和严重程度

        Returns:
            (bottleneck_type, severity)
            - bottleneck_type: "compute", "memory", "pipeline", "balanced"
            - severity: "critical", "high", "medium", "low"
        """
        # 提取关键指标
        cube_util = metrics.arithmetic.cube_utilization if metrics.arithmetic else 100.0
        l2_hit_rate = metrics.memory.l2_cache_hit_rate if metrics.memory else 100.0
        pipe_util = metrics.pipeline.pipe_utilization if metrics.pipeline else 100.0
        stall_rate = metrics.pipeline.stall_rate if metrics.pipeline else 0.0

        # 瓶颈判断阈值
        # 判断瓶颈类型
        if cube_util < CRITICAL_THRESHOLD:
            bottleneck_type = BOTTLENECK_COMPUTE
            severity = SEVERITY_CRITICAL
        elif l2_hit_rate < CRITICAL_THRESHOLD:
            bottleneck_type = BOTTLENECK_MEMORY
            severity = SEVERITY_CRITICAL
        elif stall_rate > 80.0:
            bottleneck_type = BOTTLENECK_PIPELINE
            severity = SEVERITY_CRITICAL
        elif cube_util < HIGH_THRESHOLD:
            bottleneck_type = BOTTLENECK_COMPUTE
            severity = SEVERITY_HIGH
        elif l2_hit_rate < HIGH_THRESHOLD:
            bottleneck_type = BOTTLENECK_MEMORY
            severity = SEVERITY_HIGH
        elif pipe_util < MEDIUM_THRESHOLD:
            bottleneck_type = BOTTLENECK_PIPELINE
            severity = SEVERITY_MEDIUM
        else:
            bottleneck_type = BOTTLENECK_BALANCED
            severity = SEVERITY_LOW

        return bottleneck_type, severity

    def _generate_diagnosis(
        self,
        metrics: AICMetrics,
        bottleneck_type: str
    ) -> List[str]:
        """生成诊断结果"""
        diagnosis = []

        if bottleneck_type == BOTTLENECK_COMPUTE and metrics.arithmetic:
            cube_util = metrics.arithmetic.cube_utilization
            vector_util = metrics.arithmetic.vector_utilization

            if cube_util < CRITICAL_THRESHOLD:
                diagnosis.append(
                    f"Cube 单元利用率极低 ({cube_util:.1f}%)，计算资源严重浪费"
                )
            elif cube_util < HIGH_THRESHOLD:
                diagnosis.append(
                    f"Cube 单元利用率偏低 ({cube_util:.1f}%)，未充分发挥计算能力"
                )

            if vector_util > cube_util and vector_util > 50:
                diagnosis.append(
                    f"Vector 利用率 ({vector_util:.1f}%) 高于 Cube，可能存在计算模式不匹配"
                )

        elif bottleneck_type == BOTTLENECK_MEMORY and metrics.memory:
            l2_hit = metrics.memory.l2_cache_hit_rate
            ub_usage = metrics.memory.ub_usage

            if l2_hit < CRITICAL_THRESHOLD:
                diagnosis.append(
                    f"L2 缓存命中率极低 ({l2_hit:.1f}%)，存在大量内存访问延迟"
                )
            elif l2_hit < HIGH_THRESHOLD:
                diagnosis.append(
                    f"L2 缓存命中率偏低 ({l2_hit:.1f}%)，数据局部性不佳"
                )

            if ub_usage > 80:
                diagnosis.append(
                    f"Unified Buffer 使用率过高 ({ub_usage:.1f}%)，可能导致溢出到外部内存"
                )

        elif bottleneck_type == BOTTLENECK_PIPELINE and metrics.pipeline:
            pipe_util = metrics.pipeline.pipe_utilization
            stall_rate = metrics.pipeline.stall_rate
            conflict_ratio = metrics.pipeline.resource_conflict_ratio

            if stall_rate > 50:
                diagnosis.append(
                    f"流水线停顿率过高 ({stall_rate:.1f}%)，指令调度存在严重问题"
                )

            if conflict_ratio > 30:
                diagnosis.append(
                    f"资源冲突率过高 ({conflict_ratio:.1f}%)，存在资源竞争"
                )

        if not diagnosis:
            diagnosis.append("各项指标较为均衡，无明显瓶颈")

        return diagnosis

    def _generate_recommendations(
        self,
        metrics: AICMetrics,
        bottleneck_type: str
    ) -> List[str]:
        """生成优化建议"""
        recommendations = []

        if bottleneck_type == BOTTLENECK_COMPUTE:
            recommendations.extend([
                "考虑使用算子融合减少内存访问",
                "优化数据布局以提高计算密度",
                "检查是否可以使用更高维度的计算指令",
                "考虑调整 batch size 或 tile size 以提高并行度",
            ])
        elif bottleneck_type == BOTTLENECK_MEMORY:
            recommendations.extend([
                "优化数据访问模式以提高缓存命中率",
                "考虑使用数据预取技术",
                "优化内存布局以提高数据局部性",
                "考虑使用数据类型转换减少内存带宽压力",
            ])
        elif bottleneck_type == BOTTLENECK_PIPELINE:
            recommendations.extend([
                "优化指令调度以减少停顿",
                "考虑使用指令重排以减少资源冲突",
                "优化数据依赖以减少流水线气泡",
            ])
        else:
            recommendations.extend([
                "当前性能较为均衡，可以考虑进一步优化微架构细节",
            ])

        return recommendations

    def _generate_aikg_prompts(
        self,
        metrics: AICMetrics,
        bottleneck_type: str
    ) -> Dict[str, str]:
        """生成 AIKG 优化提示词"""
        prompts = {}

        # 硬件约束提示词
        hardware_constraints = []

        if metrics.arithmetic:
            hardware_constraints.append(
                f"Cube 利用率: {metrics.arithmetic.cube_utilization:.1f}%"
            )

        if metrics.memory:
            hardware_constraints.append(
                f"L2 缓存命中率: {metrics.memory.l2_cache_hit_rate:.1f}%"
            )

        if metrics.pipeline:
            hardware_constraints.append(
                f"流水线利用率: {metrics.pipeline.pipe_utilization:.1f}%"
            )

        prompts["hardware_constraints"] = "\n".join(hardware_constraints)

        # 优化方向提示词
        if bottleneck_type == BOTTLENECK_COMPUTE:
            prompts["optimization_direction"] = (
                "重点优化计算密度，考虑使用更高维度的计算指令，"
                "优化数据布局以减少非计算操作"
            )
        elif bottleneck_type == BOTTLENECK_MEMORY:
            prompts["optimization_direction"] = (
                "重点优化数据访问模式，提高缓存命中率，"
                "考虑使用数据预取和更好的内存布局"
            )
        elif bottleneck_type == BOTTLENECK_PIPELINE:
            prompts["optimization_direction"] = (
                "重点优化指令调度，减少流水线停顿，"
                "优化数据依赖以减少资源冲突"
            )
        else:
            prompts["optimization_direction"] = "综合优化计算和内存访问"

        return prompts

    def _format_analysis_summary(
        self,
        analysis_results: List[AICAnalysisResult]
    ) -> str:
        """格式化分析摘要"""
        lines = [
            "## 详细算子硬件分析结果",
            f"分析 {len(analysis_results)} 个算子",
            "",
        ]

        # 按严重程度分组
        critical = [r for r in analysis_results if r.severity == SEVERITY_CRITICAL]
        high = [r for r in analysis_results if r.severity == SEVERITY_HIGH]
        medium = [r for r in analysis_results if r.severity == SEVERITY_MEDIUM]

        if critical:
            lines.append("### 严重瓶颈算子")
            for r in critical:
                lines.append(f"- **{r.operator_name}** ({r.bottleneck_type})")
                for diag in r.diagnosis[:2]:
                    lines.append(f"  - {diag}")
            lines.append("")

        if high:
            lines.append("### 高优先级瓶颈")
            for r in high:
                lines.append(f"- **{r.operator_name}** ({r.bottleneck_type})")
            lines.append("")

        if medium:
            lines.append(f"### 中等优先级瓶颈 ({len(medium)} 个)")
            for r in medium[:5]:
                lines.append(f"- {r.operator_name}")

        return "\n".join(lines)


@dataclass
class DetailedOperatorAnalysisData:
    """详细算子分析数据"""
    profiling_path: str
    analyzed_operators: List[str]
    analysis_results: List[AICAnalysisResult]

    # 统计信息
    critical_count: int = 0
    high_count: int = 0
    medium_count: int = 0

    def __post_init__(self):
        """计算统计信息"""
        self.critical_count = sum(
            1 for r in self.analysis_results if r.severity == SEVERITY_CRITICAL
        )
        self.high_count = sum(
            1 for r in self.analysis_results if r.severity == SEVERITY_HIGH
        )
        self.medium_count = sum(
            1 for r in self.analysis_results if r.severity == SEVERITY_MEDIUM
        )

    def to_prompt_text(self) -> str:
        """转换为 LLM Prompt 格式"""
        lines = [
            "## 详细算子分析数据摘要",
            f"- 分析算子数: {len(self.analyzed_operators)}",
            f"- 严重瓶颈: {self.critical_count}",
            f"- 高优先级: {self.high_count}",
            f"- 中等优先级: {self.medium_count}",
            "",
        ]

        # Top 严重瓶颈
        critical_results = [
            r for r in self.analysis_results if r.severity == SEVERITY_CRITICAL
        ]
        if critical_results:
            lines.append("### Top 严重瓶颈算子")
            for r in critical_results[:5]:
                lines.append(f"- **{r.operator_name}** ({r.bottleneck_type})")
                if r.metrics.arithmetic:
                    lines.append(
                        f"  - Cube 利用率: {r.metrics.arithmetic.cube_utilization:.1f}%"
                    )
                if r.metrics.memory:
                    lines.append(
                        f"  - L2 缓存命中率: {r.metrics.memory.l2_cache_hit_rate:.1f}%"
                    )
                for diag in r.diagnosis[:2]:
                    lines.append(f"  - {diag}")
            lines.append("")

        return "\n".join(lines)
