"""
详细算子分析 Agent V2

基于扩展 AIC metrics 的深度算子瓶颈分析，- 指令级分析：Cube/Vector/Scalar 指令混合比
- 内存层次分析：L2/UB/L0 己度分析
- 流水线停顿分析：停顿原因细分
- 深度瓶颈识别：多维度综合判断
- 针对性优化建议生成
"""

from __future__ import annotations

import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field

from npu_mfu_analyzer.agents.base_agent import BaseAgent, AnalysisResult
from npu_mfu_analyzer.llm.llm_interface import LLMInterface
from npu_mfu_analyzer.llm.prompts import MFU_ANALYSIS_SYSTEM
from npu_mfu_analyzer.data_loader.aic_metrics import (
    AICMetrics,
    ExtendedAICMetrics,
    ExtendedArithmeticUtilization,
    ExtendedMemoryMetrics,
    ExtendedPipelineMetrics,
    DeepBottleneckAnalysis,
    BottleneckCategory,
    ComputeBottleneckType,
    MemoryBottleneckType,
    PipelineBottleneckType,
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


# ============================================================================
# 优化策略知识库
# ============================================================================

OPTIMIZATION_STRATEGIES = {
    # 计算瓶颈优化策略
    "compute_bound": {
        "cube_underutilized": [
            "考虑使用更高维度的矩阵乘法以充分利用 Cube 单元",
            "优化数据布局（如 NCHW → NC1HWC0）以提高计算密度",
            "检查是否可以合并多个小矩阵乘法为一个大矩阵乘法",
            "调整 tile size 以增加 Cube 的计算负载",
        ],
        "vector_heavy": [
            "评估是否可以将部分 Vector 操作转为 Cube 操作（如使用矩阵乘法替代向量运算）",
            "优化向量运算的并行度，充分利用 Vector 单元",
            "检查是否存在不必要的逐元素操作",
        ],
        "low_issue_rate": [
            "检查是否存在频繁的 kernel launch 开销，考虑使用算子融合",
            "优化内存访问模式以减少等待时间",
            "检查是否存在过多的同步操作",
        ],
        "instruction_imbalance": [
            "优化指令调度以平衡 Cube/Vector/Scalar 负载",
            "检查是否可以通过算子融合减少指令数量",
        ],
    },

    # 内存瓶颈优化策略
    "memory_bound": {
        "l2_miss": [
            "优化数据访问模式以提高缓存命中率（如分块访问、数据预取）",
            "调整 tile size 以增加数据复用",
            "考虑使用更高效的内存布局",
            "检查是否存在不必要的跨步访问",
        ],
        "ub_pressure": [
            "减小 tile size 以降低 Unified Buffer 压力",
            "考虑使用 double buffering 技术",
            "优化数据分块策略以减少 UB 溢出",
            "检查是否可以减少中间结果的存储",
        ],
        "hbm_saturated": [
            "使用更高效的数据类型（如 FP16/BF16 代替 FP32）",
            "优化数据布局以减少访存次数",
            "考虑使用算子融合减少中间结果的存储",
            "检查是否存在冗余的数据传输",
        ],
        "poor_locality": [
            "优化数据访问模式以提高空间局部性",
            "考虑调整内存布局（如行主序 vs 列主序）",
            "使用数据预取技术",
        ],
    },

    # 流水线瓶颈优化策略
    "pipeline_bound": {
        "mte_stall": [
            "优化 MTE（Memory Transfer Engine）单元的数据调度",
            "使用数据预取隐藏内存延迟",
            "调整内存访问模式以减少 MTE 等待",
        ],
        "dependency_stall": [
            "优化指令调度以减少数据依赖",
            "考虑使用循环展开增加指令级并行",
            "检查是否可以重排操作以减少依赖等待",
        ],
        "resource_conflict": [
            "调整计算资源分配以减少冲突",
            "优化流水线调度策略",
            "检查是否存在过多的资源竞争",
        ],
        "memory_stall": [
            "优化内存访问模式以减少等待",
            "使用 double buffering 隐藏内存延迟",
            "考虑使用更高的缓存利用率",
        ],
        "sync_stall": [
            "减少不必要的同步操作",
            "优化同步点的位置",
            "考虑使用异步执行",
        ],
    },

    # 均衡状态的优化建议
    "balanced": [
        "当前性能较为均衡，可考虑进一步优化微架构细节",
        "检查是否可以通过算子融合减少 launch 开销",
        "评估是否可以使用更高效的算子实现",
        "考虑调整全局参数（如 batch size、sequence length）",
    ],

    # 混合瓶颈的优化建议
    "mixed": [
        "存在多维度瓶颈，建议优先解决最主要的瓶颈",
        "考虑综合优化策略，避免优化一个维度导致另一个维度恶化",
        "分步骤进行优化，每次优化后重新评估",
    ],
}


# ============================================================================
# 详细算子分析 Agent V2
# ============================================================================

class DetailedOperatorAgentV2(BaseAgent):
    """
    详细算子分析 Agent V2

    基于扩展 AIC metrics 的深度算子瓶颈分析

    功能：
    1. 指令级分析：分析 Cube/Vector/Scalar 指令混合比
    2. 内存层次分析：L2 Cache / UB / L0 深度分析
    3. 流水线停顿分析：停顿原因细分（MTE/依赖/内存/同步）
    4. 深度瓶颈识别：多维度综合判断瓶颈类型
    5. 针对性优化建议：基于瓶颈类型生成具体建议
    """

    PROMPT_TEMPLATE = """
你是昇腾 NPU 硬件优化专家。分析以下算子的详细硬件指标：

{data_summary}

## 分析任务

### 1. 瓶颈识别
分析以下维度的瓶颈：
- **计算瓶颈**: Cube/Vector/Scalar 利用率是否合理？指令发射效率如何？
- **内存瓶颈**: L2 缓存命中率是否正常？UB/L0 压力如何？是否存在内存带宽饱和？
- **流水线瓶颈**: 流水线停顿率是否过高？主要停顿原因是什么？

### 2. 根因分析
- 为什么会出现这个瓶颈？
- 数据访问模式是否合理？
- 算子实现是否存在优化空间？
- 是否存在资源竞争或依赖问题？

### 3. 优化建议
针对识别出的瓶颈，给出具体优化建议：
- 算子融合建议
- 数据类型/布局优化
- Tiling 策略调整
- 调度策略优化

请给出详细的分析结论和具体优化建议。
"""

    def __init__(
        self,
        llm: LLMInterface,
        config: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            name="DetailedOperatorAgentV2",
            llm=llm,
            system_prompt=MFU_ANALYSIS_SYSTEM,
            config=config
        )

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
                - aic_metrics_dict: 预加载的 AIC metrics 字典 (可选)

        Returns:
            AnalysisResult
        """
        try:
            # 1. 获取 AIC metrics
            aic_metrics_dict = self._get_aic_metrics(data)

            if not aic_metrics_dict:
                return AnalysisResult(
                    agent_name=self.name,
                    success=False,
                    summary="未找到 AIC metrics 数据",
                    details={"suggestion": "请使用 --aic-metrics 选项采集数据"},
                    error="No AIC metrics found"
                )

            # 2. 获取要分析的算子列表
            operator_names = self._get_operator_names(data, aic_metrics_dict)

            if not operator_names:
                return AnalysisResult(
                    agent_name=self.name,
                    success=False,
                    summary="没有需要分析的算子",
                    error="No operators to analyze"
                )

            # 3. 执行深度分析
            analysis_results = []
            for op_name in operator_names:
                metrics = aic_metrics_dict.get(op_name)
                if not metrics:
                    continue

                # 转换为扩展指标（如果需要）
                extended_metrics = self._ensure_extended_metrics(metrics)

                # 执行深度分析
                result = self._analyze_operator_deep(op_name, extended_metrics)
                if result:
                    analysis_results.append(result)

            if not analysis_results:
                return AnalysisResult(
                    agent_name=self.name,
                    success=False,
                    summary="没有找到匹配的 AIC metrics 数据",
                    error="No matching AIC metrics found"
                )

            # 4. 生成综合分析报告
            data_summary = self._format_analysis_summary(analysis_results)
            prompt = self.format_prompt(
                self.PROMPT_TEMPLATE,
                data_summary=data_summary
            )
            response = await self.call_llm(prompt)

            # 5. 构建结果
            critical_count = sum(
                1 for r in analysis_results
                if r.deep_bottleneck and r.deep_bottleneck.severity_score >= 60
            )
            high_count = sum(
                1 for r in analysis_results
                if r.deep_bottleneck and 40 <= r.deep_bottleneck.severity_score < 60
            )

            return AnalysisResult(
                agent_name=self.name,
                success=True,
                summary=(
                    f"深度分析 {len(analysis_results)} 个算子, "
                    f"{critical_count} 个严重瓶颈, "
                    f"{high_count} 个高优先级瓶颈"
                ),
                details={
                    "analysis_results": [
                        {
                            "operator": r.op_name,
                            "bottleneck_category": r.deep_bottleneck.bottleneck_category if r.deep_bottleneck else "unknown",
                            "severity_score": r.deep_bottleneck.severity_score if r.deep_bottleneck else 0,
                            "primary_optimization": r.deep_bottleneck.primary_optimization if r.deep_bottleneck else None,
                        }
                        for r in analysis_results
                    ],
                },
                recommendations=self._extract_recommendations(response),
                raw_response=response,
            )

        except Exception as e:
            logger.error(f"Detailed operator analysis V2 failed: {e}", exc_info=True)
            return AnalysisResult(
                agent_name=self.name,
                success=False,
                summary="详细算子分析失败",
                error=str(e),
            )

    def _get_aic_metrics(self, data: Dict[str, Any]) -> Dict[str, AICMetrics]:
        """获取 AIC metrics 字典"""
        # 优先使用预加载的数据
        if "aic_metrics_dict" in data:
            return data["aic_metrics_dict"]

        # 从 profiling loader 加载
        if "profiling_loader" in data:
            loader = data["profiling_loader"]
            return loader.get_aic_metrics()

        return {}

    def _get_operator_names(
        self,
        data: Dict[str, Any],
        aic_metrics_dict: Dict[str, AICMetrics]
    ) -> List[str]:
        """获取要分析的算子列表"""
        # 优先使用指定的算子列表
        if "operator_names" in data:
            return data["operator_names"]

        # 使用 top_operators
        if "top_operators" in data:
            return [op.get("name", "") for op in data["top_operators"][:10]]

        # 使用全部可用的算子
        return list(aic_metrics_dict.keys())[:20]

    def _ensure_extended_metrics(self, metrics: AICMetrics) -> ExtendedAICMetrics:
        """确保使用扩展指标结构"""
        if isinstance(metrics, ExtendedAICMetrics):
            return metrics

        # 转换为扩展指标
        return ExtendedAICMetrics(
            op_name=metrics.op_name,
            op_type=metrics.op_type,
            duration_us=metrics.duration_us,
            arithmetic=metrics.arithmetic,
            memory=metrics.memory,
            pipeline=metrics.pipeline,
            raw_data=metrics.raw_data,
        )

    def _analyze_operator_deep(
        self,
        op_name: str,
        metrics: ExtendedAICMetrics
    ) -> Optional[OperatorDeepAnalysisResult]:
        """
        对单个算子执行深度分析

        Args:
            op_name: 算子名称
            metrics: 扩展 AIC 指标

        Returns:
            OperatorDeepAnalysisResult
        """
        # 1. 指令级分析
        instruction_analysis = self._analyze_instruction_mix(metrics)

        # 2. 内存层次分析
        memory_analysis = self._analyze_memory_hierarchy(metrics)

        # 3. 流水线停顿分析
        stall_analysis = self._analyze_stall_reasons(metrics)

        # 4. 综合瓶颈判断
        bottleneck = self._identify_deep_bottleneck(
            instruction_analysis, memory_analysis, stall_analysis
        )

        # 5. 生成优化建议
        recommendations = self._generate_targeted_recommendations(bottleneck)

        return OperatorDeepAnalysisResult(
            op_name=op_name,
            metrics=metrics,
            instruction_analysis=instruction_analysis,
            memory_analysis=memory_analysis,
            stall_analysis=stall_analysis,
            deep_bottleneck=bottleneck,
            recommendations=recommendations,
        )

    def _analyze_instruction_mix(
        self,
        metrics: ExtendedAICMetrics
    ) -> InstructionMixAnalysis:
        """
        分析指令混合比

        分析 Cube/Vector/Scalar 指令的使用情况
        """
        arithmetic = metrics.extended_arithmetic or metrics.arithmetic
        if not arithmetic:
            return InstructionMixAnalysis(pattern="unknown")

        # 基础利用率
        cube_util = arithmetic.cube_utilization
        vector_util = arithmetic.vector_utilization
        scalar_util = arithmetic.scalar_utilization

        # 指令级统计（如果可用）
        cube_instr = 0
        vector_instr = 0
        scalar_instr = 0
        issue_rate = 0.0

        if isinstance(arithmetic, ExtendedArithmeticUtilization):
            cube_instr = arithmetic.cube_instructions or 0
            vector_instr = arithmetic.vector_instructions or 0
            scalar_instr = arithmetic.scalar_instructions or 0
            issue_rate = arithmetic.instruction_issue_rate or 0.0

        # 计算指令比例
        total_instr = cube_instr + vector_instr + scalar_instr
        cube_ratio = cube_instr / total_instr if total_instr > 0 else 0
        vector_ratio = vector_instr / total_instr if total_instr > 0 else 0

        # 判断计算模式
        if cube_util > vector_util * 1.5:
            pattern = "cube_dominant"
        elif vector_util > cube_util * 1.5:
            pattern = "vector_dominant"
        elif scalar_util > max(cube_util, vector_util):
            pattern = "scalar_heavy"
        else:
            pattern = "balanced"

        # 判断是否存在问题
        issues = []
        if cube_util < 20:
            issues.append("cube_underutilized")
        if vector_util > cube_util * 2 and vector_util > 50:
            issues.append("vector_heavy")
        if issue_rate < 50:
            issues.append("low_issue_rate")

        return InstructionMixAnalysis(
            pattern=pattern,
            cube_utilization=cube_util,
            vector_utilization=vector_util,
            scalar_utilization=scalar_util,
            cube_instruction_ratio=cube_ratio,
            vector_instruction_ratio=vector_ratio,
            instruction_issue_rate=issue_rate,
            issues=issues,
        )

    def _analyze_memory_hierarchy(
        self,
        metrics: ExtendedAICMetrics
    ) -> MemoryHierarchyAnalysis:
        """
        分析内存层次性能

        分析 L2 Cache / UB / L0 的使用情况
        """
        memory = metrics.extended_memory or metrics.memory
        if not memory:
            return MemoryHierarchyAnalysis(bottleneck_type="unknown")

        # 基础指标
        l2_hit_rate = memory.l2_cache_hit_rate
        ub_usage = memory.ub_usage
        l0_usage = memory.l0_usage

        # 扩展指标
        l2_read_bw = 0.0
        l2_write_bw = 0.0
        ub_spill_count = 0
        hbm_access_count = 0

        if isinstance(memory, ExtendedMemoryMetrics):
            l2_read_bw = memory.l2_read_bandwidth or 0.0
            l2_write_bw = memory.l2_write_bandwidth or 0.0
            ub_spill_count = memory.ub_spill_count or 0
            hbm_access_count = memory.hbm_access_count or 0

        # 判断内存瓶颈类型
        bottleneck_type = "none"
        issues = []

        if l2_hit_rate < 50:
            bottleneck_type = "l2_miss"
            issues.append("l2_miss")
        elif ub_usage > 80:
            bottleneck_type = "ub_pressure"
            issues.append("ub_pressure")
        elif ub_spill_count > 100:
            bottleneck_type = "ub_pressure"
            issues.append("ub_spill")

        # 计算局部性评分
        locality_score = l2_hit_rate
        if ub_usage > 70:
            locality_score -= 20
        if l2_hit_rate < 60:
            locality_score -= 10

        locality_score = max(0, min(100, locality_score))

        return MemoryHierarchyAnalysis(
            bottleneck_type=bottleneck_type,
            l2_hit_rate=l2_hit_rate,
            l2_read_bandwidth=l2_read_bw,
            l2_write_bandwidth=l2_write_bw,
            ub_utilization=ub_usage,
            l0_utilization=l0_usage,
            ub_spill_count=ub_spill_count,
            locality_score=locality_score,
            issues=issues,
        )

    def _analyze_stall_reasons(
        self,
        metrics: ExtendedAICMetrics
    ) -> StallAnalysis:
        """
        分析流水线停顿原因

        细分停顿原因：MTE/依赖/内存/同步
        """
        pipeline = metrics.extended_pipeline or metrics.pipeline
        if not pipeline:
            return StallAnalysis(primary_cause="unknown")

        # 基础指标
        stall_rate = pipeline.stall_rate
        pipe_util = pipeline.pipe_utilization
        conflict_ratio = pipeline.resource_conflict_ratio

        # 扩展指标
        mte_stall = 0.0
        vec_stall = 0.0
        dependency_stall = 0.0
        memory_stall = 0.0
        sync_stall = 0.0

        if isinstance(pipeline, ExtendedPipelineMetrics):
            mte_stall = pipeline.mte_stall_rate or 0.0
            vec_stall = pipeline.vec_stall_rate or 0.0
            dependency_stall = pipeline.dependency_stall_rate or 0.0
            memory_stall = pipeline.memory_stall_rate or 0.0
            sync_stall = pipeline.sync_stall_rate or 0.0

        # 构建停顿分解
        stall_breakdown = {
            "mte": mte_stall,
            "vector": vec_stall,
            "dependency": dependency_stall,
            "memory": memory_stall,
            "sync": sync_stall,
            "other": max(0, stall_rate - mte_stall - dependency_stall - memory_stall - sync_stall),
        }

        # 找出主要停顿原因
        primary_cause = max(stall_breakdown.items(), key=lambda x: x[1])[0]
        primary_rate = stall_breakdown[primary_cause]

        # 判断停顿严重程度
        if stall_rate > 50:
            severity = "critical"
        elif stall_rate > 30:
            severity = "high"
        elif stall_rate > 15:
            severity = "medium"
        else:
            severity = "low"

        return StallAnalysis(
            primary_cause=primary_cause,
            primary_rate=primary_rate,
            stall_breakdown=stall_breakdown,
            total_stall_rate=stall_rate,
            pipe_utilization=pipe_util,
            conflict_ratio=conflict_ratio,
            severity=severity,
        )

    def _identify_deep_bottleneck(
        self,
        instruction: InstructionMixAnalysis,
        memory: MemoryHierarchyAnalysis,
        stall: StallAnalysis
    ) -> DeepBottleneckAnalysis:
        """
        综合判断深度瓶颈

        多维度综合分析，确定主要瓶颈类型和优化方向
        """
        analysis = DeepBottleneckAnalysis(bottleneck_category="balanced")

        # 计算各维度的瓶颈评分
        compute_score = 0
        memory_score = 0
        pipeline_score = 0

        # 计算瓶颈评分
        if instruction.pattern != "balanced" or instruction.issues:
            if "cube_underutilized" in instruction.issues:
                compute_score += 40
            if "vector_heavy" in instruction.issues:
                compute_score += 30
            if "low_issue_rate" in instruction.issues:
                compute_score += 25

        if memory.bottleneck_type != "none":
            if memory.bottleneck_type == "l2_miss":
                memory_score += 40
            elif memory.bottleneck_type == "ub_pressure":
                memory_score += 35
            if memory.locality_score < 50:
                memory_score += 20

        if stall.severity in ("critical", "high"):
            pipeline_score += 40 if stall.severity == "critical" else 25
            if stall.primary_rate > 30:
                pipeline_score += 15

        # 确定主要瓶颈类型
        max_score = max(compute_score, memory_score, pipeline_score)

        if max_score < 20:
            analysis.bottleneck_category = "balanced"
        elif compute_score == max_score:
            analysis.bottleneck_category = "compute_bound"
            analysis.compute_detail = instruction.issues[0] if instruction.issues else "cube_underutilized"
        elif memory_score == max_score:
            analysis.bottleneck_category = "memory_bound"
            analysis.memory_detail = memory.bottleneck_type
        elif pipeline_score == max_score:
            analysis.bottleneck_category = "pipeline_bound"
            analysis.pipeline_detail = f"{stall.primary_cause}_stall"

        # 如果多个维度都有问题，标记为混合瓶颈
        problem_count = sum(1 for s in [compute_score, memory_score, pipeline_score] if s > 20)
        if problem_count >= 2:
            analysis.bottleneck_category = "mixed"

        # 计算严重程度评分
        analysis.severity_score = compute_score + memory_score + pipeline_score

        # 设置主要优化建议
        self._set_primary_optimization(analysis, instruction, memory, stall)

        # 设置次要优化建议
        self._set_secondary_optimizations(analysis, instruction, memory, stall)

        # 预估加速比
        if analysis.severity_score > 60:
            analysis.estimated_speedup = 1.5
        elif analysis.severity_score > 40:
            analysis.estimated_speedup = 1.3
        elif analysis.severity_score > 20:
            analysis.estimated_speedup = 1.15
        else:
            analysis.estimated_speedup = 1.05

        return analysis

    def _set_primary_optimization(
        self,
        analysis: DeepBottleneckAnalysis,
        instruction: InstructionMixAnalysis,
        memory: MemoryHierarchyAnalysis,
        stall: StallAnalysis
    ):
        """设置主要优化建议"""
        strategies = OPTIMIZATION_STRATEGIES.get(analysis.bottleneck_category, {})

        if isinstance(strategies, dict):
            if analysis.compute_detail and analysis.compute_detail in strategies:
                opts = strategies[analysis.compute_detail]
                analysis.primary_optimization = opts[0] if opts else None
            elif analysis.memory_detail and analysis.memory_detail in strategies:
                opts = strategies[analysis.memory_detail]
                analysis.primary_optimization = opts[0] if opts else None
            elif analysis.pipeline_detail:
                key = analysis.pipeline_detail.replace("_stall", "")
                if key in strategies:
                    opts = strategies[key]
                    analysis.primary_optimization = opts[0] if opts else None
        elif isinstance(strategies, list) and strategies:
            analysis.primary_optimization = strategies[0]

        if not analysis.primary_optimization:
            analysis.primary_optimization = "当前性能较为均衡，可考虑进一步优化微架构细节"

    def _set_secondary_optimizations(
        self,
        analysis: DeepBottleneckAnalysis,
        instruction: InstructionMixAnalysis,
        memory: MemoryHierarchyAnalysis,
        stall: StallAnalysis
    ):
        """设置次要优化建议"""
        secondary = []

        # 基于各维度分析添加建议
        if "cube_underutilized" in instruction.issues:
            secondary.append("检查是否可以使用更高维度的矩阵乘法")
        if "vector_heavy" in instruction.issues:
            secondary.append("评估是否可以将 Vector 操作转为 Cube 操作")
        if memory.locality_score < 60:
            secondary.append("优化数据访问模式以提高局部性")
        if stall.primary_rate > 20:
            secondary.append(f"优化 {stall.primary_cause} 以减少停顿")

        analysis.secondary_optimizations = secondary[:3]

    def _generate_targeted_recommendations(
        self,
        bottleneck: DeepBottleneckAnalysis
    ) -> List[str]:
        """生成针对性优化建议"""
        recommendations = []

        if bottleneck.primary_optimization:
            recommendations.append(bottleneck.primary_optimization)

        recommendations.extend(bottleneck.secondary_optimizations)

        if bottleneck.estimated_speedup > 1.1:
            recommendations.append(
                f"预期优化后可获得约 {(bottleneck.estimated_speedup - 1) * 100:.0f}% 的性能提升"
            )

        return recommendations

    def _format_analysis_summary(
        self,
        analysis_results: List[OperatorDeepAnalysisResult]
    ) -> str:
        """格式化分析摘要"""
        lines = [
            "## 详细算子硬件分析结果 V2",
            f"分析 {len(analysis_results)} 个算子",
            "",
        ]

        # 按严重程度分组
        critical = [r for r in analysis_results
                    if r.deep_bottleneck and r.deep_bottleneck.severity_score >= 60]
        high = [r for r in analysis_results
                if r.deep_bottleneck and 40 <= r.deep_bottleneck.severity_score < 60]
        medium = [r for r in analysis_results
                  if r.deep_bottleneck and 20 <= r.deep_bottleneck.severity_score < 40]

        if critical:
            lines.append("### 严重瓶颈算子")
            for r in critical[:5]:
                b = r.deep_bottleneck
                lines.append(f"- **{r.op_name}** ({b.bottleneck_category})")
                if b.primary_optimization:
                    lines.append(f"  - 建议: {b.primary_optimization}")
            lines.append("")

        if high:
            lines.append("### 高优先级瓶颈")
            for r in high[:5]:
                b = r.deep_bottleneck
                lines.append(f"- **{r.op_name}** ({b.bottleneck_category})")
            lines.append("")

        if medium:
            lines.append(f"### 中等优先级瓶颈 ({len(medium)} 个)")
            for r in medium[:5]:
                b = r.deep_bottleneck
                lines.append(f"- {r.op_name} ({b.bottleneck_category})")

        return "\n".join(lines)

    def _extract_recommendations(self, response: str) -> List[str]:
        """从 LLM 响应中提取优化建议"""
        recommendations = []

        lines = response.split("\n")
        in_recommendation = False

        for line in lines:
            line_lower = line.lower()
            if "建议" in line or "优化" in line or "suggestion" in line_lower:
                in_recommendation = True
            if in_recommendation and line.strip().startswith(("-", "*", "•", "1", "2", "3", "4", "5")):
                clean_line = line.strip().lstrip("-*•0123456789. )")
                if clean_line and len(clean_line) > 5:
                    recommendations.append(clean_line)

        return recommendations[:10]


# ============================================================================
# 辅助数据结构
# ============================================================================

@dataclass
class InstructionMixAnalysis:
    """指令混合分析结果"""
    pattern: str                                   # 计算模式: cube_dominant/vector_dominant/balanced/scalar_heavy
    cube_utilization: float = 0.0
    vector_utilization: float = 0.0
    scalar_utilization: float = 0.0
    cube_instruction_ratio: float = 0.0            # Cube 指令占比
    vector_instruction_ratio: float = 0.0          # Vector 指令占比
    instruction_issue_rate: float = 0.0            # 指令发射率
    issues: List[str] = field(default_factory=list)  # 检测到的问题


@dataclass
class MemoryHierarchyAnalysis:
    """内存层次分析结果"""
    bottleneck_type: str                           # 瓶颈类型: l2_miss/ub_pressure/hbm_saturated/none
    l2_hit_rate: float = 0.0
    l2_read_bandwidth: float = 0.0
    l2_write_bandwidth: float = 0.0
    ub_utilization: float = 0.0
    l0_utilization: float = 0.0
    ub_spill_count: int = 0
    locality_score: float = 0.0                    # 数据局部性评分 (0-100)
    issues: List[str] = field(default_factory=list)


@dataclass
class StallAnalysis:
    """流水线停顿分析结果"""
    primary_cause: str                              # 主要停顿原因: mte/dependency/memory/sync/other
    primary_rate: float = 0.0                       # 主要停顿率
    stall_breakdown: Dict[str, float] = field(default_factory=dict)  # 停顿分解
    total_stall_rate: float = 0.0
    pipe_utilization: float = 0.0
    conflict_ratio: float = 0.0
    severity: str = "low"                           # 严重程度: critical/high/medium/low


@dataclass
class OperatorDeepAnalysisResult:
    """算子深度分析结果"""
    op_name: str
    metrics: ExtendedAICMetrics

    # 各维度分析
    instruction_analysis: Optional[InstructionMixAnalysis] = None
    memory_analysis: Optional[MemoryHierarchyAnalysis] = None
    stall_analysis: Optional[StallAnalysis] = None

    # 综合瓶颈分析
    deep_bottleneck: Optional[DeepBottleneckAnalysis] = None

    # 优化建议
    recommendations: List[str] = field(default_factory=list)
