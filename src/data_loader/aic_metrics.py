"""
AIC Metrics 数据结构定义

定义用于存储昇腾 NPU AI Core 详细硬件指标的数据结构。
这些指标通过 msprof op --aic-metrics 采集。
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List


@dataclass
class ArithmeticUtilization:
    """
    算术单元利用率指标

    昇腾 AI Core 包含三种计算单元：
    - Cube: 矩阵计算单元，用于矩阵乘法等计算密集型操作
    - Vector: 向量计算单元，用于向量运算
    - Scalar: 标量计算单元，用于标量运算和条件判断
    """
    cube_utilization: float = 0.0  # Cube单元利用率 (0-100)
    vector_utilization: float = 0.0  # Vector单元利用率 (0-100)
    scalar_utilization: float = 0.0  # Scalar单元利用率 (0-100)
    total_cycles: int = 0  # 总周期数

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "cube_utilization": self.cube_utilization,
            "vector_utilization": self.vector_utilization,
            "scalar_utilization": self.scalar_utilization,
            "total_cycles": self.total_cycles,
        }


@dataclass
class MemoryMetrics:
    """
    内存访问指标

    昇腾 AI Core 的内存层次结构：
    - HBM: 高带宽内存
    - L2 Cache: 二级缓存
    - L0 Buffer: 一级缓冲区
    - UB (Unified Buffer): 统一缓冲区
    """
    l2_cache_hit_rate: float = 0.0  # L2缓存命中率 (0-100)
    l2_read_bandwidth: float = 0.0  # L2读带宽 (GB/s)
    l2_write_bandwidth: float = 0.0  # L2写带宽 (GB/s)
    ub_usage: float = 0.0  # Unified Buffer使用率 (0-100)
    l0_usage: float = 0.0  # L0 Buffer使用率 (0-100)
    memory_efficiency: float = 0.0  # 内存效率 (0-100)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "l2_cache_hit_rate": self.l2_cache_hit_rate,
            "l2_read_bandwidth": self.l2_read_bandwidth,
            "l2_write_bandwidth": self.l2_write_bandwidth,
            "ub_usage": self.ub_usage,
            "l0_usage": self.l0_usage,
            "memory_efficiency": self.memory_efficiency,
        }


@dataclass
class PipelineMetrics:
    """
    流水线利用率指标

    描述 AI Core 流水线的执行效率：
    - pipe_utilization: 流水线利用率
    - stall_rate: 停顿率（由于数据依赖或资源冲突导致的等待）
    - resource_conflict_ratio: 资源冲突率
    """
    pipe_utilization: float = 0.0  # 流水线利用率 (0-100)
    stall_rate: float = 0.0  # 停顿率 (0-100)
    resource_conflict_ratio: float = 0.0  # 资源冲突率 (0-100)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "pipe_utilization": self.pipe_utilization,
            "stall_rate": self.stall_rate,
            "resource_conflict_ratio": self.resource_conflict_ratio,
        }


@dataclass
class AICMetrics:
    """
    AIC 硬件指标完整数据结构

    包含单个算子的所有详细硬件性能指标。
    """
    op_name: str  # 算子名称
    op_type: str  # 算子类型
    duration_us: float = 0.0  # 执行时间(微秒)

    # 算术单元利用率
    arithmetic: Optional[ArithmeticUtilization] = None

    # 内存访问指标
    memory: Optional[MemoryMetrics] = None

    # 流水线指标
    pipeline: Optional[PipelineMetrics] = None

    # 原始数据(用于调试)
    raw_data: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        result = {
            "op_name": self.op_name,
            "op_type": self.op_type,
            "duration_us": self.duration_us,
        }

        if self.arithmetic:
            result.update({f"arithmetic_{k}": v for k, v in self.arithmetic.to_dict().items()})

        if self.memory:
            result.update({f"memory_{k}": v for k, v in self.memory.to_dict().items()})

        if self.pipeline:
            result.update({f"pipeline_{k}": v for k, v in self.pipeline.to_dict().items()})

        return result

    def get_summary(self) -> str:
        """获取指标摘要字符串"""
        parts = [f"{self.op_name} ({self.op_type})"]

        if self.arithmetic:
            parts.append(f"Cube: {self.arithmetic.cube_utilization:.1f}%")

        if self.memory:
            parts.append(f"L2: {self.memory.l2_cache_hit_rate:.1f}%")

        if self.pipeline:
            parts.append(f"Pipe: {self.pipeline.pipe_utilization:.1f}%")

        return " | ".join(parts)


@dataclass
class AICAnalysisResult:
    """
    AIC 指标分析结果

    包含对单个算子的硬件性能分析结果。
    """
    operator_name: str
    bottleneck_type: str  # "compute", "memory", "pipeline", "balanced"
    severity: str  # "critical", "high", "medium", "low"

    # 详细指标
    metrics: AICMetrics

    # 诊断结果
    diagnosis: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)

    # AIKG 集成
    aikg_prompts: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "operator_name": self.operator_name,
            "bottleneck_type": self.bottleneck_type,
            "severity": self.severity,
            "metrics": self.metrics.to_dict(),
            "diagnosis": self.diagnosis,
            "recommendations": self.recommendations,
            "aikg_prompts": self.aikg_prompts,
        }


@dataclass
class DetailedOperatorAnalysisData:
    """
    详细算子分析数据聚合

    包含多个算子的分析结果汇总。
    """
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
            1 for r in self.analysis_results if r.severity == "critical"
        )
        self.high_count = sum(
            1 for r in self.analysis_results if r.severity == "high"
        )
        self.medium_count = sum(
            1 for r in self.analysis_results if r.severity == "medium"
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
            r for r in self.analysis_results if r.severity == "critical"
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


# 瓶颈类型常量
BOTTLENECK_COMPUTE = "compute"
BOTTLENECK_MEMORY = "memory"
BOTTLENECK_PIPELINE = "pipeline"
BOTTLENECK_BALANCED = "balanced"

# 严重程度常量
SEVERITY_CRITICAL = "critical"
SEVERITY_HIGH = "high"
SEVERITY_MEDIUM = "medium"
SEVERITY_LOW = "low"

# 瓶颈判断阈值
CRITICAL_THRESHOLD = 20.0
HIGH_THRESHOLD = 40.0
MEDIUM_THRESHOLD = 60.0


# ============================================================================
# 扩展指标数据结构 (用于深度分析)
# ============================================================================

@dataclass
class ExtendedArithmeticUtilization(ArithmeticUtilization):
    """
    扩展的算术单元利用率指标

    在基础指标上增加指令级统计信息
    """
    # 指令级统计
    cube_instructions: int = 0              # Cube 指令数
    vector_instructions: int = 0            # Vector 指令数
    scalar_instructions: int = 0            # Scalar 指令数
    instruction_issue_rate: float = 0.0     # 指令发射率 (0-100)

    # 活跃周期统计
    cube_active_cycles: int = 0             # Cube 活跃周期
    vector_active_cycles: int = 0           # Vector 活跃周期
    scalar_active_cycles: int = 0           # Scalar 活跃周期

    # 计算效率
    cube_efficiency: float = 0.0            # Cube 效率 (实际/理论吞吐量)
    vector_efficiency: float = 0.0          # Vector 效率

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        base = super().to_dict()
        base.update({
            "cube_instructions": self.cube_instructions,
            "vector_instructions": self.vector_instructions,
            "scalar_instructions": self.scalar_instructions,
            "instruction_issue_rate": self.instruction_issue_rate,
            "cube_active_cycles": self.cube_active_cycles,
            "vector_active_cycles": self.vector_active_cycles,
            "scalar_active_cycles": self.scalar_active_cycles,
            "cube_efficiency": self.cube_efficiency,
            "vector_efficiency": self.vector_efficiency,
        })
        return base

    def get_instruction_mix(self) -> Dict[str, float]:
        """获取指令混合比例"""
        total = self.cube_instructions + self.vector_instructions + self.scalar_instructions
        if total == 0:
            return {"cube": 0, "vector": 0, "scalar": 0}
        return {
            "cube": self.cube_instructions / total,
            "vector": self.vector_instructions / total,
            "scalar": self.scalar_instructions / total,
        }

    def classify_compute_pattern(self) -> str:
        """
        分类计算模式

        Returns:
            "cube_dominant", "vector_dominant", "scalar_heavy", "balanced"
        """
        mix = self.get_instruction_mix()

        if mix["cube"] > 0.6:
            return "cube_dominant"
        elif mix["vector"] > 0.6:
            return "vector_dominant"
        elif mix["scalar"] > 0.3:
            return "scalar_heavy"
        else:
            return "balanced"


@dataclass
class ExtendedMemoryMetrics(MemoryMetrics):
    """
    扩展的内存访问指标

    在基础指标上增加内存层次详细信息
    """
    # L2 Cache 详细指标
    l2_read_bytes: int = 0                  # L2 读取字节数
    l2_write_bytes: int = 0                 # L2 写入字节数
    l2_read_requests: int = 0               # L2 读请求数
    l2_write_requests: int = 0              # L2 写请求数
    l2_miss_count: int = 0                  # L2 未命中次数

    # UB (Unified Buffer) 详细指标
    ub_peak_usage: float = 0.0              # UB 峰值使用量 (KB)
    ub_spill_count: int = 0                 # UB 溢出次数
    ub_spill_bytes: int = 0                 # UB 溢出字节数
    ub_conflict_rate: float = 0.0           # UB 冲突率

    # L0 Buffer 详细指标
    l0a_utilization: float = 0.0            # L0A 使用率
    l0b_utilization: float = 0.0            # L0B 使用率
    l0c_utilization: float = 0.0            # L0C 使用率

    # HBM 访问指标
    hbm_read_bytes: int = 0                 # HBM 读取字节数
    hbm_write_bytes: int = 0                # HBM 写入字节数
    hbm_access_count: int = 0               # HBM 访问次数
    hbm_bandwidth_utilization: float = 0.0  # HBM 带宽利用率

    # 数据局部性
    locality_score: float = 0.0             # 数据局部性评分 (0-100)
    reuse_distance: float = 0.0             # 平均复用距离

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        base = super().to_dict()
        base.update({
            "l2_read_bytes": self.l2_read_bytes,
            "l2_write_bytes": self.l2_write_bytes,
            "l2_read_requests": self.l2_read_requests,
            "l2_write_requests": self.l2_write_requests,
            "l2_miss_count": self.l2_miss_count,
            "ub_peak_usage": self.ub_peak_usage,
            "ub_spill_count": self.ub_spill_count,
            "ub_conflict_rate": self.ub_conflict_rate,
            "l0a_utilization": self.l0a_utilization,
            "l0b_utilization": self.l0b_utilization,
            "l0c_utilization": self.l0c_utilization,
            "hbm_read_bytes": self.hbm_read_bytes,
            "hbm_write_bytes": self.hbm_write_bytes,
            "hbm_bandwidth_utilization": self.hbm_bandwidth_utilization,
            "locality_score": self.locality_score,
        })
        return base

    def classify_memory_bottleneck(self) -> str:
        """
        分类内存瓶颈类型

        Returns:
            "l2_miss", "ub_pressure", "hbm_saturated", "poor_locality", "none"
        """
        if self.l2_cache_hit_rate < 50:
            return "l2_miss"
        elif self.ub_usage > 80 or self.ub_spill_count > 100:
            return "ub_pressure"
        elif self.hbm_bandwidth_utilization > 80:
            return "hbm_saturated"
        elif self.locality_score < 50:
            return "poor_locality"
        else:
            return "none"

    def calc_locality_score(self) -> float:
        """
        计算数据局部性评分

        基于 L2 命中率、UB 使用效率和 HBM 访问模式
        """
        score = 0.0

        # L2 命中率贡献 (40%)
        score += self.l2_cache_hit_rate * 0.4

        # UB 效率贡献 (30%) - UB 使用率适中最好
        ub_efficiency = 100 - abs(self.ub_usage - 50)  # 50% 使用率最佳
        score += ub_efficiency * 0.3

        # HBM 带宽利用率贡献 (30%) - 越低越好（说明数据在缓存中）
        score += (100 - self.hbm_bandwidth_utilization) * 0.3

        self.locality_score = max(0.0, min(100.0, score))
        return self.locality_score


@dataclass
class ExtendedPipelineMetrics(PipelineMetrics):
    """
    扩展的流水线利用率指标

    在基础指标上增加停顿原因细分
    """
    # 停顿原因细分
    mte_stall_rate: float = 0.0             # MTE (Memory Transfer Engine) 停顿率
    vec_stall_rate: float = 0.0             # Vector 单元停顿率
    scalar_stall_rate: float = 0.0          # Scalar 单元停顿率
    dependency_stall_rate: float = 0.0      # 依赖停顿率
    memory_stall_rate: float = 0.0          # 内存停顿率
    sync_stall_rate: float = 0.0            # 同步停顿率

    # 资源冲突细分
    mte_conflict_rate: float = 0.0          # MTE 资源冲突率
    vec_conflict_rate: float = 0.0          # Vector 资源冲突率
    ub_conflict_rate: float = 0.0           # UB 冲突率

    # 流水线效率
    issue_rate: float = 0.0                 # 指令发射率
    commit_rate: float = 0.0                # 指令提交率
    branch_misprediction_rate: float = 0.0  # 分支预测错误率

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        base = super().to_dict()
        base.update({
            "mte_stall_rate": self.mte_stall_rate,
            "vec_stall_rate": self.vec_stall_rate,
            "scalar_stall_rate": self.scalar_stall_rate,
            "dependency_stall_rate": self.dependency_stall_rate,
            "memory_stall_rate": self.memory_stall_rate,
            "sync_stall_rate": self.sync_stall_rate,
            "mte_conflict_rate": self.mte_conflict_rate,
            "vec_conflict_rate": self.vec_conflict_rate,
            "ub_conflict_rate": self.ub_conflict_rate,
            "issue_rate": self.issue_rate,
            "commit_rate": self.commit_rate,
            "branch_misprediction_rate": self.branch_misprediction_rate,
        })
        return base

    def get_stall_breakdown(self) -> Dict[str, float]:
        """获取停顿原因分布"""
        return {
            "mte": self.mte_stall_rate,
            "vector": self.vec_stall_rate,
            "scalar": self.scalar_stall_rate,
            "dependency": self.dependency_stall_rate,
            "memory": self.memory_stall_rate,
            "sync": self.sync_stall_rate,
        }

    def get_primary_stall_cause(self) -> Tuple[str, float]:
        """
        获取主要停顿原因

        Returns:
            (停顿类型, 停顿率)
        """
        breakdown = self.get_stall_breakdown()
        if not breakdown:
            return ("unknown", 0.0)
        return max(breakdown.items(), key=lambda x: x[1])

    def classify_stall_severity(self) -> str:
        """
        分类停顿严重程度

        Returns:
            "critical", "high", "medium", "low"
        """
        if self.stall_rate > 50:
            return "critical"
        elif self.stall_rate > 30:
            return "high"
        elif self.stall_rate > 15:
            return "medium"
        else:
            return "low"


@dataclass
class DeepBottleneckAnalysis:
    """
    深度瓶颈分析结果

    包含更细粒度的瓶颈诊断信息
    """
    # 瓶颈大类
    bottleneck_category: str  # compute_bound, memory_bound, pipeline_bound, balanced, mixed

    # 计算瓶颈细分
    compute_detail: Optional[str] = None
    # 可能值: "cube_underutilized", "vector_heavy", "low_issue_rate", "instruction_imbalance"

    # 内存瓶颈细分
    memory_detail: Optional[str] = None
    # 可能值: "l2_miss", "ub_pressure", "hbm_saturated", "poor_locality"

    # 流水线瓶颈细分
    pipeline_detail: Optional[str] = None
    # 可能值: "mte_stall", "dependency_stall", "resource_conflict", "memory_stall"

    # 综合评分
    severity_score: float = 0.0              # 严重程度评分 (0-100)
    confidence: float = 0.0                  # 置信度 (0-1)

    # 优化方向
    primary_optimization: str = ""           # 主要优化方向
    secondary_optimizations: List[str] = field(default_factory=list)

    # 预期效果
    estimated_speedup: float = 0.0           # 预期加速比

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "bottleneck_category": self.bottleneck_category,
            "compute_detail": self.compute_detail,
            "memory_detail": self.memory_detail,
            "pipeline_detail": self.pipeline_detail,
            "severity_score": round(self.severity_score, 2),
            "confidence": round(self.confidence, 2),
            "primary_optimization": self.primary_optimization,
            "secondary_optimizations": self.secondary_optimizations,
            "estimated_speedup": round(self.estimated_speedup, 2),
        }

    def to_summary(self) -> str:
        """生成简要描述"""
        parts = [f"瓶颈类型: {self.bottleneck_category}"]

        if self.compute_detail:
            parts.append(f"计算: {self.compute_detail}")
        if self.memory_detail:
            parts.append(f"内存: {self.memory_detail}")
        if self.pipeline_detail:
            parts.append(f"流水线: {self.pipeline_detail}")

        parts.append(f"严重度: {self.severity_score:.0f}/100")
        parts.append(f"建议: {self.primary_optimization}")

        return " | ".join(parts)


@dataclass
class ExtendedAICMetrics(AICMetrics):
    """
    扩展的 AIC 硬件指标

    包含更详细的硬件性能指标用于深度分析
    """
    # 扩展指标
    extended_arithmetic: Optional[ExtendedArithmeticUtilization] = None
    extended_memory: Optional[ExtendedMemoryMetrics] = None
    extended_pipeline: Optional[ExtendedPipelineMetrics] = None

    # 深度分析结果
    deep_bottleneck: Optional[DeepBottleneckAnalysis] = None

    # 性能计数器
    ai_core_frequency_mhz: float = 0.0      # AI Core 频率
    ai_vector_frequency_mhz: float = 0.0    # AI Vector 频率

    # 功耗和温度 (如果可用)
    power_watts: float = 0.0                # 功耗
    temperature_celsius: float = 0.0        # 温度

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        base = super().to_dict()

        if self.extended_arithmetic:
            base["extended_arithmetic"] = self.extended_arithmetic.to_dict()
        if self.extended_memory:
            base["extended_memory"] = self.extended_memory.to_dict()
        if self.extended_pipeline:
            base["extended_pipeline"] = self.extended_pipeline.to_dict()
        if self.deep_bottleneck:
            base["deep_bottleneck"] = self.deep_bottleneck.to_dict()

        base["ai_core_frequency_mhz"] = self.ai_core_frequency_mhz
        base["power_watts"] = self.power_watts
        base["temperature_celsius"] = self.temperature_celsius

        return base

    def perform_deep_analysis(self) -> DeepBottleneckAnalysis:
        """
        执行深度瓶颈分析

        Returns:
            DeepBottleneckAnalysis
        """
        analysis = DeepBottleneckAnalysis(bottleneck_category="balanced")

        # 分析计算瓶颈
        if self.extended_arithmetic:
            arith = self.extended_arithmetic

            if arith.cube_utilization < 20:
                analysis.bottleneck_category = "compute_bound"
                analysis.compute_detail = "cube_underutilized"
                analysis.severity_score += 40
            elif arith.cube_utilization < 40:
                analysis.compute_detail = "cube_underutilized"
                analysis.severity_score += 20

            if arith.instruction_issue_rate < 50:
                analysis.compute_detail = "low_issue_rate"
                analysis.severity_score += 15

            pattern = arith.classify_compute_pattern()
            if pattern == "vector_dominant":
                analysis.secondary_optimizations.append(
                    "评估是否可以将 Vector 操作转为 Cube 操作"
                )

        # 分析内存瓶颈
        if self.extended_memory:
            mem = self.extended_memory

            mem_bottleneck = mem.classify_memory_bottleneck()
            if mem_bottleneck != "none":
                if analysis.bottleneck_category == "compute_bound":
                    analysis.bottleneck_category = "mixed"
                else:
                    analysis.bottleneck_category = "memory_bound"
                analysis.memory_detail = mem_bottleneck
                analysis.severity_score += 30

            if mem.ub_spill_count > 100:
                analysis.secondary_optimizations.append(
                    "减小 tile size 以降低 UB 压力"
                )

        # 分析流水线瓶颈
        if self.extended_pipeline:
            pipe = self.extended_pipeline

            if pipe.stall_rate > 30:
                if analysis.bottleneck_category in ("compute_bound", "memory_bound"):
                    analysis.bottleneck_category = "mixed"
                else:
                    analysis.bottleneck_category = "pipeline_bound"

                stall_type, stall_rate = pipe.get_primary_stall_cause()
                analysis.pipeline_detail = f"{stall_type}_stall"
                analysis.severity_score += stall_rate

        # 设置优化建议
        analysis = self._generate_optimization_suggestions(analysis)

        # 计算置信度
        analysis.confidence = min(1.0, len([
            x for x in [self.extended_arithmetic, self.extended_memory, self.extended_pipeline]
            if x is not None
        ]) / 3.0)

        self.deep_bottleneck = analysis
        return analysis

    def _generate_optimization_suggestions(
        self,
        analysis: DeepBottleneckAnalysis
    ) -> DeepBottleneckAnalysis:
        """生成优化建议"""
        suggestions = {
            "compute_bound": {
                "cube_underutilized": "优化数据布局以提高计算密度，考虑使用更高维度的矩阵乘法",
                "vector_heavy": "评估是否可以将部分 Vector 操作转为 Cube 操作",
                "low_issue_rate": "检查是否存在频繁的 kernel launch 开销，考虑算子融合",
            },
            "memory_bound": {
                "l2_miss": "优化数据访问模式以提高缓存命中率，考虑使用数据预取",
                "ub_pressure": "减小 tile size 以降低 UB 压力，考虑使用 double buffering",
                "hbm_saturated": "使用更高效的数据类型，优化数据布局以减少访存次数",
                "poor_locality": "优化数据访问模式，考虑调整内存布局",
            },
            "pipeline_bound": {
                "mte_stall": "优化 MTE 单元的数据调度，使用数据预取隐藏延迟",
                "dependency_stall": "优化指令调度以减少依赖，考虑循环展开",
                "resource_conflict": "调整计算资源分配，优化流水线调度策略",
            },
        }

        if analysis.bottleneck_category in suggestions:
            category_suggestions = suggestions[analysis.bottleneck_category]

            # 设置主要优化建议
            if analysis.compute_detail and analysis.compute_detail in category_suggestions:
                analysis.primary_optimization = category_suggestions[analysis.compute_detail]
            elif analysis.memory_detail and analysis.memory_detail in category_suggestions:
                analysis.primary_optimization = category_suggestions[analysis.memory_detail]
            elif analysis.pipeline_detail and analysis.pipeline_detail in category_suggestions:
                analysis.primary_optimization = category_suggestions[analysis.pipeline_detail]
            elif "default" in category_suggestions:
                analysis.primary_optimization = category_suggestions["default"]

        if not analysis.primary_optimization:
            analysis.primary_optimization = "当前性能较为均衡，可考虑进一步优化微架构细节"

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


# ============================================================================
# 瓶颈类型枚举
# ============================================================================

from enum import Enum


class BottleneckCategory(Enum):
    """瓶颈类型分类"""
    COMPUTE_BOUND = "compute_bound"         # 计算受限
    MEMORY_BOUND = "memory_bound"           # 内存受限
    PIPELINE_BOUND = "pipeline_bound"       # 流水线受限
    BALANCED = "balanced"                   # 均衡
    MIXED = "mixed"                         # 混合瓶颈


class ComputeBottleneckType(Enum):
    """计算瓶颈细分类型"""
    CUBE_UNDERUTILIZED = "cube_underutilized"   # Cube 利用率低
    VECTOR_HEAVY = "vector_heavy"               # Vector 负载过重
    LOW_ISSUE_RATE = "low_issue_rate"           # 指令发射率低
    INSTRUCTION_IMBALANCE = "instruction_imbalance"  # 指令不平衡


class MemoryBottleneckType(Enum):
    """内存瓶颈细分类型"""
    L2_MISS = "l2_miss"                     # L2 命中率低
    UB_PRESSURE = "ub_pressure"             # UB 压力大
    HBM_SATURATED = "hbm_saturated"         # HBM 带宽饱和
    POOR_LOCALITY = "poor_locality"         # 数据局部性差


class PipelineBottleneckType(Enum):
    """流水线瓶颈细分类型"""
    MTE_STALL = "mte_stall"                 # MTE 停顿
    DEPENDENCY_STALL = "dependency_stall"   # 依赖停顿
    RESOURCE_CONFLICT = "resource_conflict" # 资源冲突
    MEMORY_STALL = "memory_stall"           # 内存停顿
    SYNC_STALL = "sync_stall"               # 同步停顿
