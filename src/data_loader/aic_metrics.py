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
