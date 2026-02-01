"""
Roofline Model 性能天花板分析

基于 Roofline 模型分析计算与内存带宽的理论上限
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum
import math
import logging

logger = logging.getLogger(__name__)


class BoundType(Enum):
    """性能受限类型"""
    COMPUTE_BOUND = "compute_bound"      # 计算受限
    MEMORY_BOUND = "memory_bound"        # 内存带宽受限
    COMM_BOUND = "communication_bound"   # 通信受限
    BALANCED = "balanced"                # 平衡点附近


class PrecisionType(Enum):
    """计算精度类型"""
    FP32 = "fp32"
    FP16 = "fp16"
    BF16 = "bf16"
    INT8 = "int8"


@dataclass
class HardwareSpec:
    """硬件规格"""
    name: str
    # 算力 (TFLOPS)
    fp32_tflops: float
    fp16_tflops: float
    bf16_tflops: float
    # 带宽 (GB/s)
    hbm_bandwidth_gbps: float
    # 可选参数
    int8_tops: float = 0
    hccs_bandwidth_gbps: float = 0
    rdma_bandwidth_gbps: float = 0
    # 其他
    aicore_count: int = 0
    l2_cache_mb: float = 0


@dataclass
class OperatorProfile:
    """算子性能数据"""
    name: str
    flops: float                    # 浮点运算数
    memory_bytes: float             # 内存访问量（读+写）
    duration_us: float              # 执行时间（微秒）
    precision: PrecisionType = PrecisionType.FP16
    
    @property
    def arithmetic_intensity(self) -> float:
        """计算强度 (FLOP/Byte)"""
        if self.memory_bytes == 0:
            return float('inf')
        return self.flops / self.memory_bytes
    
    @property
    def achieved_tflops(self) -> float:
        """实际算力 (TFLOPS)"""
        if self.duration_us == 0:
            return 0
        return self.flops / (self.duration_us * 1e6)  # us -> s, then to TFLOPS
    
    @property
    def achieved_bandwidth_gbps(self) -> float:
        """实际带宽 (GB/s)"""
        if self.duration_us == 0:
            return 0
        return self.memory_bytes / (self.duration_us * 1e-6) / 1e9


@dataclass
class RooflinePoint:
    """Roofline 图上的点"""
    name: str
    arithmetic_intensity: float     # 计算强度 (FLOP/Byte)
    achieved_performance: float     # 实际性能 (TFLOPS)
    theoretical_peak: float         # 理论峰值 (TFLOPS)
    memory_roof: float              # 内存天花板 (TFLOPS at this AI)
    bound_type: BoundType
    efficiency: float               # 效率 (%)
    gap_to_roof: float              # 距天花板差距 (TFLOPS)


@dataclass
class RooflineAnalysis:
    """Roofline 分析结果"""
    hardware: HardwareSpec
    precision: PrecisionType
    # 天花板参数
    compute_roof_tflops: float      # 计算天花板
    memory_roof_slope: float        # 内存天花板斜率 (TFLOPS per FLOP/Byte)
    ridge_point: float              # 脊点（平衡点）的计算强度
    # 分析点
    points: List[RooflinePoint] = field(default_factory=list)
    # 汇总
    overall_bound: BoundType = BoundType.BALANCED
    avg_efficiency: float = 0
    bottleneck_operators: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "hardware": self.hardware.name,
            "precision": self.precision.value,
            "compute_roof_tflops": self.compute_roof_tflops,
            "memory_roof_slope": self.memory_roof_slope,
            "ridge_point": self.ridge_point,
            "overall_bound": self.overall_bound.value,
            "avg_efficiency": self.avg_efficiency,
            "bottleneck_operators": self.bottleneck_operators,
            "points": [
                {
                    "name": p.name,
                    "arithmetic_intensity": p.arithmetic_intensity,
                    "achieved_performance": p.achieved_performance,
                    "efficiency": p.efficiency,
                    "bound_type": p.bound_type.value,
                }
                for p in self.points
            ],
            "suggestions": self.suggestions,
        }


class RooflineModeler:
    """
    Roofline 性能模型分析器
    
    基于经典 Roofline 模型分析性能瓶颈：
    - 计算天花板：硬件理论峰值算力
    - 内存天花板：内存带宽 × 计算强度
    - 脊点：两个天花板的交点
    """
    
    # 预定义硬件规格
    HARDWARE_SPECS = {
        "atlas_a2_280t": HardwareSpec(
            name="Atlas A2 (280T)",
            fp32_tflops=140,
            fp16_tflops=280,
            bf16_tflops=280,
            int8_tops=560,
            hbm_bandwidth_gbps=1500,
            hccs_bandwidth_gbps=56,
            rdma_bandwidth_gbps=25,
            aicore_count=24,
            l2_cache_mb=192,
        ),
        "atlas_a2_313t": HardwareSpec(
            name="Atlas A2 (313T)",
            fp32_tflops=156,
            fp16_tflops=313,
            bf16_tflops=313,
            int8_tops=626,
            hbm_bandwidth_gbps=1800,
            hccs_bandwidth_gbps=56,
            rdma_bandwidth_gbps=25,
            aicore_count=24,
            l2_cache_mb=192,
        ),
        "atlas_a2_376t": HardwareSpec(
            name="Atlas A2 (376T)",
            fp32_tflops=188,
            fp16_tflops=376,
            bf16_tflops=376,
            int8_tops=752,
            hbm_bandwidth_gbps=2000,
            hccs_bandwidth_gbps=56,
            rdma_bandwidth_gbps=25,
            aicore_count=24,
            l2_cache_mb=192,
        ),
    }
    
    def __init__(
        self,
        hardware: HardwareSpec = None,
        hardware_name: str = "atlas_a2_280t",
    ):
        """
        初始化 Roofline 分析器
        
        Args:
            hardware: 硬件规格，如果为 None 则使用 hardware_name 查找
            hardware_name: 预定义硬件名称
        """
        if hardware:
            self.hardware = hardware
        else:
            self.hardware = self.HARDWARE_SPECS.get(
                hardware_name, 
                self.HARDWARE_SPECS["atlas_a2_280t"]
            )
    
    def get_compute_roof(self, precision: PrecisionType) -> float:
        """获取计算天花板 (TFLOPS)"""
        if precision == PrecisionType.FP32:
            return self.hardware.fp32_tflops
        elif precision in (PrecisionType.FP16, PrecisionType.BF16):
            return self.hardware.fp16_tflops
        elif precision == PrecisionType.INT8:
            return self.hardware.int8_tops
        return self.hardware.fp16_tflops
    
    def get_memory_roof(
        self, 
        arithmetic_intensity: float,
        bandwidth_gbps: float = None,
    ) -> float:
        """
        获取内存天花板 (TFLOPS)
        
        Args:
            arithmetic_intensity: 计算强度 (FLOP/Byte)
            bandwidth_gbps: 带宽，默认使用 HBM 带宽
            
        Returns:
            在该计算强度下的内存天花板
        """
        if bandwidth_gbps is None:
            bandwidth_gbps = self.hardware.hbm_bandwidth_gbps
        
        # 内存天花板 = 带宽 × 计算强度
        # 单位：GB/s × FLOP/Byte = GFLOP/s = 0.001 TFLOPS
        return bandwidth_gbps * arithmetic_intensity / 1000
    
    def get_ridge_point(
        self, 
        precision: PrecisionType,
        bandwidth_gbps: float = None,
    ) -> float:
        """
        获取脊点（平衡点）的计算强度
        
        脊点是计算天花板和内存天花板的交点
        
        Returns:
            脊点的计算强度 (FLOP/Byte)
        """
        compute_roof = self.get_compute_roof(precision)
        if bandwidth_gbps is None:
            bandwidth_gbps = self.hardware.hbm_bandwidth_gbps
        
        # 计算天花板 = 内存天花板
        # Peak TFLOPS = Bandwidth GB/s × AI / 1000
        # AI = Peak TFLOPS × 1000 / Bandwidth
        return compute_roof * 1000 / bandwidth_gbps
    
    def analyze_operator(
        self,
        op: OperatorProfile,
        precision: PrecisionType = None,
    ) -> RooflinePoint:
        """
        分析单个算子在 Roofline 图上的位置
        
        Args:
            op: 算子性能数据
            precision: 计算精度，默认使用算子的精度
            
        Returns:
            RooflinePoint: Roofline 分析点
        """
        if precision is None:
            precision = op.precision
        
        compute_roof = self.get_compute_roof(precision)
        ai = op.arithmetic_intensity
        memory_roof = self.get_memory_roof(ai)
        ridge = self.get_ridge_point(precision)
        
        # 确定天花板（取较低者）
        theoretical_peak = min(compute_roof, memory_roof)
        
        # 确定受限类型
        if ai < ridge * 0.8:
            bound_type = BoundType.MEMORY_BOUND
        elif ai > ridge * 1.2:
            bound_type = BoundType.COMPUTE_BOUND
        else:
            bound_type = BoundType.BALANCED
        
        # 计算效率
        achieved = op.achieved_tflops
        efficiency = (achieved / theoretical_peak * 100) if theoretical_peak > 0 else 0
        gap = theoretical_peak - achieved
        
        return RooflinePoint(
            name=op.name,
            arithmetic_intensity=ai,
            achieved_performance=achieved,
            theoretical_peak=theoretical_peak,
            memory_roof=memory_roof,
            bound_type=bound_type,
            efficiency=efficiency,
            gap_to_roof=gap,
        )
    
    def analyze(
        self,
        operators: List[OperatorProfile],
        precision: PrecisionType = PrecisionType.FP16,
    ) -> RooflineAnalysis:
        """
        分析一组算子的 Roofline 性能
        
        Args:
            operators: 算子列表
            precision: 默认计算精度
            
        Returns:
            RooflineAnalysis: 完整的 Roofline 分析结果
        """
        compute_roof = self.get_compute_roof(precision)
        ridge = self.get_ridge_point(precision)
        memory_slope = self.hardware.hbm_bandwidth_gbps / 1000  # TFLOPS per FLOP/Byte
        
        analysis = RooflineAnalysis(
            hardware=self.hardware,
            precision=precision,
            compute_roof_tflops=compute_roof,
            memory_roof_slope=memory_slope,
            ridge_point=ridge,
        )
        
        # 分析每个算子
        compute_bound_count = 0
        memory_bound_count = 0
        total_efficiency = 0
        low_efficiency_ops = []
        
        for op in operators:
            point = self.analyze_operator(op, precision)
            analysis.points.append(point)
            
            if point.bound_type == BoundType.COMPUTE_BOUND:
                compute_bound_count += 1
            elif point.bound_type == BoundType.MEMORY_BOUND:
                memory_bound_count += 1
            
            total_efficiency += point.efficiency
            
            if point.efficiency < 50:
                low_efficiency_ops.append((op.name, point.efficiency))
        
        # 汇总分析
        if operators:
            analysis.avg_efficiency = total_efficiency / len(operators)
        
        # 确定整体受限类型
        if compute_bound_count > memory_bound_count * 2:
            analysis.overall_bound = BoundType.COMPUTE_BOUND
        elif memory_bound_count > compute_bound_count * 2:
            analysis.overall_bound = BoundType.MEMORY_BOUND
        else:
            analysis.overall_bound = BoundType.BALANCED
        
        # 识别瓶颈算子
        low_efficiency_ops.sort(key=lambda x: x[1])
        analysis.bottleneck_operators = [op[0] for op in low_efficiency_ops[:5]]
        
        # 生成建议
        analysis.suggestions = self._generate_suggestions(analysis)
        
        return analysis
    
    def _generate_suggestions(self, analysis: RooflineAnalysis) -> List[str]:
        """生成优化建议"""
        suggestions = []
        
        # 基于整体受限类型
        if analysis.overall_bound == BoundType.MEMORY_BOUND:
            suggestions.append("整体为内存带宽受限，优化方向：")
            suggestions.append("  1. 增加算子计算强度（算子融合、减少中间结果写回）")
            suggestions.append("  2. 优化内存访问模式（连续访问、避免 Bank Conflict）")
            suggestions.append("  3. 利用 L2 Cache（优化 Tiling 策略）")
        elif analysis.overall_bound == BoundType.COMPUTE_BOUND:
            suggestions.append("整体为计算受限，优化方向：")
            suggestions.append("  1. 检查算子实现效率（是否充分利用 Vector/Cube 单元）")
            suggestions.append("  2. 确认是否使用最优精度（FP16/BF16）")
            suggestions.append("  3. 检查是否有冗余计算")
        else:
            suggestions.append("整体接近平衡点，性能较优")
        
        # 基于效率
        if analysis.avg_efficiency < 30:
            suggestions.append(f"平均效率仅 {analysis.avg_efficiency:.1f}%，存在严重性能问题")
        elif analysis.avg_efficiency < 50:
            suggestions.append(f"平均效率 {analysis.avg_efficiency:.1f}%，有较大优化空间")
        
        # 瓶颈算子
        if analysis.bottleneck_operators:
            suggestions.append(f"重点优化算子: {', '.join(analysis.bottleneck_operators[:3])}")
        
        return suggestions
    
    def estimate_theoretical_mfu(
        self,
        model_flops: float,
        model_memory_bytes: float,
        step_time_ms: float,
        num_devices: int = 1,
        precision: PrecisionType = PrecisionType.FP16,
    ) -> Dict[str, Any]:
        """
        估算理论 MFU 上限
        
        Args:
            model_flops: 模型每步 FLOPS
            model_memory_bytes: 模型每步内存访问量
            step_time_ms: 实际 step 时间
            num_devices: 设备数量
            precision: 计算精度
            
        Returns:
            理论 MFU 分析结果
        """
        # 计算模型的计算强度
        ai = model_flops / model_memory_bytes if model_memory_bytes > 0 else 0
        
        compute_roof = self.get_compute_roof(precision) * num_devices
        memory_roof = self.get_memory_roof(ai) * num_devices
        ridge = self.get_ridge_point(precision)
        
        # 理论天花板
        theoretical_roof = min(compute_roof, memory_roof)
        
        # 实际性能
        step_time_s = step_time_ms / 1000
        actual_tflops = model_flops / step_time_s / 1e12
        
        # 实际 MFU（相对于计算峰值）
        actual_mfu = actual_tflops / compute_roof * 100
        
        # 理论最大 MFU（考虑内存带宽限制）
        theoretical_max_mfu = theoretical_roof / compute_roof * 100
        
        # 相对效率（相对于理论天花板）
        roof_efficiency = actual_tflops / theoretical_roof * 100 if theoretical_roof > 0 else 0
        
        # 确定受限类型
        if ai < ridge:
            bound_type = BoundType.MEMORY_BOUND
            bound_desc = "内存带宽受限"
        else:
            bound_type = BoundType.COMPUTE_BOUND
            bound_desc = "计算受限"
        
        return {
            "arithmetic_intensity": ai,
            "ridge_point": ridge,
            "bound_type": bound_type.value,
            "bound_description": bound_desc,
            "compute_roof_tflops": compute_roof,
            "memory_roof_tflops": memory_roof,
            "theoretical_roof_tflops": theoretical_roof,
            "actual_tflops": actual_tflops,
            "actual_mfu_percent": actual_mfu,
            "theoretical_max_mfu_percent": theoretical_max_mfu,
            "roof_efficiency_percent": roof_efficiency,
            "mfu_gap_percent": theoretical_max_mfu - actual_mfu,
        }
    
    def generate_roofline_data(
        self,
        precision: PrecisionType = PrecisionType.FP16,
        ai_range: Tuple[float, float] = (0.1, 1000),
        num_points: int = 100,
    ) -> Dict[str, List[float]]:
        """
        生成 Roofline 图数据（用于可视化）
        
        Args:
            precision: 计算精度
            ai_range: 计算强度范围
            num_points: 数据点数量
            
        Returns:
            包含 x, y 数据的字典
        """
        compute_roof = self.get_compute_roof(precision)
        
        # 生成对数均匀分布的计算强度
        ai_min, ai_max = ai_range
        ais = [
            ai_min * (ai_max / ai_min) ** (i / (num_points - 1))
            for i in range(num_points)
        ]
        
        # 计算每个点的天花板
        roofline = []
        for ai in ais:
            memory_roof = self.get_memory_roof(ai)
            roofline.append(min(compute_roof, memory_roof))
        
        return {
            "arithmetic_intensity": ais,
            "performance_tflops": roofline,
            "compute_roof": compute_roof,
            "ridge_point": self.get_ridge_point(precision),
        }
