"""
带宽效率检查技能

检查 HBM 带宽和通信带宽效率
"""

from typing import Dict, Any, List
import logging

from ..base_skill import (
    BaseSkill,
    SkillMetadata,
    SkillCategory,
    SkillPriority,
    SkillInput,
    SkillOutput,
    SkillResult,
)

logger = logging.getLogger(__name__)


class CheckBandwidthEfficiencySkill(BaseSkill):
    """
    检查带宽效率
    
    对比实测带宽与理论带宽
    """
    
    @property
    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="check_bandwidth_efficiency",
            display_name="检查带宽效率",
            description="检查 HBM/HCCS/RDMA 带宽效率，识别带宽瓶颈",
            category=SkillCategory.COMMUNICATION,
            priority=SkillPriority.HIGH,
            version="1.0.0",
            inputs=[
                SkillInput(
                    name="measured_bandwidth_gbps",
                    type="float",
                    required=True,
                    description="实测带宽 (GB/s)",
                ),
                SkillInput(
                    name="theoretical_bandwidth_gbps",
                    type="float",
                    required=True,
                    description="理论带宽 (GB/s)",
                ),
                SkillInput(
                    name="bandwidth_type",
                    type="str",
                    required=False,
                    default="hbm",
                    description="带宽类型: hbm/hccs/rdma",
                ),
                SkillInput(
                    name="data_size_mb",
                    type="float",
                    required=False,
                    default=0,
                    description="传输数据大小 (MB)，用于判断是否达到峰值",
                ),
            ],
            outputs=[
                SkillOutput(name="efficiency", type="float", description="带宽效率 %"),
                SkillOutput(name="is_bottleneck", type="bool", description="是否为瓶颈"),
                SkillOutput(name="severity", type="str", description="问题严重程度"),
            ],
            tags=["bandwidth", "hbm", "hccs", "rdma", "communication", "bottleneck"],
        )
    
    def execute(
        self,
        measured_bandwidth_gbps: float,
        theoretical_bandwidth_gbps: float,
        bandwidth_type: str = "hbm",
        data_size_mb: float = 0,
        **kwargs,
    ) -> SkillResult:
        """执行带宽效率检查"""
        
        # 计算效率
        efficiency = (measured_bandwidth_gbps / theoretical_bandwidth_gbps * 100 
                     if theoretical_bandwidth_gbps > 0 else 0)
        
        # 不同带宽类型的期望效率阈值
        efficiency_thresholds = {
            "hbm": {"good": 80, "acceptable": 60, "poor": 40},
            "hccs": {"good": 85, "acceptable": 70, "poor": 50},
            "rdma": {"good": 75, "acceptable": 55, "poor": 35},
        }
        
        thresholds = efficiency_thresholds.get(bandwidth_type.lower(), 
                                               efficiency_thresholds["hbm"])
        
        # 判断严重程度
        if efficiency >= thresholds["good"]:
            severity = "正常"
            is_bottleneck = False
        elif efficiency >= thresholds["acceptable"]:
            severity = "轻微"
            is_bottleneck = False
        elif efficiency >= thresholds["poor"]:
            severity = "中等"
            is_bottleneck = True
        else:
            severity = "严重"
            is_bottleneck = True
        
        # 生成建议
        suggestions = []
        
        if is_bottleneck:
            if bandwidth_type.lower() == "hbm":
                suggestions.append("HBM 带宽利用率偏低，可能原因：")
                suggestions.append("  1. 内存访问模式不连续，建议优化数据布局")
                suggestions.append("  2. 小 Tensor 操作过多，建议算子融合")
                suggestions.append("  3. 检查是否有不必要的 Host-Device 数据拷贝")
            
            elif bandwidth_type.lower() == "hccs":
                suggestions.append("HCCS 节点内通信带宽偏低，可能原因：")
                suggestions.append("  1. 通信数据量较小，未达到峰值带宽")
                suggestions.append("  2. 通信与计算重叠不足")
                suggestions.append("  3. 检查 HCCL 配置是否最优")
            
            elif bandwidth_type.lower() == "rdma":
                suggestions.append("RDMA 节点间通信带宽偏低，可能原因：")
                suggestions.append("  1. 网络拥塞或链路质量问题")
                suggestions.append("  2. 小消息通信开销大，建议消息聚合")
                suggestions.append("  3. 检查 RDMA QP 配置")
        
        # 数据大小相关建议
        if data_size_mb > 0:
            if data_size_mb < 1:  # 小于 1MB
                suggestions.append(f"传输数据量较小 ({data_size_mb:.2f}MB)，难以达到峰值带宽是正常现象")
            elif data_size_mb > 100:  # 大于 100MB
                if efficiency < thresholds["acceptable"]:
                    suggestions.append(f"大数据量 ({data_size_mb:.1f}MB) 下效率仍低，建议深入排查")
        
        return SkillResult(
            skill_name=self.metadata.name,
            success=True,
            data={
                "efficiency": round(efficiency, 2),
                "is_bottleneck": is_bottleneck,
                "severity": severity,
                "measured_bandwidth_gbps": measured_bandwidth_gbps,
                "theoretical_bandwidth_gbps": theoretical_bandwidth_gbps,
                "bandwidth_type": bandwidth_type,
                "gap_gbps": round(theoretical_bandwidth_gbps - measured_bandwidth_gbps, 2),
            },
            summary=f"{bandwidth_type.upper()} 带宽效率 {efficiency:.1f}% ({severity})，"
                    f"实测 {measured_bandwidth_gbps:.1f} / 理论 {theoretical_bandwidth_gbps:.1f} GB/s",
            suggestions=suggestions,
            confidence=0.9,
        )


class AnalyzeCollectiveOpsSkill(BaseSkill):
    """
    分析集合通信操作效率
    """
    
    @property
    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="analyze_collective_ops",
            display_name="分析集合通信操作",
            description="分析 AllReduce/ReduceScatter/AllGather 等集合操作的效率",
            category=SkillCategory.COMMUNICATION,
            priority=SkillPriority.HIGH,
            version="1.0.0",
            inputs=[
                SkillInput(name="op_type", type="str", required=True,
                          description="操作类型: allreduce/reducescatter/allgather/all2all"),
                SkillInput(name="data_size_bytes", type="float", required=True,
                          description="数据大小（字节）"),
                SkillInput(name="duration_us", type="float", required=True,
                          description="操作耗时（微秒）"),
                SkillInput(name="world_size", type="int", required=True,
                          description="通信组大小"),
                SkillInput(name="theoretical_bandwidth_gbps", type="float", required=True,
                          description="理论带宽 (GB/s)"),
            ],
            outputs=[
                SkillOutput(name="achieved_bandwidth_gbps", type="float", description="实测带宽"),
                SkillOutput(name="algorithm_bandwidth_gbps", type="float", description="算法带宽"),
                SkillOutput(name="bus_bandwidth_gbps", type="float", description="总线带宽"),
                SkillOutput(name="efficiency", type="float", description="效率 %"),
            ],
            tags=["collective", "allreduce", "communication", "hccl"],
        )
    
    def execute(
        self,
        op_type: str,
        data_size_bytes: float,
        duration_us: float,
        world_size: int,
        theoretical_bandwidth_gbps: float,
        **kwargs,
    ) -> SkillResult:
        """分析集合操作效率"""
        
        # 转换单位
        data_size_gb = data_size_bytes / (1024**3)
        duration_s = duration_us / 1e6
        
        # 计算实测带宽
        achieved_bw = data_size_gb / duration_s if duration_s > 0 else 0
        
        # 计算算法带宽（考虑集合操作的数据放大因子）
        # Ring AllReduce: 每个元素传输 2*(n-1)/n 次
        # ReduceScatter/AllGather: 每个元素传输 (n-1)/n 次
        op_type_lower = op_type.lower()
        if op_type_lower == "allreduce":
            factor = 2 * (world_size - 1) / world_size
        elif op_type_lower in ("reducescatter", "allgather"):
            factor = (world_size - 1) / world_size
        elif op_type_lower == "all2all":
            factor = (world_size - 1) / world_size
        else:
            factor = 1.0
        
        algorithm_bw = achieved_bw * factor
        bus_bw = algorithm_bw  # 对于 Ring 算法，bus bandwidth = algorithm bandwidth
        
        # 计算效率
        efficiency = (bus_bw / theoretical_bandwidth_gbps * 100 
                     if theoretical_bandwidth_gbps > 0 else 0)
        
        # 生成建议
        suggestions = []
        if efficiency < 50:
            suggestions.append(f"{op_type} 效率 < 50%，可能存在问题")
            if data_size_bytes < 1024 * 1024:  # < 1MB
                suggestions.append("数据量较小，考虑使用 Tree 算法或延迟通信")
            else:
                suggestions.append("检查网络链路状态和 HCCL 配置")
        elif efficiency < 70:
            suggestions.append(f"{op_type} 效率在 50-70%，有优化空间")
        
        return SkillResult(
            skill_name=self.metadata.name,
            success=True,
            data={
                "op_type": op_type,
                "achieved_bandwidth_gbps": round(achieved_bw, 2),
                "algorithm_bandwidth_gbps": round(algorithm_bw, 2),
                "bus_bandwidth_gbps": round(bus_bw, 2),
                "efficiency": round(efficiency, 2),
                "data_size_mb": round(data_size_bytes / (1024**2), 2),
                "duration_ms": round(duration_us / 1000, 2),
            },
            summary=f"{op_type} 效率 {efficiency:.1f}%，总线带宽 {bus_bw:.1f} GB/s",
            suggestions=suggestions,
            confidence=0.9,
        )
