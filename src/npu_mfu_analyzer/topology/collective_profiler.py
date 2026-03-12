"""
Collective Op Profiler - 集合通信分析器

深入分析 AllReduce、ReduceScatter、AllGather 等集合操作的性能
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum
import logging
import math

logger = logging.getLogger(__name__)


class CollectiveOpType(Enum):
    """集合操作类型"""
    ALLREDUCE = "allreduce"
    REDUCESCATTER = "reducescatter"
    ALLGATHER = "allgather"
    BROADCAST = "broadcast"
    REDUCE = "reduce"
    ALL2ALL = "all2all"
    SEND = "send"
    RECV = "recv"
    UNKNOWN = "unknown"


class CollectiveAlgorithm(Enum):
    """集合操作算法"""
    RING = "ring"
    TREE = "tree"
    RECURSIVE_HALVING = "recursive_halving"
    BUCKET = "bucket"
    DIRECT = "direct"
    UNKNOWN = "unknown"


@dataclass
class CollectiveOpStats:
    """单个集合操作的统计信息"""
    op_type: CollectiveOpType
    name: str
    
    # 基本信息
    data_size_bytes: int = 0
    duration_us: float = 0.0
    count: int = 1  # 调用次数
    
    # 通信组信息
    group_size: int = 1
    group_name: str = ""
    
    # 带宽指标
    achieved_bandwidth_gbps: float = 0.0
    algorithm_bandwidth_gbps: float = 0.0  # 算法带宽
    bus_bandwidth_gbps: float = 0.0  # 总线带宽
    
    # 理论值
    theoretical_bandwidth_gbps: float = 0.0
    bandwidth_efficiency: float = 0.0
    
    # 算法类型
    algorithm: CollectiveAlgorithm = CollectiveAlgorithm.UNKNOWN
    
    def __post_init__(self):
        """计算带宽"""
        if self.duration_us > 0 and self.data_size_bytes > 0:
            # 实测带宽 (GB/s)
            self.achieved_bandwidth_gbps = (
                self.data_size_bytes / self.duration_us / 1000
            )
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "op_type": self.op_type.value,
            "name": self.name,
            "data_size_bytes": self.data_size_bytes,
            "duration_us": self.duration_us,
            "count": self.count,
            "group_size": self.group_size,
            "achieved_bandwidth_gbps": self.achieved_bandwidth_gbps,
            "algorithm_bandwidth_gbps": self.algorithm_bandwidth_gbps,
            "bus_bandwidth_gbps": self.bus_bandwidth_gbps,
            "theoretical_bandwidth_gbps": self.theoretical_bandwidth_gbps,
            "bandwidth_efficiency": self.bandwidth_efficiency,
        }


@dataclass
class CollectiveProfilingResult:
    """集合通信分析结果"""
    # 各操作类型的统计
    op_stats: Dict[CollectiveOpType, List[CollectiveOpStats]] = field(default_factory=dict)
    
    # 汇总指标
    total_comm_time_ms: float = 0.0
    total_data_volume_gb: float = 0.0
    avg_bandwidth_efficiency: float = 0.0
    
    # 按类型统计的时间占比
    time_by_op_type: Dict[str, float] = field(default_factory=dict)
    
    # 瓶颈操作
    bottleneck_ops: List[CollectiveOpStats] = field(default_factory=list)
    
    # 低效操作（带宽利用率低）
    inefficient_ops: List[CollectiveOpStats] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_comm_time_ms": self.total_comm_time_ms,
            "total_data_volume_gb": self.total_data_volume_gb,
            "avg_bandwidth_efficiency": self.avg_bandwidth_efficiency,
            "time_by_op_type": self.time_by_op_type,
            "bottleneck_ops_count": len(self.bottleneck_ops),
            "inefficient_ops_count": len(self.inefficient_ops),
        }
    
    def to_prompt_text(self) -> str:
        """转换为 LLM Prompt 格式"""
        lines = [
            "## 集合通信分析结果",
            "",
            "### 总体概况",
            f"- **总通信时间**: {self.total_comm_time_ms:.2f} ms",
            f"- **总数据量**: {self.total_data_volume_gb:.2f} GB",
            f"- **平均带宽效率**: {self.avg_bandwidth_efficiency:.1%}",
            "",
            "### 各操作类型时间占比",
        ]
        
        # 按时间占比排序
        sorted_types = sorted(
            self.time_by_op_type.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        for op_type, ratio in sorted_types:
            if ratio > 0.01:  # 只显示占比超过 1% 的
                lines.append(f"- **{op_type}**: {ratio:.1%}")
        
        # 瓶颈操作
        if self.bottleneck_ops:
            lines.extend([
                "",
                "### 瓶颈操作 (耗时最长)",
            ])
            for op in self.bottleneck_ops[:5]:
                lines.append(
                    f"- `{op.name}`: {op.duration_us/1000:.2f}ms, "
                    f"效率 {op.bandwidth_efficiency:.1%}"
                )
        
        # 低效操作
        if self.inefficient_ops:
            lines.extend([
                "",
                "### 低效操作 (带宽利用率 < 50%)",
            ])
            for op in self.inefficient_ops[:5]:
                lines.append(
                    f"- `{op.name}`: 带宽效率 {op.bandwidth_efficiency:.1%}, "
                    f"数据量 {op.data_size_bytes/1e9:.3f}GB"
                )
        
        return "\n".join(lines)


class CollectiveProfiler:
    """
    集合通信分析器
    
    分析 AllReduce、ReduceScatter、AllGather 等集合操作的性能，
    计算算法带宽、总线带宽、带宽效率等指标。
    """
    
    def __init__(
        self,
        theoretical_bandwidth_gbps: float = 56.0,  # HCCS 默认带宽
    ):
        """
        Args:
            theoretical_bandwidth_gbps: 理论带宽 (GB/s)
        """
        self.theoretical_bandwidth = theoretical_bandwidth_gbps
    
    def profile(
        self,
        comm_events: List[Dict[str, Any]],
        group_sizes: Optional[Dict[str, int]] = None,
    ) -> CollectiveProfilingResult:
        """
        分析集合通信事件
        
        Args:
            comm_events: 通信事件列表
            group_sizes: 通信组大小映射 {group_name: size}
            
        Returns:
            CollectiveProfilingResult
        """
        if group_sizes is None:
            group_sizes = {}
        
        op_stats = {}
        total_time = 0.0
        total_data = 0.0
        
        for event in comm_events:
            stat = self._parse_event(event, group_sizes)
            if stat is None:
                continue
            
            # 计算带宽指标
            self._calculate_bandwidth_metrics(stat)
            
            # 按类型分组
            if stat.op_type not in op_stats:
                op_stats[stat.op_type] = []
            op_stats[stat.op_type].append(stat)
            
            total_time += stat.duration_us
            total_data += stat.data_size_bytes
        
        # 计算时间占比
        time_by_type = {}
        for op_type, stats in op_stats.items():
            type_time = sum(s.duration_us for s in stats)
            time_by_type[op_type.value] = type_time / total_time if total_time > 0 else 0
        
        # 计算平均带宽效率
        all_efficiencies = []
        for stats in op_stats.values():
            for s in stats:
                if s.bandwidth_efficiency > 0:
                    all_efficiencies.append(s.bandwidth_efficiency)
        avg_efficiency = sum(all_efficiencies) / len(all_efficiencies) if all_efficiencies else 0
        
        # 识别瓶颈操作（耗时最长的）
        all_ops = []
        for stats in op_stats.values():
            all_ops.extend(stats)
        bottleneck_ops = sorted(all_ops, key=lambda x: x.duration_us, reverse=True)[:10]
        
        # 识别低效操作（带宽利用率低于 50%）
        inefficient_ops = [
            op for op in all_ops 
            if 0 < op.bandwidth_efficiency < 0.5
        ]
        inefficient_ops = sorted(
            inefficient_ops, 
            key=lambda x: x.bandwidth_efficiency
        )[:10]
        
        return CollectiveProfilingResult(
            op_stats=op_stats,
            total_comm_time_ms=total_time / 1000,
            total_data_volume_gb=total_data / 1e9,
            avg_bandwidth_efficiency=avg_efficiency,
            time_by_op_type=time_by_type,
            bottleneck_ops=bottleneck_ops,
            inefficient_ops=inefficient_ops,
        )
    
    def _parse_event(
        self,
        event: Dict[str, Any],
        group_sizes: Dict[str, int],
    ) -> Optional[CollectiveOpStats]:
        """解析单个事件"""
        name = event.get("name", "").lower()
        
        # 识别操作类型
        op_type = self._identify_op_type(name)
        if op_type == CollectiveOpType.UNKNOWN:
            return None
        
        # 提取参数
        args = event.get("args", {})
        data_size = args.get("data_size", args.get("size", 0))
        duration = event.get("dur", event.get("duration", 0))
        group_name = str(args.get("group_name", args.get("groupName", "")))
        group_size = args.get("group_size", args.get("commSize", 1))
        
        # 从 group_sizes 查找
        if not group_size or group_size == 1:
            group_size = group_sizes.get(group_name, 1)
        
        return CollectiveOpStats(
            op_type=op_type,
            name=event.get("name", "unknown"),
            data_size_bytes=int(data_size),
            duration_us=float(duration),
            group_size=int(group_size),
            group_name=group_name,
            theoretical_bandwidth_gbps=self.theoretical_bandwidth,
        )
    
    def _identify_op_type(self, name: str) -> CollectiveOpType:
        """识别操作类型"""
        name_lower = name.lower()
        
        if "allreduce" in name_lower:
            return CollectiveOpType.ALLREDUCE
        elif "reducescatter" in name_lower or "reduce_scatter" in name_lower:
            return CollectiveOpType.REDUCESCATTER
        elif "allgather" in name_lower or "all_gather" in name_lower:
            return CollectiveOpType.ALLGATHER
        elif "broadcast" in name_lower:
            return CollectiveOpType.BROADCAST
        elif "all2all" in name_lower or "alltoall" in name_lower:
            return CollectiveOpType.ALL2ALL
        elif "reduce" in name_lower and "scatter" not in name_lower:
            return CollectiveOpType.REDUCE
        elif "send" in name_lower:
            return CollectiveOpType.SEND
        elif "recv" in name_lower:
            return CollectiveOpType.RECV
        
        return CollectiveOpType.UNKNOWN
    
    def _calculate_bandwidth_metrics(self, stat: CollectiveOpStats):
        """
        计算带宽指标
        
        对于不同的集合操作，算法带宽和总线带宽的计算公式不同：
        
        AllReduce (Ring):
            - 算法带宽 = 2 * (n-1) / n * data_size / time
            - 总线带宽 = 2 * data_size * (n-1) / time
            
        ReduceScatter / AllGather:
            - 算法带宽 = (n-1) / n * data_size / time
            - 总线带宽 = data_size * (n-1) / time
        """
        n = stat.group_size
        data_size = stat.data_size_bytes
        duration = stat.duration_us
        
        if duration <= 0 or data_size <= 0 or n <= 1:
            return
        
        # 基础带宽 (GB/s)
        base_bw = data_size / duration / 1000
        
        if stat.op_type == CollectiveOpType.ALLREDUCE:
            # AllReduce: 2 * (n-1) / n 的数据量
            stat.algorithm_bandwidth_gbps = 2 * (n - 1) / n * base_bw
            stat.bus_bandwidth_gbps = 2 * (n - 1) * base_bw
        elif stat.op_type in (CollectiveOpType.REDUCESCATTER, CollectiveOpType.ALLGATHER):
            # ReduceScatter / AllGather: (n-1) / n 的数据量
            stat.algorithm_bandwidth_gbps = (n - 1) / n * base_bw
            stat.bus_bandwidth_gbps = (n - 1) * base_bw
        elif stat.op_type == CollectiveOpType.ALL2ALL:
            # All2All: 每个 rank 发送 n-1 份数据
            stat.algorithm_bandwidth_gbps = (n - 1) * base_bw
            stat.bus_bandwidth_gbps = (n - 1) * base_bw
        else:
            # 其他操作使用简单带宽
            stat.algorithm_bandwidth_gbps = base_bw
            stat.bus_bandwidth_gbps = base_bw
        
        # 计算带宽效率
        if stat.theoretical_bandwidth_gbps > 0:
            stat.bandwidth_efficiency = (
                stat.algorithm_bandwidth_gbps / stat.theoretical_bandwidth_gbps
            )
            stat.bandwidth_efficiency = min(stat.bandwidth_efficiency, 1.0)
    
    def estimate_optimal_algorithm(
        self,
        op_type: CollectiveOpType,
        data_size: int,
        group_size: int,
    ) -> CollectiveAlgorithm:
        """
        根据数据量和组大小推荐最优算法
        
        启发式规则：
        - 小数据量 (< 256KB): Tree 算法（低延迟）
        - 大数据量 (> 256KB): Ring 算法（高带宽）
        - All2All: Direct 算法
        """
        if op_type == CollectiveOpType.ALL2ALL:
            return CollectiveAlgorithm.DIRECT
        
        # 256KB 阈值
        if data_size < 256 * 1024:
            return CollectiveAlgorithm.TREE
        else:
            return CollectiveAlgorithm.RING


def profile_collective_ops_from_loader(loader) -> CollectiveProfilingResult:
    """
    从 ProfilingLoader 分析集合通信
    
    Args:
        loader: ProfilingLoader 实例
        
    Returns:
        CollectiveProfilingResult
    """
    profiler = CollectiveProfiler()
    
    comm_events = []
    try:
        overlap_events = loader.get_overlap_events()
        comm_events = overlap_events.get("hccl", [])
    except Exception as e:
        logger.warning(f"Failed to collect comm events: {e}")
    
    return profiler.profile(comm_events)
