"""
通信矩阵深度分析模块

提供硬件级通信分析能力：
- 链路性能分析：HCCS/RDMA 链路级带宽利用率、延迟分析
- 拓扑矩阵构建：NPU 间通信路径、跨 Ring 开销分析
- 慢链路检测：带宽异常、延迟异常检测
- 通信矩阵汇总：rank 间通信数据量、带宽、时间统计
"""

import logging
import sqlite3
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict
from enum import Enum

import pandas as pd

from npu_mfu_analyzer.topology.topology_analyzer import TopologyAnalyzer, TopologyInfo, LinkType

logger = logging.getLogger(__name__)


class TransportType(Enum):
    """传输类型"""
    HCCS = "hccs"           # 节点内高速互联 (High-speed Chip-to-Chip Scalability)
    RDMA = "rdma"           # 节点间 RDMA
    ROCE = "roce"           # RoCE v2
    PCIE = "pcie"           # PCIe
    HBM = "hbm"             # HBM 内存传输
    UNKNOWN = "unknown"


class CommOpType(Enum):
    """通信操作类型"""
    # 集合通信 (Collective)
    ALL_REDUCE = "hcom_all_reduce"
    ALL_GATHER = "hcom_all_gather"
    REDUCE_SCATTER = "hcom_reduce_scatter"
    BROADCAST = "hcom_broadcast"
    ALL_TO_ALL = "hcom_all_to_all"
    REDUCE = "hcom_reduce"
    ALL_REDUCE_MC2 = "hcom_all_reduce_mc2"

    # P2P 通信
    SEND = "hcom_send"
    RECEIVE = "hcom_receive"
    BATCH_SEND_RECV = "hcom_batchsendrecv"

    # 未知类型
    UNKNOWN = "unknown"

    @classmethod
    def from_op_name(cls, op_name: str) -> "CommOpType":
        """从操作名称推断类型"""
        op_lower = op_name.lower()
        for op_type in cls:
            if op_type.value in op_lower:
                return op_type
        return cls.UNKNOWN

    @property
    def is_collective(self) -> bool:
        """是否为集合通信"""
        return self in {
            CommOpType.ALL_REDUCE,
            CommOpType.ALL_GATHER,
            CommOpType.REDUCE_SCATTER,
            CommOpType.BROADCAST,
            CommOpType.ALL_TO_ALL,
            CommOpType.REDUCE,
            CommOpType.ALL_REDUCE_MC2,
        }

    @property
    def is_p2p(self) -> bool:
        """是否为点对点通信"""
        return self in {
            CommOpType.SEND,
            CommOpType.RECEIVE,
            CommOpType.BATCH_SEND_RECV,
        }


@dataclass
class LinkMetrics:
    """
    链路性能指标

    记录两个 rank 之间通信链路的详细性能数据
    """
    # 基本信息
    src_rank: int
    dst_rank: int
    transport_type: TransportType

    # 带宽指标
    theoretical_bandwidth_gbps: float = 0.0    # 理论带宽 (GB/s)
    achieved_bandwidth_gbps: float = 0.0       # 实测带宽 (GB/s)
    bandwidth_utilization: float = 0.0          # 带宽利用率 (0-1)
    peak_bandwidth_gbps: float = 0.0            # 峰值带宽 (GB/s)

    # 数据量指标
    total_transit_size_mb: float = 0.0         # 总传输数据量 (MB)
    total_transit_time_ms: float = 0.0         # 总传输时间 (ms)

    # 延迟指标
    avg_latency_us: float = 0.0                # 平均延迟 (us)
    min_latency_us: float = float('inf')       # 最小延迟
    max_latency_us: float = 0.0                # 最大延迟
    p50_latency_us: float = 0.0                # P50 延迟
    p99_latency_us: float = 0.0                # P99 延迟

    # 操作统计
    op_count: int = 0                          # 操作次数
    large_packet_ratio: float = 0.0            # 大包比例
    small_packet_ratio: float = 0.0            # 小包比例

    # 时间分解
    wait_time_ms: float = 0.0                  # 等待时间 (ms)
    sync_time_ms: float = 0.0                  # 同步时间 (ms)
    idle_time_ms: float = 0.0                  # 空闲时间 (ms)

    # 异常标记
    is_slow_link: bool = False                 # 是否慢链路
    is_bottleneck: bool = False                # 是否瓶颈链路
    anomaly_type: Optional[str] = None         # 异常类型
    anomaly_score: float = 0.0                 # 异常评分 (0-1)

    def compute_derived_metrics(self):
        """计算派生指标"""
        # 计算带宽利用率
        if self.theoretical_bandwidth_gbps > 0:
            self.bandwidth_utilization = min(
                self.achieved_bandwidth_gbps / self.theoretical_bandwidth_gbps,
                1.0
            )

        # 计算实测带宽
        if self.total_transit_time_ms > 0:
            self.achieved_bandwidth_gbps = self.total_transit_size_mb / self.total_transit_time_ms

    def compute_anomaly_score(self, avg_bandwidth: float, std_bandwidth: float) -> float:
        """
        计算异常评分

        Args:
            avg_bandwidth: 同类型链路平均带宽
            std_bandwidth: 同类型链路带宽标准差

        Returns:
            异常评分 (0-1)，越高越异常
        """
        if avg_bandwidth <= 0:
            return 0.0

        # Z-score 计算异常程度
        if std_bandwidth > 0:
            z_score = (avg_bandwidth - self.achieved_bandwidth_gbps) / std_bandwidth
            # 转换为 0-1 评分
            self.anomaly_score = max(0.0, min(1.0, z_score / 3.0))
        else:
            # 无标准差时，使用相对偏差
            relative_dev = (avg_bandwidth - self.achieved_bandwidth_gbps) / avg_bandwidth
            self.anomaly_score = max(0.0, min(1.0, relative_dev))

        return self.anomaly_score

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "src_rank": self.src_rank,
            "dst_rank": self.dst_rank,
            "transport_type": self.transport_type.value,
            "theoretical_bandwidth_gbps": round(self.theoretical_bandwidth_gbps, 2),
            "achieved_bandwidth_gbps": round(self.achieved_bandwidth_gbps, 2),
            "bandwidth_utilization": f"{self.bandwidth_utilization:.1%}",
            "peak_bandwidth_gbps": round(self.peak_bandwidth_gbps, 2),
            "total_transit_size_mb": round(self.total_transit_size_mb, 2),
            "total_transit_time_ms": round(self.total_transit_time_ms, 2),
            "avg_latency_us": round(self.avg_latency_us, 2),
            "op_count": self.op_count,
            "wait_time_ms": round(self.wait_time_ms, 2),
            "sync_time_ms": round(self.sync_time_ms, 2),
            "is_slow_link": self.is_slow_link,
            "is_bottleneck": self.is_bottleneck,
            "anomaly_type": self.anomaly_type,
            "anomaly_score": round(self.anomaly_score, 3),
        }

    def to_summary(self) -> str:
        """生成简短摘要"""
        return (
            f"Rank {self.src_rank} ↔ {self.dst_rank}: "
            f"{self.achieved_bandwidth_gbps:.2f} GB/s "
            f"({self.bandwidth_utilization:.1%} util, {self.transport_type.value})"
        )


@dataclass
class CommunicationMatrix:
    """
    通信矩阵

    记录整个集群的通信拓扑和性能数据
    """
    world_size: int

    # 链路指标矩阵: link_metrics[(src_rank, dst_rank)] = LinkMetrics
    link_metrics: Dict[Tuple[int, int], LinkMetrics] = field(default_factory=dict)

    # 拓扑信息
    num_machines: int = 1
    npus_per_machine: int = 8

    # 统计摘要
    total_comm_data_mb: float = 0.0            # 总通信数据量
    total_comm_time_ms: float = 0.0            # 总通信时间
    avg_bandwidth_gbps: float = 0.0            # 平均带宽
    peak_bandwidth_gbps: float = 0.0           # 峰值带宽

    # 通信比例
    intra_node_ratio: float = 0.0              # 节点内通信比例
    inter_node_ratio: float = 0.0              # 节点间通信比例
    hccs_ratio: float = 0.0                    # HCCS 通信比例
    rdma_ratio: float = 0.0                    # RDMA 通信比例

    # 异常链路
    slow_links: List[LinkMetrics] = field(default_factory=list)
    bottleneck_links: List[LinkMetrics] = field(default_factory=list)

    # 通信热点
    top_comm_pairs: List[LinkMetrics] = field(default_factory=list)

    def get_link(self, src_rank: int, dst_rank: int) -> Optional[LinkMetrics]:
        """获取链路指标"""
        key = (min(src_rank, dst_rank), max(src_rank, dst_rank))
        return self.link_metrics.get(key)

    def set_link(self, src_rank: int, dst_rank: int, metrics: LinkMetrics):
        """设置链路指标"""
        key = (min(src_rank, dst_rank), max(src_rank, dst_rank))
        self.link_metrics[key] = metrics

    def get_matrix_2d(self) -> List[List[float]]:
        """
        获取 2D 带宽矩阵 (用于可视化)

        Returns:
            world_size x world_size 矩阵，matrix[i][j] 表示 rank i 到 rank j 的带宽
        """
        matrix = [[0.0] * self.world_size for _ in range(self.world_size)]
        for (src, dst), metrics in self.link_metrics.items():
            matrix[src][dst] = metrics.achieved_bandwidth_gbps
            matrix[dst][src] = metrics.achieved_bandwidth_gbps
        return matrix

    def get_utilization_matrix_2d(self) -> List[List[float]]:
        """获取 2D 利用率矩阵"""
        matrix = [[0.0] * self.world_size for _ in range(self.world_size)]
        for (src, dst), metrics in self.link_metrics.items():
            matrix[src][dst] = metrics.bandwidth_utilization
            matrix[dst][src] = metrics.bandwidth_utilization
        return matrix

    def compute_summary(self):
        """计算统计摘要"""
        if not self.link_metrics:
            return

        # 计算总量
        self.total_comm_data_mb = sum(
            m.total_transit_size_mb for m in self.link_metrics.values()
        )
        self.total_comm_time_ms = sum(
            m.total_transit_time_ms for m in self.link_metrics.values()
        )

        # 计算平均带宽
        if self.total_comm_time_ms > 0:
            self.avg_bandwidth_gbps = self.total_comm_data_mb / self.total_comm_time_ms

        # 计算峰值带宽
        self.peak_bandwidth_gbps = max(
            (m.achieved_bandwidth_gbps for m in self.link_metrics.values()),
            default=0.0
        )

        # 计算通信比例
        intra_size = 0.0
        inter_size = 0.0
        hccs_size = 0.0
        rdma_size = 0.0

        for metrics in self.link_metrics.values():
            size = metrics.total_transit_size_mb
            if metrics.transport_type == TransportType.HCCS:
                intra_size += size
                hccs_size += size
            elif metrics.transport_type in (TransportType.RDMA, TransportType.ROCE):
                inter_size += size
                rdma_size += size

        if self.total_comm_data_mb > 0:
            self.intra_node_ratio = intra_size / self.total_comm_data_mb
            self.inter_node_ratio = inter_size / self.total_comm_data_mb
            self.hccs_ratio = hccs_size / self.total_comm_data_mb
            self.rdma_ratio = rdma_size / self.total_comm_data_mb

        # 计算通信热点 (Top 10)
        self.top_comm_pairs = sorted(
            self.link_metrics.values(),
            key=lambda m: m.total_transit_size_mb,
            reverse=True
        )[:10]

    def to_prompt_text(self) -> str:
        """转换为 LLM Prompt 格式"""
        lines = [
            "## 通信矩阵分析结果",
            "",
            "### 概览",
            f"- World Size: {self.world_size}",
            f"- 机器数: {self.num_machines}",
            f"- NPU/机器: {self.npus_per_machine}",
            f"- 总通信数据量: {self.total_comm_data_mb:.2f} MB",
            f"- 总通信时间: {self.total_comm_time_ms:.2f} ms",
            f"- 平均带宽: {self.avg_bandwidth_gbps:.2f} GB/s",
            f"- 峰值带宽: {self.peak_bandwidth_gbps:.2f} GB/s",
            "",
            "### 通信分布",
            f"- 节点内 (HCCS): {self.intra_node_ratio:.1%}",
            f"- 节点间 (RDMA): {self.inter_node_ratio:.1%}",
            "",
        ]

        if self.slow_links:
            lines.extend([
                "### 慢链路检测",
                f"检测到 {len(self.slow_links)} 条慢链路：",
            ])
            for link in self.slow_links[:5]:
                lines.append(f"  - {link.to_summary()}")
            lines.append("")

        if self.bottleneck_links:
            lines.extend([
                "### 瓶颈链路",
                f"检测到 {len(self.bottleneck_links)} 条高负载链路：",
            ])
            for link in self.bottleneck_links[:5]:
                lines.append(f"  - {link.to_summary()}")
            lines.append("")

        if self.top_comm_pairs:
            lines.extend([
                "### Top 通信热点 (按数据量)",
            ])
            for i, link in enumerate(self.top_comm_pairs[:5], 1):
                lines.append(f"  {i}. {link.to_summary()}")

        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "world_size": self.world_size,
            "num_machines": self.num_machines,
            "npus_per_machine": self.npus_per_machine,
            "total_comm_data_mb": round(self.total_comm_data_mb, 2),
            "total_comm_time_ms": round(self.total_comm_time_ms, 2),
            "avg_bandwidth_gbps": round(self.avg_bandwidth_gbps, 2),
            "peak_bandwidth_gbps": round(self.peak_bandwidth_gbps, 2),
            "intra_node_ratio": f"{self.intra_node_ratio:.1%}",
            "inter_node_ratio": f"{self.inter_node_ratio:.1%}",
            "slow_links_count": len(self.slow_links),
            "bottleneck_links_count": len(self.bottleneck_links),
            "link_metrics": [m.to_dict() for m in self.link_metrics.values()],
        }


@dataclass
class CommOpStatistics:
    """
    通信操作统计

    记录单个通信操作的详细统计信息
    """
    op_name: str                               # 操作名称
    op_type: CommOpType                        # 操作类型
    group_name: str                            # 通信组名称

    # 时间统计
    total_time_ms: float = 0.0                 # 总时间
    transit_time_ms: float = 0.0               # 传输时间
    wait_time_ms: float = 0.0                  # 等待时间
    sync_time_ms: float = 0.0                  # 同步时间
    idle_time_ms: float = 0.0                  # 空闲时间

    # 带宽统计
    total_size_mb: float = 0.0                 # 总数据量
    avg_bandwidth_gbps: float = 0.0            # 平均带宽
    peak_bandwidth_gbps: float = 0.0           # 峰值带宽

    # 调用统计
    call_count: int = 0                        # 调用次数
    participating_ranks: List[int] = field(default_factory=list)

    # 效率指标
    wait_ratio: float = 0.0                    # 等待时间比例
    sync_ratio: float = 0.0                    # 同步时间比例
    efficiency: float = 0.0                    # 效率 (transit_time / total_time)

    def compute_ratios(self):
        """计算时间比例"""
        total = self.transit_time_ms + self.wait_time_ms
        if total > 0:
            self.wait_ratio = self.wait_time_ms / total

        total = self.transit_time_ms + self.sync_time_ms
        if total > 0:
            self.sync_ratio = self.sync_time_ms / total

        if self.total_time_ms > 0:
            self.efficiency = self.transit_time_ms / self.total_time_ms

        if self.transit_time_ms > 0:
            self.avg_bandwidth_gbps = self.total_size_mb / self.transit_time_ms

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "op_name": self.op_name,
            "op_type": self.op_type.value,
            "group_name": self.group_name,
            "total_time_ms": round(self.total_time_ms, 2),
            "transit_time_ms": round(self.transit_time_ms, 2),
            "wait_time_ms": round(self.wait_time_ms, 2),
            "sync_time_ms": round(self.sync_time_ms, 2),
            "total_size_mb": round(self.total_size_mb, 2),
            "avg_bandwidth_gbps": round(self.avg_bandwidth_gbps, 2),
            "call_count": self.call_count,
            "wait_ratio": f"{self.wait_ratio:.1%}",
            "sync_ratio": f"{self.sync_ratio:.1%}",
            "efficiency": f"{self.efficiency:.1%}",
        }


@dataclass
class CommunicationMatrixReport:
    """
    通信矩阵分析报告

    包含完整的通信分析结果和建议
    """
    # 基本信息
    profiling_path: str
    world_size: int

    # 通信矩阵
    matrix: CommunicationMatrix

    # 操作统计
    op_statistics: List[CommOpStatistics] = field(default_factory=list)

    # 分析结论
    summary: str = ""
    recommendations: List[str] = field(default_factory=list)

    # 告警信息
    warnings: List[str] = field(default_factory=list)

    def to_prompt_text(self) -> str:
        """转换为 LLM Prompt 格式"""
        lines = [
            self.matrix.to_prompt_text(),
            "",
        ]

        if self.warnings:
            lines.extend([
                "### 告警",
            ])
            for warning in self.warnings:
                lines.append(f"  ⚠️ {warning}")
            lines.append("")

        if self.recommendations:
            lines.extend([
                "### 优化建议",
            ])
            for i, rec in enumerate(self.recommendations, 1):
                lines.append(f"  {i}. {rec}")

        return "\n".join(lines)


# ============================================================================
# 通信矩阵分析器
# ============================================================================

class CommunicationMatrixAnalyzer:
    """
    通信矩阵深度分析器

    功能：
    1. 解析通信矩阵数据
    2. 构建链路级性能指标
    3. 检测慢链路和瓶颈链路
    4. 分析通信效率
    5. 生成优化建议
    """

    # 慢链路检测阈值
    SLOW_LINK_BANDWIDTH_THRESHOLD = 0.7    # 低于平均值 70% 为慢链路
    BOTTLENECK_UTIL_THRESHOLD = 0.9        # 利用率超过 90% 为瓶颈
    ANOMALY_ZSCORE_THRESHOLD = 2.0         # Z-score 超过 2.0 为异常

    def __init__(
        self,
        world_size: int,
        npus_per_machine: int = 8,
        hccs_bandwidth: float = 56.0,      # GB/s
        rdma_bandwidth: float = 25.0,      # GB/s
        roce_bandwidth: float = 25.0,      # GB/s
    ):
        """
        初始化通信矩阵分析器

        Args:
            world_size: 总进程数
            npus_per_machine: 每机器 NPU 数
            hccs_bandwidth: HCCS 理论带宽 (GB/s)
            rdma_bandwidth: RDMA 理论带宽 (GB/s)
            roce_bandwidth: RoCE 理论带宽 (GB/s)
        """
        self.world_size = world_size
        self.npus_per_machine = npus_per_machine
        self.num_machines = max(1, (world_size + npus_per_machine - 1) // npus_per_machine)

        self.hccs_bandwidth = hccs_bandwidth
        self.rdma_bandwidth = rdma_bandwidth
        self.roce_bandwidth = roce_bandwidth

        # 拓扑分析器
        self._topology_analyzer = TopologyAnalyzer(
            world_size=world_size,
            npus_per_machine=npus_per_machine,
            hccs_bandwidth=hccs_bandwidth,
            rdma_bandwidth=rdma_bandwidth,
        )
        self._topology: Optional[TopologyInfo] = None

        # 分析结果
        self._matrix: Optional[CommunicationMatrix] = None

    def analyze_from_db(
        self,
        db_path: str,
        step_ids: Optional[List[int]] = None,
    ) -> CommunicationMatrix:
        """
        从 DB 文件分析通信矩阵

        Args:
            db_path: profiling DB 文件路径
            step_ids: 要分析的 step ID 列表，None 表示全部

        Returns:
            CommunicationMatrix
        """
        logger.info(f"开始分析通信矩阵: {db_path}")

        # 初始化矩阵
        self._matrix = CommunicationMatrix(
            world_size=self.world_size,
            num_machines=self.num_machines,
            npus_per_machine=self.npus_per_machine,
        )

        # 构建拓扑
        self._topology = self._topology_analyzer.build_topology()

        # 读取通信矩阵数据
        matrix_data = self._read_matrix_data(db_path, step_ids)
        if not matrix_data.empty:
            self._process_matrix_data(matrix_data)

        # 读取带宽数据
        bandwidth_data = self._read_bandwidth_data(db_path, step_ids)
        if not bandwidth_data.empty:
            self._process_bandwidth_data(bandwidth_data)

        # 读取时间数据
        time_data = self._read_time_data(db_path, step_ids)
        if not time_data.empty:
            self._process_time_data(time_data)

        # 计算派生指标
        self._compute_derived_metrics()

        # 检测慢链路和瓶颈
        self._detect_anomalies()

        # 计算摘要
        self._matrix.compute_summary()

        logger.info(
            f"通信矩阵分析完成: {len(self._matrix.link_metrics)} 条链路, "
            f"{len(self._matrix.slow_links)} 条慢链路, "
            f"{len(self._matrix.bottleneck_links)} 条瓶颈链路"
        )

        return self._matrix

    def analyze_from_events(
        self,
        comm_events: List[Dict[str, Any]],
    ) -> CommunicationMatrix:
        """
        从通信事件列表分析通信矩阵

        Args:
            comm_events: 通信事件列表，每个事件包含：
                - src_rank, dst_rank: 源/目标 rank
                - data_size: 数据量 (bytes)
                - duration: 持续时间 (us)
                - op_name: 操作名称
                - group_name: 通信组名称
                - transport_type: 传输类型 (可选)

        Returns:
            CommunicationMatrix
        """
        logger.info(f"从事件列表分析通信矩阵: {len(comm_events)} 个事件")

        # 初始化矩阵
        self._matrix = CommunicationMatrix(
            world_size=self.world_size,
            num_machines=self.num_machines,
            npus_per_machine=self.npus_per_machine,
        )

        # 构建拓扑
        self._topology = self._topology_analyzer.build_topology()

        # 聚合链路数据
        link_data = defaultdict(lambda: {
            "total_size": 0,
            "total_time": 0,
            "op_count": 0,
            "transport_type": None,
            "wait_time": 0,
            "sync_time": 0,
        })

        for event in comm_events:
            src = event.get("src_rank")
            dst = event.get("dst_rank")

            if src is None or dst is None:
                continue

            data_size = event.get("data_size", 0) or 0
            duration = event.get("duration", 0) or 0  # us
            wait_time = event.get("wait_time", 0) or 0
            sync_time = event.get("sync_time", 0) or 0
            transport = event.get("transport_type")

            key = (min(src, dst), max(src, dst))
            link_data[key]["total_size"] += data_size
            link_data[key]["total_time"] += duration
            link_data[key]["op_count"] += 1
            link_data[key]["wait_time"] += wait_time
            link_data[key]["sync_time"] += sync_time

            if transport:
                link_data[key]["transport_type"] = transport

        # 构建 LinkMetrics
        for (src, dst), data in link_data.items():
            # 判断链路类型
            is_inter_node = not self._topology.is_same_machine(src, dst)

            # 确定传输类型
            transport_str = data.get("transport_type")
            if transport_str:
                try:
                    transport_type = TransportType(transport_str.lower())
                except ValueError:
                    transport_type = TransportType.RDMA if is_inter_node else TransportType.HCCS
            else:
                transport_type = TransportType.RDMA if is_inter_node else TransportType.HCCS

            # 获取理论带宽
            theoretical_bw = self._get_theoretical_bandwidth(transport_type)

            metrics = LinkMetrics(
                src_rank=src,
                dst_rank=dst,
                transport_type=transport_type,
                theoretical_bandwidth_gbps=theoretical_bw,
                total_transit_size_mb=data["total_size"] / (1024 * 1024),
                total_transit_time_ms=data["total_time"] / 1000,
                op_count=data["op_count"],
                wait_time_ms=data["wait_time"] / 1000,
                sync_time_ms=data["sync_time"] / 1000,
            )

            self._matrix.set_link(src, dst, metrics)

        # 计算派生指标
        self._compute_derived_metrics()

        # 检测慢链路和瓶颈
        self._detect_anomalies()

        # 计算摘要
        self._matrix.compute_summary()

        return self._matrix

    def _get_theoretical_bandwidth(self, transport_type: TransportType) -> float:
        """获取传输类型的理论带宽"""
        if transport_type == TransportType.HCCS:
            return self.hccs_bandwidth
        elif transport_type == TransportType.RDMA:
            return self.rdma_bandwidth
        elif transport_type == TransportType.ROCE:
            return self.roce_bandwidth
        else:
            return self.rdma_bandwidth  # 默认使用 RDMA 带宽

    def _read_matrix_data(
        self,
        db_path: str,
        step_ids: Optional[List[int]] = None,
    ) -> pd.DataFrame:
        """读取通信矩阵数据"""
        conn = sqlite3.connect(db_path)

        try:
            # 检查表是否存在
            cursor = conn.cursor()
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='CommAnalyzerMatrix'"
            )
            if cursor.fetchone() is None:
                logger.warning(f"CommAnalyzerMatrix 表不存在: {db_path}")
                return pd.DataFrame()

            # 构建查询
            query = """
                SELECT
                    step, rank_id, hccl_op_name, group_name,
                    src_rank, dst_rank, transport_type,
                    transit_size, transit_time, bandwidth
                FROM CommAnalyzerMatrix
            """

            if step_ids:
                placeholders = ",".join("?" * len(step_ids))
                query += f" WHERE step IN ({placeholders})"
                df = pd.read_sql_query(query, conn, params=step_ids)
            else:
                df = pd.read_sql_query(query, conn)

            logger.debug(f"读取到 {len(df)} 条矩阵记录")
            return df

        finally:
            conn.close()

    def _read_bandwidth_data(
        self,
        db_path: str,
        step_ids: Optional[List[int]] = None,
    ) -> pd.DataFrame:
        """读取带宽分析数据"""
        conn = sqlite3.connect(db_path)

        try:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='CommAnalyzerBandwidth'"
            )
            if cursor.fetchone() is None:
                return pd.DataFrame()

            query = """
                SELECT
                    step, rank_id, hccl_op_name, group_name,
                    transport_type, transit_size, transit_time, bandwidth,
                    large_packet_ratio, package_size, count, total_duration
                FROM CommAnalyzerBandwidth
            """

            if step_ids:
                placeholders = ",".join("?" * len(step_ids))
                query += f" WHERE step IN ({placeholders})"
                df = pd.read_sql_query(query, conn, params=step_ids)
            else:
                df = pd.read_sql_query(query, conn)

            return df

        finally:
            conn.close()

    def _read_time_data(
        self,
        db_path: str,
        step_ids: Optional[List[int]] = None,
    ) -> pd.DataFrame:
        """读取时间分析数据"""
        conn = sqlite3.connect(db_path)

        try:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='CommAnalyzerTime'"
            )
            if cursor.fetchone() is None:
                return pd.DataFrame()

            query = """
                SELECT
                    step, rank_id, hccl_op_name, group_name,
                    elapse_time, transit_time, wait_time,
                    synchronization_time, idle_time
                FROM CommAnalyzerTime
            """

            if step_ids:
                placeholders = ",".join("?" * len(step_ids))
                query += f" WHERE step IN ({placeholders})"
                df = pd.read_sql_query(query, conn, params=step_ids)
            else:
                df = pd.read_sql_query(query, conn)

            return df

        finally:
            conn.close()

    def _process_matrix_data(self, df: pd.DataFrame):
        """处理矩阵数据"""
        if df.empty:
            return

        # 按链路聚合
        grouped = df.groupby(['src_rank', 'dst_rank'])

        for (src, dst), group in grouped:
            src, dst = int(src), int(dst)

            # 判断链路类型
            is_inter_node = not self._topology.is_same_machine(src, dst)

            # 确定传输类型
            transport_type = TransportType.RDMA if is_inter_node else TransportType.HCCS
            if 'transport_type' in group.columns:
                transport_str = group['transport_type'].iloc[0]
                if transport_str:
                    try:
                        transport_type = TransportType(transport_str.lower())
                    except ValueError:
                        pass

            theoretical_bw = self._get_theoretical_bandwidth(transport_type)

            # 计算聚合指标
            total_size_mb = group['transit_size'].sum() / (1024 * 1024)
            total_time_ms = group['transit_time'].sum()

            metrics = LinkMetrics(
                src_rank=src,
                dst_rank=dst,
                transport_type=transport_type,
                theoretical_bandwidth_gbps=theoretical_bw,
                total_transit_size_mb=total_size_mb,
                total_transit_time_ms=total_time_ms,
                op_count=len(group),
            )

            # 设置峰值带宽
            if 'bandwidth' in group.columns:
                metrics.peak_bandwidth_gbps = group['bandwidth'].max()

            self._matrix.set_link(src, dst, metrics)

    def _process_bandwidth_data(self, df: pd.DataFrame):
        """处理带宽数据"""
        # 带宽数据主要用于补充分析
        pass

    def _process_time_data(self, df: pd.DataFrame):
        """处理时间数据"""
        if df.empty:
            return

        # 按操作聚合时间
        grouped = df.groupby(['hccl_op_name', 'group_name'])

        for (op_name, group_name), group in grouped:
            op_stats = CommOpStatistics(
                op_name=op_name,
                op_type=CommOpType.from_op_name(op_name),
                group_name=group_name,
                total_time_ms=group['elapse_time'].sum(),
                transit_time_ms=group['transit_time'].sum(),
                wait_time_ms=group['wait_time'].sum(),
                sync_time_ms=group['synchronization_time'].sum(),
                idle_time_ms=group['idle_time'].sum(),
                call_count=len(group),
            )
            op_stats.compute_ratios()

    def _compute_derived_metrics(self):
        """计算派生指标"""
        if not self._matrix.link_metrics:
            return

        # 计算每条链路的带宽利用率
        for metrics in self._matrix.link_metrics.values():
            metrics.compute_derived_metrics()

    def _detect_anomalies(self):
        """检测慢链路和瓶颈链路"""
        if not self._matrix.link_metrics:
            return

        # 分别计算节点内和节点间的平均带宽和标准差
        intra_bws = []
        inter_bws = []

        for metrics in self._matrix.link_metrics.values():
            is_inter = metrics.transport_type in (TransportType.RDMA, TransportType.ROCE)
            if is_inter:
                inter_bws.append(metrics.achieved_bandwidth_gbps)
            else:
                intra_bws.append(metrics.achieved_bandwidth_gbps)

        intra_avg = sum(intra_bws) / len(intra_bws) if intra_bws else 0
        inter_avg = sum(inter_bws) / len(inter_bws) if inter_bws else 0

        # 计算标准差
        intra_std = self._calc_std(intra_bws, intra_avg)
        inter_std = self._calc_std(inter_bws, inter_avg)

        # 检测慢链路和瓶颈
        for metrics in self._matrix.link_metrics.values():
            is_inter = metrics.transport_type in (TransportType.RDMA, TransportType.ROCE)
            avg_bw = inter_avg if is_inter else intra_avg
            std_bw = inter_std if is_inter else intra_std

            # 计算异常评分
            metrics.compute_anomaly_score(avg_bw, std_bw)

            # 慢链路检测
            if metrics.achieved_bandwidth_gbps < avg_bw * self.SLOW_LINK_BANDWIDTH_THRESHOLD:
                metrics.is_slow_link = True
                metrics.anomaly_type = "low_bandwidth"
                self._matrix.slow_links.append(metrics)

            # 瓶颈链路检测
            if metrics.bandwidth_utilization > self.BOTTLENECK_UTIL_THRESHOLD:
                metrics.is_bottleneck = True
                if not metrics.anomaly_type:
                    metrics.anomaly_type = "high_utilization"
                self._matrix.bottleneck_links.append(metrics)

        # 按异常评分排序慢链路
        self._matrix.slow_links.sort(key=lambda m: m.anomaly_score, reverse=True)

        # 按利用率排序瓶颈链路
        self._matrix.bottleneck_links.sort(
            key=lambda m: m.bandwidth_utilization, reverse=True
        )

    def _calc_std(self, values: List[float], mean: float) -> float:
        """计算标准差"""
        if len(values) < 2:
            return 0.0
        variance = sum((v - mean) ** 2 for v in values) / len(values)
        return math.sqrt(variance)

    def get_communication_matrix(self) -> Optional[CommunicationMatrix]:
        """获取通信矩阵"""
        return self._matrix

    def get_link_report(self) -> List[Dict[str, Any]]:
        """获取链路报告"""
        if not self._matrix:
            return []
        return [m.to_dict() for m in self._matrix.link_metrics.values()]

    def get_top_comm_pairs(self, top_n: int = 10) -> List[Dict[str, Any]]:
        """获取通信量最大的 rank 对"""
        if not self._matrix:
            return []
        return [m.to_dict() for m in self._matrix.top_comm_pairs[:top_n]]

    def generate_recommendations(self) -> List[str]:
        """生成优化建议"""
        recommendations = []

        if not self._matrix:
            return recommendations

        # 慢链路建议
        if self._matrix.slow_links:
            slow_link = self._matrix.slow_links[0]
            recommendations.append(
                f"检测到慢链路 (Rank {slow_link.src_rank} ↔ {slow_link.dst_rank})，"
                f"建议检查网络连接和 HCCS/RDMA 配置"
            )

        # 瓶颈链路建议
        if self._matrix.bottleneck_links:
            recommendations.append(
                f"检测到 {len(self._matrix.bottleneck_links)} 条高负载链路，"
                f"建议优化通信拓扑或增加通信重叠"
            )

        # 节点间通信比例建议
        if self._matrix.inter_node_ratio > 0.5:
            recommendations.append(
                f"节点间通信比例较高 ({self._matrix.inter_node_ratio:.1%})，"
                f"建议优化模型并行策略以减少跨节点通信"
            )

        # 带宽利用率建议
        avg_util = sum(
            m.bandwidth_utilization for m in self._matrix.link_metrics.values()
        ) / len(self._matrix.link_metrics) if self._matrix.link_metrics else 0

        if avg_util < 0.5:
            recommendations.append(
                f"平均带宽利用率较低 ({avg_util:.1%})，"
                f"建议检查通信模式或增加通信数据量"
            )

        return recommendations


# ============================================================================
# 便捷函数
# ============================================================================

def analyze_communication_matrix(
    db_path: str,
    world_size: int,
    npus_per_machine: int = 8,
) -> CommunicationMatrix:
    """
    分析通信矩阵

    Args:
        db_path: profiling DB 文件路径
        world_size: 总进程数
        npus_per_machine: 每机器 NPU 数

    Returns:
        CommunicationMatrix
    """
    analyzer = CommunicationMatrixAnalyzer(
        world_size=world_size,
        npus_per_machine=npus_per_machine,
    )
    return analyzer.analyze_from_db(db_path)