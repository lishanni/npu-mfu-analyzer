"""
Topology Analyzer - 拓扑分析器

分析多卡集群的物理拓扑结构，识别节点内/节点间通信路径和带宽瓶颈
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Any
from enum import Enum
import logging
import math

logger = logging.getLogger(__name__)


class LinkType(Enum):
    """链路类型"""
    HCCS = "hccs"        # 节点内高速互联
    RDMA = "rdma"        # 节点间 RDMA/RoCE
    PCIE = "pcie"        # PCIe 连接
    NVLINK = "nvlink"    # NVLink（兼容 NVIDIA）
    UNKNOWN = "unknown"


@dataclass
class NPUNode:
    """NPU 节点信息"""
    rank: int
    device_id: int
    machine_id: int
    # 节点内位置（如在 8 卡机器中的位置 0-7）
    local_rank: int = 0
    # 芯片信息
    chip_name: Optional[str] = None
    aicore_count: Optional[int] = None
    
    def __hash__(self):
        return hash((self.rank, self.device_id, self.machine_id))


@dataclass
class TopologyLink:
    """拓扑链路"""
    src_rank: int
    dst_rank: int
    link_type: LinkType
    # 理论带宽 (GB/s)
    theoretical_bandwidth: float = 0.0
    # 实测带宽 (GB/s)
    achieved_bandwidth: float = 0.0
    # 链路延迟 (us)
    latency_us: float = 0.0
    # 是否跨节点
    is_inter_node: bool = False


@dataclass
class TopologyInfo:
    """拓扑信息"""
    # 总节点数
    world_size: int
    # 机器数
    num_machines: int
    # 每机器 NPU 数
    npus_per_machine: int
    
    # NPU 节点列表
    nodes: List[NPUNode] = field(default_factory=list)
    # 链路列表
    links: List[TopologyLink] = field(default_factory=list)
    
    # 节点内链路带宽 (GB/s)
    intra_node_bandwidth: float = 0.0
    # 节点间链路带宽 (GB/s)
    inter_node_bandwidth: float = 0.0
    
    # rank 到机器的映射
    rank_to_machine: Dict[int, int] = field(default_factory=dict)
    # 机器到 rank 列表的映射
    machine_to_ranks: Dict[int, List[int]] = field(default_factory=dict)
    
    def get_link(self, src_rank: int, dst_rank: int) -> Optional[TopologyLink]:
        """获取两个 rank 之间的链路"""
        for link in self.links:
            if link.src_rank == src_rank and link.dst_rank == dst_rank:
                return link
        return None
    
    def is_same_machine(self, rank1: int, rank2: int) -> bool:
        """判断两个 rank 是否在同一机器"""
        return self.rank_to_machine.get(rank1) == self.rank_to_machine.get(rank2)
    
    def get_machine_ranks(self, machine_id: int) -> List[int]:
        """获取指定机器上的所有 rank"""
        return self.machine_to_ranks.get(machine_id, [])


@dataclass
class TopologyMetrics:
    """拓扑分析指标"""
    # 节点内带宽利用率
    intra_node_bw_utilization: float = 0.0
    # 节点间带宽利用率
    inter_node_bw_utilization: float = 0.0
    # 带宽瓶颈位置
    bandwidth_bottleneck: Optional[str] = None
    # 瓶颈链路
    bottleneck_links: List[TopologyLink] = field(default_factory=list)
    # 慢链路（带宽低于平均的链路）
    slow_links: List[TopologyLink] = field(default_factory=list)
    # 平均链路带宽
    avg_link_bandwidth: float = 0.0
    # 链路带宽标准差
    link_bandwidth_std: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "intra_node_bw_utilization": self.intra_node_bw_utilization,
            "inter_node_bw_utilization": self.inter_node_bw_utilization,
            "bandwidth_bottleneck": self.bandwidth_bottleneck,
            "bottleneck_links_count": len(self.bottleneck_links),
            "slow_links_count": len(self.slow_links),
            "avg_link_bandwidth": self.avg_link_bandwidth,
            "link_bandwidth_std": self.link_bandwidth_std,
        }
    
    def to_prompt_text(self) -> str:
        """转换为 LLM Prompt 格式"""
        lines = [
            "## 拓扑分析结果",
            "",
            "### 带宽利用率",
            f"- **节点内 (HCCS) 带宽利用率**: {self.intra_node_bw_utilization:.1%}",
            f"- **节点间 (RDMA) 带宽利用率**: {self.inter_node_bw_utilization:.1%}",
            "",
        ]
        
        if self.bandwidth_bottleneck:
            lines.extend([
                "### 瓶颈分析",
                f"- **主要瓶颈**: {self.bandwidth_bottleneck}",
            ])
            
            if self.bottleneck_links:
                lines.append(f"- **瓶颈链路数**: {len(self.bottleneck_links)}")
        
        if self.slow_links:
            lines.extend([
                "",
                "### 慢链路",
                f"检测到 {len(self.slow_links)} 条慢链路（带宽低于平均值）:",
            ])
            for link in self.slow_links[:5]:  # 只显示前 5 条
                lines.append(
                    f"  - Rank {link.src_rank} -> {link.dst_rank}: "
                    f"{link.achieved_bandwidth:.2f} GB/s "
                    f"({'跨节点' if link.is_inter_node else '节点内'})"
                )
        
        lines.extend([
            "",
            "### 统计信息",
            f"- **平均链路带宽**: {self.avg_link_bandwidth:.2f} GB/s",
            f"- **带宽标准差**: {self.link_bandwidth_std:.2f} GB/s",
        ])
        
        return "\n".join(lines)


class TopologyAnalyzer:
    """
    拓扑分析器
    
    分析多卡集群的物理拓扑结构，包括：
    - 自动识别物理拓扑（节点内 HCCS vs 节点间 RDMA）
    - 构建 rank 到物理位置的映射
    - 分析通信路径的带宽瓶颈
    """
    
    # 默认带宽配置 (GB/s)
    DEFAULT_HCCS_BANDWIDTH = 56.0  # HCCS per link
    DEFAULT_RDMA_BANDWIDTH = 25.0  # 200Gbps RDMA
    DEFAULT_PCIE_BANDWIDTH = 32.0  # PCIe Gen5 x16
    
    def __init__(
        self,
        world_size: int,
        npus_per_machine: int = 8,
        hccs_bandwidth: float = None,
        rdma_bandwidth: float = None,
    ):
        """
        Args:
            world_size: 总进程数
            npus_per_machine: 每机器 NPU 数，默认 8
            hccs_bandwidth: HCCS 带宽 (GB/s)
            rdma_bandwidth: RDMA 带宽 (GB/s)
        """
        self.world_size = world_size
        self.npus_per_machine = npus_per_machine
        self.num_machines = math.ceil(world_size / npus_per_machine)
        
        self.hccs_bandwidth = hccs_bandwidth or self.DEFAULT_HCCS_BANDWIDTH
        self.rdma_bandwidth = rdma_bandwidth or self.DEFAULT_RDMA_BANDWIDTH
        
        self._topology: Optional[TopologyInfo] = None
    
    def build_topology(
        self,
        device_info: Optional[List[Dict[str, Any]]] = None,
    ) -> TopologyInfo:
        """
        构建拓扑信息
        
        Args:
            device_info: 设备信息列表，每个元素包含 rank、device_id、chip_name 等
            
        Returns:
            TopologyInfo
        """
        nodes = []
        rank_to_machine = {}
        machine_to_ranks = {}
        
        # 构建节点信息
        for rank in range(self.world_size):
            machine_id = rank // self.npus_per_machine
            local_rank = rank % self.npus_per_machine
            
            # 从 device_info 获取详细信息
            chip_name = None
            aicore_count = None
            if device_info:
                for info in device_info:
                    if info.get("rank") == rank:
                        chip_name = info.get("chip_name")
                        aicore_count = info.get("aicore_count")
                        break
            
            node = NPUNode(
                rank=rank,
                device_id=local_rank,
                machine_id=machine_id,
                local_rank=local_rank,
                chip_name=chip_name,
                aicore_count=aicore_count,
            )
            nodes.append(node)
            
            rank_to_machine[rank] = machine_id
            if machine_id not in machine_to_ranks:
                machine_to_ranks[machine_id] = []
            machine_to_ranks[machine_id].append(rank)
        
        # 构建链路信息
        links = self._build_links(rank_to_machine)
        
        self._topology = TopologyInfo(
            world_size=self.world_size,
            num_machines=self.num_machines,
            npus_per_machine=self.npus_per_machine,
            nodes=nodes,
            links=links,
            intra_node_bandwidth=self.hccs_bandwidth,
            inter_node_bandwidth=self.rdma_bandwidth,
            rank_to_machine=rank_to_machine,
            machine_to_ranks=machine_to_ranks,
        )
        
        return self._topology
    
    def _build_links(self, rank_to_machine: Dict[int, int]) -> List[TopologyLink]:
        """构建链路信息"""
        links = []
        
        # 为所有可能的 rank 对建立链路
        for src_rank in range(self.world_size):
            for dst_rank in range(self.world_size):
                if src_rank >= dst_rank:
                    continue
                
                src_machine = rank_to_machine[src_rank]
                dst_machine = rank_to_machine[dst_rank]
                
                is_inter_node = src_machine != dst_machine
                
                if is_inter_node:
                    link_type = LinkType.RDMA
                    bandwidth = self.rdma_bandwidth
                else:
                    link_type = LinkType.HCCS
                    bandwidth = self.hccs_bandwidth
                
                link = TopologyLink(
                    src_rank=src_rank,
                    dst_rank=dst_rank,
                    link_type=link_type,
                    theoretical_bandwidth=bandwidth,
                    is_inter_node=is_inter_node,
                )
                links.append(link)
        
        return links
    
    def analyze(
        self,
        comm_events: List[Dict[str, Any]],
    ) -> TopologyMetrics:
        """
        分析拓扑性能
        
        Args:
            comm_events: 通信事件列表，每个事件包含 src_rank, dst_rank, data_size, duration 等
            
        Returns:
            TopologyMetrics
        """
        if self._topology is None:
            self.build_topology()
        
        # 计算每条链路的实测带宽
        link_bandwidths = self._calculate_link_bandwidths(comm_events)
        
        # 更新链路带宽
        for link in self._topology.links:
            key = (link.src_rank, link.dst_rank)
            if key in link_bandwidths:
                link.achieved_bandwidth = link_bandwidths[key]
        
        # 计算带宽利用率
        intra_bw_util, inter_bw_util = self._calculate_bandwidth_utilization()
        
        # 识别瓶颈和慢链路
        bottleneck, bottleneck_links = self._identify_bottleneck()
        slow_links = self._identify_slow_links()
        
        # 计算统计信息
        achieved_bws = [link.achieved_bandwidth for link in self._topology.links if link.achieved_bandwidth > 0]
        avg_bw = sum(achieved_bws) / len(achieved_bws) if achieved_bws else 0.0
        std_bw = self._calculate_std(achieved_bws, avg_bw)
        
        return TopologyMetrics(
            intra_node_bw_utilization=intra_bw_util,
            inter_node_bw_utilization=inter_bw_util,
            bandwidth_bottleneck=bottleneck,
            bottleneck_links=bottleneck_links,
            slow_links=slow_links,
            avg_link_bandwidth=avg_bw,
            link_bandwidth_std=std_bw,
        )
    
    def _calculate_link_bandwidths(
        self,
        comm_events: List[Dict[str, Any]]
    ) -> Dict[Tuple[int, int], float]:
        """计算每条链路的实测带宽"""
        link_data = {}  # {(src, dst): [(data_size, duration), ...]}
        
        for event in comm_events:
            src_rank = event.get("src_rank")
            dst_rank = event.get("dst_rank")
            data_size = event.get("data_size", 0)  # bytes
            duration = event.get("duration", event.get("dur", 0))  # us
            
            if src_rank is None or dst_rank is None:
                continue
            
            # 确保 src < dst
            if src_rank > dst_rank:
                src_rank, dst_rank = dst_rank, src_rank
            
            key = (src_rank, dst_rank)
            if key not in link_data:
                link_data[key] = []
            
            if duration > 0 and data_size > 0:
                link_data[key].append((data_size, duration))
        
        # 计算平均带宽
        bandwidths = {}
        for key, measurements in link_data.items():
            total_size = sum(m[0] for m in measurements)
            total_time = sum(m[1] for m in measurements)
            if total_time > 0:
                # 带宽 = 数据量 / 时间 (bytes/us = MB/s, 转换为 GB/s)
                bandwidths[key] = (total_size / total_time) / 1000
        
        return bandwidths
    
    def _calculate_bandwidth_utilization(self) -> Tuple[float, float]:
        """计算节点内/节点间带宽利用率"""
        intra_achieved = []
        inter_achieved = []
        
        for link in self._topology.links:
            if link.achieved_bandwidth > 0:
                if link.is_inter_node:
                    inter_achieved.append(link.achieved_bandwidth / link.theoretical_bandwidth)
                else:
                    intra_achieved.append(link.achieved_bandwidth / link.theoretical_bandwidth)
        
        intra_util = sum(intra_achieved) / len(intra_achieved) if intra_achieved else 0.0
        inter_util = sum(inter_achieved) / len(inter_achieved) if inter_achieved else 0.0
        
        return min(intra_util, 1.0), min(inter_util, 1.0)
    
    def _identify_bottleneck(self) -> Tuple[Optional[str], List[TopologyLink]]:
        """识别带宽瓶颈"""
        intra_util, inter_util = self._calculate_bandwidth_utilization()
        
        bottleneck_links = []
        
        # 找出利用率接近 100% 的链路
        for link in self._topology.links:
            if link.achieved_bandwidth > 0:
                util = link.achieved_bandwidth / link.theoretical_bandwidth
                if util > 0.9:  # 利用率超过 90%
                    bottleneck_links.append(link)
        
        # 判断瓶颈类型
        if inter_util > 0.9 and inter_util > intra_util:
            return "节点间 RDMA 带宽饱和", bottleneck_links
        elif intra_util > 0.9:
            return "节点内 HCCS 带宽饱和", bottleneck_links
        elif bottleneck_links:
            return "部分链路带宽饱和", bottleneck_links
        
        return None, []
    
    def _identify_slow_links(self) -> List[TopologyLink]:
        """识别慢链路"""
        slow_links = []
        
        # 分别计算节点内和节点间的平均带宽
        intra_bws = [link.achieved_bandwidth for link in self._topology.links 
                     if not link.is_inter_node and link.achieved_bandwidth > 0]
        inter_bws = [link.achieved_bandwidth for link in self._topology.links 
                     if link.is_inter_node and link.achieved_bandwidth > 0]
        
        intra_avg = sum(intra_bws) / len(intra_bws) if intra_bws else 0
        inter_avg = sum(inter_bws) / len(inter_bws) if inter_bws else 0
        
        # 找出低于平均值 20% 以上的链路
        for link in self._topology.links:
            if link.achieved_bandwidth > 0:
                if link.is_inter_node and inter_avg > 0:
                    if link.achieved_bandwidth < inter_avg * 0.8:
                        slow_links.append(link)
                elif not link.is_inter_node and intra_avg > 0:
                    if link.achieved_bandwidth < intra_avg * 0.8:
                        slow_links.append(link)
        
        return slow_links
    
    def _calculate_std(self, values: List[float], mean: float) -> float:
        """计算标准差"""
        if len(values) < 2:
            return 0.0
        variance = sum((v - mean) ** 2 for v in values) / len(values)
        return math.sqrt(variance)
    
    def get_topology(self) -> Optional[TopologyInfo]:
        """获取拓扑信息"""
        return self._topology
    
    def get_communication_path(self, src_rank: int, dst_rank: int) -> Dict[str, Any]:
        """
        获取两个 rank 之间的通信路径信息
        
        Returns:
            包含路径类型、带宽、跳数等信息的字典
        """
        if self._topology is None:
            self.build_topology()
        
        src_machine = self._topology.rank_to_machine.get(src_rank)
        dst_machine = self._topology.rank_to_machine.get(dst_rank)
        
        is_same_machine = src_machine == dst_machine
        
        link = self._topology.get_link(
            min(src_rank, dst_rank), 
            max(src_rank, dst_rank)
        )
        
        return {
            "src_rank": src_rank,
            "dst_rank": dst_rank,
            "src_machine": src_machine,
            "dst_machine": dst_machine,
            "is_same_machine": is_same_machine,
            "link_type": link.link_type.value if link else "unknown",
            "theoretical_bandwidth": link.theoretical_bandwidth if link else 0,
            "achieved_bandwidth": link.achieved_bandwidth if link else 0,
            "hops": 1 if is_same_machine else 2,  # 简化假设
        }


def analyze_topology_from_loader(loader) -> TopologyMetrics:
    """
    从 ProfilingLoader 分析拓扑
    
    Args:
        loader: ProfilingLoader 实例
        
    Returns:
        TopologyMetrics
    """
    info = loader.detect()
    
    analyzer = TopologyAnalyzer(
        world_size=info.rank_count,
        npus_per_machine=8,  # 默认 8 卡/机器
    )
    
    # 构建拓扑
    analyzer.build_topology()
    
    # 收集通信事件
    comm_events = []
    try:
        overlap_events = loader.get_overlap_events()
        comm_events = overlap_events.get("hccl", [])
    except Exception as e:
        logger.warning(f"Failed to collect comm events: {e}")
    
    return analyzer.analyze(comm_events)
