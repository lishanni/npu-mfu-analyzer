"""
HCCS Ring 拓扑解析器

解析昇腾 NPU 集群的 HCCS (High-speed Chip-to-Chip Scalability) Ring 拓扑结构。

HCCS 是华为昇腾 NPU 的片间互联技术，类似于 NVIDIA 的 NVLink。
在 8 卡机器中，通常有多个 HCCS Ring，跨 Ring 通信效率会降低。

典型拓扑:
- Atlas 800 (8x 910B): 2 个 HCCS Ring，每个 Ring 4 卡
  Ring 0: NPU 0-1-2-3
  Ring 1: NPU 4-5-6-7
  跨 Ring 通信需要通过 PCIe/Host，带宽约为 Ring 内的 1/2
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Any
from enum import Enum
import os
import json
import logging
import re

logger = logging.getLogger(__name__)


class HCCSLinkType(Enum):
    """HCCS 链路类型"""
    INTRA_RING = "intra_ring"      # Ring 内直连
    CROSS_RING = "cross_ring"      # 跨 Ring（通过 Host/PCIe）
    INTER_NODE = "inter_node"      # 跨节点（RDMA）


@dataclass
class HCCSRing:
    """HCCS Ring 结构"""
    ring_id: int
    # Ring 内的 NPU 设备 ID（local rank）
    device_ids: List[int] = field(default_factory=list)
    # 对应的全局 rank（如果已知）
    ranks: List[int] = field(default_factory=list)
    # Ring 内单向带宽 (GB/s)
    bandwidth_gbps: float = 56.0
    # Ring 内延迟 (us)
    latency_us: float = 1.0
    # 机器 ID（多机场景）
    machine_id: int = 0
    
    def __contains__(self, device_id: int) -> bool:
        return device_id in self.device_ids
    
    def __len__(self) -> int:
        return len(self.device_ids)
    
    def get_ring_neighbors(self, device_id: int) -> Tuple[Optional[int], Optional[int]]:
        """
        获取 Ring 内的相邻设备
        
        Returns:
            (prev_device, next_device): 前驱和后继设备 ID
        """
        if device_id not in self.device_ids:
            return None, None
        
        idx = self.device_ids.index(device_id)
        n = len(self.device_ids)
        
        prev_idx = (idx - 1) % n
        next_idx = (idx + 1) % n
        
        return self.device_ids[prev_idx], self.device_ids[next_idx]


@dataclass
class HCCSTopology:
    """完整的 HCCS 拓扑结构"""
    # 所有 Ring
    rings: List[HCCSRing] = field(default_factory=list)
    # 设备到 Ring 的映射
    device_to_ring: Dict[int, int] = field(default_factory=dict)
    # 跨 Ring 带宽 (GB/s)
    cross_ring_bandwidth_gbps: float = 28.0  # 通常是 Ring 内的 1/2
    # 跨 Ring 延迟 (us)
    cross_ring_latency_us: float = 5.0
    # 总设备数
    total_devices: int = 0
    # 机器 ID
    machine_id: int = 0
    # 拓扑检测来源
    detection_source: str = "default"
    
    def get_ring(self, device_id: int) -> Optional[HCCSRing]:
        """获取设备所在的 Ring"""
        ring_id = self.device_to_ring.get(device_id)
        if ring_id is not None and ring_id < len(self.rings):
            return self.rings[ring_id]
        return None
    
    def is_same_ring(self, device1: int, device2: int) -> bool:
        """判断两个设备是否在同一个 Ring"""
        return self.device_to_ring.get(device1) == self.device_to_ring.get(device2)
    
    def get_link_type(self, device1: int, device2: int) -> HCCSLinkType:
        """获取两个设备之间的链路类型"""
        if self.is_same_ring(device1, device2):
            return HCCSLinkType.INTRA_RING
        return HCCSLinkType.CROSS_RING
    
    def get_bandwidth(self, device1: int, device2: int) -> float:
        """获取两个设备之间的带宽 (GB/s)"""
        if self.is_same_ring(device1, device2):
            ring = self.get_ring(device1)
            return ring.bandwidth_gbps if ring else 0.0
        return self.cross_ring_bandwidth_gbps
    
    def get_latency(self, device1: int, device2: int) -> float:
        """获取两个设备之间的延迟 (us)"""
        if self.is_same_ring(device1, device2):
            ring = self.get_ring(device1)
            return ring.latency_us if ring else 0.0
        return self.cross_ring_latency_us
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "rings": [
                {
                    "ring_id": r.ring_id,
                    "device_ids": r.device_ids,
                    "ranks": r.ranks,
                    "bandwidth_gbps": r.bandwidth_gbps,
                }
                for r in self.rings
            ],
            "cross_ring_bandwidth_gbps": self.cross_ring_bandwidth_gbps,
            "total_devices": self.total_devices,
            "detection_source": self.detection_source,
        }


@dataclass
class HCCSCommAnalysis:
    """HCCS 通信分析结果"""
    # 总通信次数
    total_comm_count: int = 0
    # Ring 内通信次数
    intra_ring_count: int = 0
    # 跨 Ring 通信次数
    cross_ring_count: int = 0
    # Ring 内总数据量 (bytes)
    intra_ring_data_bytes: int = 0
    # 跨 Ring 总数据量 (bytes)
    cross_ring_data_bytes: int = 0
    # Ring 内总时间 (us)
    intra_ring_time_us: float = 0.0
    # 跨 Ring 总时间 (us)
    cross_ring_time_us: float = 0.0
    
    @property
    def cross_ring_ratio(self) -> float:
        """跨 Ring 通信比例"""
        if self.total_comm_count == 0:
            return 0.0
        return self.cross_ring_count / self.total_comm_count
    
    @property
    def intra_ring_bandwidth_gbps(self) -> float:
        """Ring 内实测带宽"""
        if self.intra_ring_time_us <= 0:
            return 0.0
        return self.intra_ring_data_bytes / self.intra_ring_time_us / 1000
    
    @property
    def cross_ring_bandwidth_gbps(self) -> float:
        """跨 Ring 实测带宽"""
        if self.cross_ring_time_us <= 0:
            return 0.0
        return self.cross_ring_data_bytes / self.cross_ring_time_us / 1000
    
    def to_prompt_text(self) -> str:
        lines = [
            "## HCCS 通信分析",
            "",
            f"- **总通信次数**: {self.total_comm_count}",
            f"- **Ring 内通信**: {self.intra_ring_count} ({(1-self.cross_ring_ratio)*100:.1f}%)",
            f"- **跨 Ring 通信**: {self.cross_ring_count} ({self.cross_ring_ratio*100:.1f}%)",
            "",
            "### 带宽分析",
            f"- Ring 内实测带宽: {self.intra_ring_bandwidth_gbps:.2f} GB/s",
            f"- 跨 Ring 实测带宽: {self.cross_ring_bandwidth_gbps:.2f} GB/s",
        ]
        
        if self.cross_ring_ratio > 0.3:
            lines.append("")
            lines.append("**警告**: 跨 Ring 通信比例较高，可能影响集合通信效率")
        
        return "\n".join(lines)


class HCCSTopologyParser:
    """
    HCCS 拓扑解析器
    
    支持多种方式获取拓扑:
    1. 从环境变量解析 (HCCL_WORLD_GROUP, NCCL_TOPOLOGY 等)
    2. 从 Profiling 数据解析 (device_*/info.json)
    3. 从预定义的硬件配置推断
    4. 使用默认拓扑（8 卡 2 Ring）
    """
    
    # 预定义的拓扑模板
    TOPOLOGY_TEMPLATES = {
        # Atlas 800 (8x 910B) - 2 个 Ring，每个 4 卡
        "atlas800_8x910b": {
            "rings": [[0, 1, 2, 3], [4, 5, 6, 7]],
            "intra_ring_bw": 56.0,
            "cross_ring_bw": 28.0,
        },
        # Atlas 300T A2 (8x 910B) - 全连接
        "atlas300t_8x910b": {
            "rings": [[0, 1, 2, 3, 4, 5, 6, 7]],
            "intra_ring_bw": 56.0,
            "cross_ring_bw": 56.0,
        },
        # 4 卡配置
        "4card": {
            "rings": [[0, 1, 2, 3]],
            "intra_ring_bw": 56.0,
            "cross_ring_bw": 56.0,
        },
        # 2 卡配置
        "2card": {
            "rings": [[0, 1]],
            "intra_ring_bw": 56.0,
            "cross_ring_bw": 56.0,
        },
    }
    
    def __init__(self, npus_per_machine: int = 8):
        """
        Args:
            npus_per_machine: 每机器 NPU 数量
        """
        self.npus_per_machine = npus_per_machine
    
    def parse(
        self,
        profiling_path: Optional[str] = None,
        env_vars: Optional[Dict[str, str]] = None,
        template_name: Optional[str] = None,
    ) -> HCCSTopology:
        """
        解析 HCCS 拓扑
        
        优先级:
        1. 从 Profiling 数据解析
        2. 从环境变量解析
        3. 使用指定模板
        4. 使用默认拓扑
        
        Args:
            profiling_path: Profiling 数据目录
            env_vars: 环境变量字典（用于测试）
            template_name: 拓扑模板名称
            
        Returns:
            HCCSTopology
        """
        topology = None
        
        # 1. 尝试从 Profiling 数据解析
        if profiling_path:
            topology = self._parse_from_profiling(profiling_path)
            if topology:
                topology.detection_source = "profiling"
                return topology
        
        # 2. 尝试从环境变量解析
        topology = self._parse_from_env(env_vars or os.environ)
        if topology:
            topology.detection_source = "environment"
            return topology
        
        # 3. 使用指定模板
        if template_name and template_name in self.TOPOLOGY_TEMPLATES:
            topology = self._create_from_template(template_name)
            topology.detection_source = f"template:{template_name}"
            return topology
        
        # 4. 使用默认拓扑
        return self._create_default_topology()
    
    def _parse_from_profiling(self, profiling_path: str) -> Optional[HCCSTopology]:
        """从 Profiling 数据解析拓扑"""
        from pathlib import Path
        
        profiling_dir = Path(profiling_path)
        if not profiling_dir.exists():
            return None
        
        # 查找 device_*/info.json* 文件
        device_infos = []
        
        for device_dir in profiling_dir.rglob("device_*"):
            if not device_dir.is_dir():
                continue
            
            for info_file in device_dir.glob("info.json*"):
                try:
                    with open(info_file, "r") as f:
                        data = json.load(f)
                    
                    device_info_list = data.get("DeviceInfo", [])
                    if device_info_list:
                        device_info = device_info_list[0]
                        device_id = int(device_dir.name.split("_")[-1])
                        device_infos.append({
                            "device_id": device_id,
                            "chip_name": device_info.get("soc_name", ""),
                            "aicore_count": device_info.get("ai_core_num", 0),
                        })
                except Exception as e:
                    logger.debug(f"Failed to parse {info_file}: {e}")
        
        if not device_infos:
            return None
        
        # 根据设备数量选择拓扑
        num_devices = len(device_infos)
        
        if num_devices == 8:
            # 默认 8 卡 2 Ring 拓扑
            return self._create_from_template("atlas800_8x910b")
        elif num_devices == 4:
            return self._create_from_template("4card")
        elif num_devices == 2:
            return self._create_from_template("2card")
        else:
            # 自定义拓扑：所有设备在一个 Ring
            topology = HCCSTopology(total_devices=num_devices)
            ring = HCCSRing(
                ring_id=0,
                device_ids=list(range(num_devices)),
                ranks=list(range(num_devices)),
            )
            topology.rings.append(ring)
            for i in range(num_devices):
                topology.device_to_ring[i] = 0
            return topology
    
    def _parse_from_env(self, env_vars: Dict[str, str]) -> Optional[HCCSTopology]:
        """从环境变量解析拓扑"""
        # 检查 HCCL 相关环境变量
        hccl_world_size = env_vars.get("HCCL_WORLD_SIZE", env_vars.get("WORLD_SIZE", ""))
        rank_table_file = env_vars.get("RANK_TABLE_FILE", "")
        
        # 尝试解析 RANK_TABLE_FILE
        if rank_table_file and os.path.exists(rank_table_file):
            return self._parse_rank_table(rank_table_file)
        
        # 根据 WORLD_SIZE 创建默认拓扑
        if hccl_world_size:
            try:
                world_size = int(hccl_world_size)
                local_size = min(world_size, self.npus_per_machine)
                
                if local_size == 8:
                    return self._create_from_template("atlas800_8x910b")
                elif local_size == 4:
                    return self._create_from_template("4card")
                elif local_size == 2:
                    return self._create_from_template("2card")
            except ValueError:
                pass
        
        return None
    
    def _parse_rank_table(self, rank_table_file: str) -> Optional[HCCSTopology]:
        """解析 HCCL rank table 文件"""
        try:
            with open(rank_table_file, "r") as f:
                data = json.load(f)
            
            # 解析 server_list
            server_list = data.get("server_list", [])
            if not server_list:
                return None
            
            topology = HCCSTopology()
            all_device_ids = []
            
            for server in server_list:
                device_list = server.get("device", [])
                for device in device_list:
                    device_id = device.get("device_id")
                    rank_id = device.get("rank_id")
                    if device_id is not None:
                        all_device_ids.append(int(device_id))
            
            # 简单拓扑：根据设备数量分 Ring
            num_devices = len(all_device_ids)
            topology.total_devices = num_devices
            
            if num_devices == 8:
                # 2 Ring
                ring0 = HCCSRing(ring_id=0, device_ids=all_device_ids[:4])
                ring1 = HCCSRing(ring_id=1, device_ids=all_device_ids[4:])
                topology.rings = [ring0, ring1]
                for i, dev_id in enumerate(all_device_ids):
                    topology.device_to_ring[dev_id] = 0 if i < 4 else 1
            else:
                # 单 Ring
                ring = HCCSRing(ring_id=0, device_ids=all_device_ids)
                topology.rings = [ring]
                for dev_id in all_device_ids:
                    topology.device_to_ring[dev_id] = 0
            
            return topology
            
        except Exception as e:
            logger.warning(f"Failed to parse rank table: {e}")
            return None
    
    def _create_from_template(self, template_name: str) -> HCCSTopology:
        """从模板创建拓扑"""
        template = self.TOPOLOGY_TEMPLATES.get(template_name, self.TOPOLOGY_TEMPLATES["atlas800_8x910b"])
        
        topology = HCCSTopology()
        
        for ring_id, device_ids in enumerate(template["rings"]):
            ring = HCCSRing(
                ring_id=ring_id,
                device_ids=list(device_ids),
                ranks=list(device_ids),
                bandwidth_gbps=template["intra_ring_bw"],
            )
            topology.rings.append(ring)
            
            for dev_id in device_ids:
                topology.device_to_ring[dev_id] = ring_id
        
        all_devices = [d for ring in template["rings"] for d in ring]
        topology.total_devices = len(all_devices)
        topology.cross_ring_bandwidth_gbps = template["cross_ring_bw"]
        
        return topology
    
    def _create_default_topology(self) -> HCCSTopology:
        """创建默认拓扑（8 卡 2 Ring）"""
        topology = self._create_from_template("atlas800_8x910b")
        topology.detection_source = "default"
        return topology
    
    def analyze_communication(
        self,
        topology: HCCSTopology,
        comm_events: List[Dict[str, Any]],
    ) -> HCCSCommAnalysis:
        """
        分析通信事件在 HCCS 拓扑上的分布
        
        Args:
            topology: HCCS 拓扑
            comm_events: 通信事件列表，每个事件需包含 src_rank, dst_rank, data_size, duration
            
        Returns:
            HCCSCommAnalysis
        """
        analysis = HCCSCommAnalysis()
        
        for event in comm_events:
            src = event.get("src_rank", event.get("src_device"))
            dst = event.get("dst_rank", event.get("dst_device"))
            data_size = event.get("data_size", 0)
            duration = event.get("duration", event.get("dur", 0))
            
            if src is None or dst is None:
                continue
            
            analysis.total_comm_count += 1
            
            # 判断是否跨 Ring
            if topology.is_same_ring(src, dst):
                analysis.intra_ring_count += 1
                analysis.intra_ring_data_bytes += data_size
                analysis.intra_ring_time_us += duration
            else:
                analysis.cross_ring_count += 1
                analysis.cross_ring_data_bytes += data_size
                analysis.cross_ring_time_us += duration
        
        return analysis


def parse_hccs_topology_from_loader(loader) -> HCCSTopology:
    """
    从 ProfilingLoader 解析 HCCS 拓扑
    
    Args:
        loader: ProfilingLoader 实例
        
    Returns:
        HCCSTopology
    """
    parser = HCCSTopologyParser()
    return parser.parse(profiling_path=loader.base_path)


def analyze_hccs_from_loader(loader) -> Tuple[HCCSTopology, HCCSCommAnalysis]:
    """
    从 ProfilingLoader 分析 HCCS 通信
    
    Args:
        loader: ProfilingLoader 实例
        
    Returns:
        (HCCSTopology, HCCSCommAnalysis)
    """
    parser = HCCSTopologyParser()
    topology = parser.parse(profiling_path=loader.base_path)
    
    # 收集通信事件
    comm_events = []
    try:
        overlap_events = loader.get_overlap_events()
        comm_events = overlap_events.get("hccl", [])
    except Exception as e:
        logger.warning(f"Failed to collect comm events: {e}")
    
    analysis = parser.analyze_communication(topology, comm_events)
    
    return topology, analysis
