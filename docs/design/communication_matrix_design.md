# 通信矩阵深度分析模块设计文档

## 1. 概述

### 1.1 目标

为 npu-mfu-analyzer 增强硬件级通信分析能力，包括：

- **链路性能分析**：HCCS/RDMA 链路级带宽利用率、延迟分析
- **拓扑矩阵构建**：NPU 间通信路径、跨 Ring 开销分析
- **慢链路检测**：带宽异常、延迟异常、丢包检测
- **通信矩阵汇总**：rank 间通信数据量、带宽、时间统计

### 1.2 数据来源

Profiling 数据采集后的通信数据存储在以下表中：

| 表名 | 说明 | 关键字段 |
|------|------|---------|
| `CommAnalyzerMatrix` | 通信矩阵数据 | src_rank, dst_rank, transit_size, transit_time, bandwidth |
| `CommAnalyzerTime` | 通信时间数据 | transit_time, wait_time, synchronization_time, idle_time |
| `CommAnalyzerBandwidth` | 通信带宽数据 | transport_type, transit_size, bandwidth, large_packet_ratio |
| `COMMUNICATION_OP` | 通信操作详情 | opName, opType, groupName, startNS, endNS |

---

## 2. 数据结构设计

### 2.1 通信矩阵数据结构

```python
# src/analyzers/communication_matrix_analyzer.py

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum


class TransportType(Enum):
    """传输类型"""
    HCCS = "hccs"           # 节点内高速互联
    RDMA = "rdma"           # 节点间 RDMA
    ROCE = "roce"           # RoCE v2
    PCIE = "pcie"           # PCIe
    UNKNOWN = "unknown"


class CommOpType(Enum):
    """通信操作类型"""
    # 集合通信
    ALL_REDUCE = "hcom_all_reduce"
    ALL_GATHER = "hcom_all_gather"
    REDUCE_SCATTER = "hcom_reduce_scatter"
    BROADCAST = "hcom_broadcast"
    ALL_TO_ALL = "hcom_all_to_all"
    # P2P 通信
    SEND = "hcom_send"
    RECEIVE = "hcom_receive"
    BATCH_SEND_RECV = "hcom_batchsendrecv"


@dataclass
class LinkMetrics:
    """链路性能指标"""
    src_rank: int
    dst_rank: int
    transport_type: TransportType

    # 带宽指标
    theoretical_bandwidth_gbps: float = 0.0    # 理论带宽 (GB/s)
    achieved_bandwidth_gbps: float = 0.0       # 实测带宽 (GB/s)
    bandwidth_utilization: float = 0.0          # 带宽利用率 (0-1)

    # 数据量指标
    total_transit_size_mb: float = 0.0         # 总传输数据量 (MB)
    total_transit_time_ms: float = 0.0         # 总传输时间 (ms)

    # 延迟指标
    avg_latency_us: float = 0.0                # 平均延迟 (us)
    min_latency_us: float = float('inf')       # 最小延迟
    max_latency_us: float = 0.0                # 最大延迟

    # 操作统计
    op_count: int = 0                          # 操作次数
    large_packet_ratio: float = 0.0            # 大包比例

    # 异常标记
    is_slow_link: bool = False                 # 是否慢链路
    anomaly_type: Optional[str] = None         # 异常类型

    def compute_derived_metrics(self):
        """计算派生指标"""
        if self.theoretical_bandwidth_gbps > 0:
            self.bandwidth_utilization = self.achieved_bandwidth_gbps / self.theoretical_bandwidth_gbps

        if self.total_transit_time_ms > 0:
            self.achieved_bandwidth_gbps = (
                self.total_transit_size_mb / self.total_transit_time_ms
            )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "src_rank": self.src_rank,
            "dst_rank": self.dst_rank,
            "transport_type": self.transport_type.value,
            "theoretical_bandwidth_gbps": self.theoretical_bandwidth_gbps,
            "achieved_bandwidth_gbps": self.achieved_bandwidth_gbps,
            "bandwidth_utilization": f"{self.bandwidth_utilization:.1%}",
            "total_transit_size_mb": self.total_transit_size_mb,
            "total_transit_time_ms": self.total_transit_time_ms,
            "avg_latency_us": self.avg_latency_us,
            "op_count": self.op_count,
            "is_slow_link": self.is_slow_link,
            "anomaly_type": self.anomaly_type,
        }


@dataclass
class CommunicationMatrix:
    """通信矩阵"""
    world_size: int

    # 链路指标矩阵: link_metrics[(src_rank, dst_rank)] = LinkMetrics
    link_metrics: Dict[tuple, LinkMetrics] = field(default_factory=dict)

    # 拓扑信息
    num_machines: int = 1
    npus_per_machine: int = 8

    # 统计摘要
    total_comm_data_mb: float = 0.0            # 总通信数据量
    total_comm_time_ms: float = 0.0            # 总通信时间
    avg_bandwidth_gbps: float = 0.0            # 平均带宽
    intra_node_ratio: float = 0.0              # 节点内通信比例
    inter_node_ratio: float = 0.0              # 节点间通信比例

    # 异常链路
    slow_links: List[LinkMetrics] = field(default_factory=list)
    bottleneck_links: List[LinkMetrics] = field(default_factory=list)

    def get_link(self, src_rank: int, dst_rank: int) -> Optional[LinkMetrics]:
        """获取链路指标"""
        key = (min(src_rank, dst_rank), max(src_rank, dst_rank))
        return self.link_metrics.get(key)

    def get_matrix_2d(self) -> List[List[float]]:
        """获取 2D 带宽矩阵 (用于可视化)"""
        matrix = [[0.0] * self.world_size for _ in range(self.world_size)]
        for (src, dst), metrics in self.link_metrics.items():
            matrix[src][dst] = metrics.achieved_bandwidth_gbps
            matrix[dst][src] = metrics.achieved_bandwidth_gbps
        return matrix

    def to_prompt_text(self) -> str:
        """转换为 LLM Prompt 格式"""
        lines = [
            "## 通信矩阵分析结果",
            "",
            "### 概览",
            f"- World Size: {self.world_size}",
            f"- 机器数: {self.num_machines}",
            f"- 总通信数据量: {self.total_comm_data_mb:.2f} MB",
            f"- 总通信时间: {self.total_comm_time_ms:.2f} ms",
            f"- 平均带宽: {self.avg_bandwidth_gbps:.2f} GB/s",
            f"- 节点内通信比例: {self.intra_node_ratio:.1%}",
            "",
        ]

        if self.slow_links:
            lines.extend([
                "### 慢链路检测",
                f"检测到 {len(self.slow_links)} 条慢链路：",
            ])
            for link in self.slow_links[:5]:
                lines.append(
                    f"  - Rank {link.src_rank} ↔ {link.dst_rank}: "
                    f"{link.achieved_bandwidth_gbps:.2f} GB/s "
                    f"({link.transport_type.value})"
                )

        if self.bottleneck_links:
            lines.extend([
                "",
                "### 瓶颈链路",
                f"检测到 {len(self.bottleneck_links)} 条高负载链路：",
            ])
            for link in self.bottleneck_links[:5]:
                lines.append(
                    f"  - Rank {link.src_rank} ↔ {link.dst_rank}: "
                    f"利用率 {link.bandwidth_utilization:.1%}"
                )

        return "\n".join(lines)


@dataclass
class CommOpStatistics:
    """通信操作统计"""
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

    # 调用统计
    call_count: int = 0                        # 调用次数
    participating_ranks: List[int] = field(default_factory=list)

    # 效率指标
    wait_ratio: float = 0.0                    # 等待时间比例
    sync_ratio: float = 0.0                    # 同步时间比例

    def compute_ratios(self):
        """计算时间比例"""
        total = self.transit_time_ms + self.wait_time_ms
        if total > 0:
            self.wait_ratio = self.wait_time_ms / total

        total = self.transit_time_ms + self.sync_time_ms
        if total > 0:
            self.sync_ratio = self.sync_time_ms / total
```

### 2.2 通信矩阵分析器

```python
# src/analyzers/communication_matrix_analyzer.py (续)

import logging
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict
import pandas as pd

from src.topology.topology_analyzer import TopologyAnalyzer, TopologyInfo, LinkType

logger = logging.getLogger(__name__)


class CommunicationMatrixAnalyzer:
    """
    通信矩阵深度分析器

    功能：
    1. 解析通信矩阵数据
    2. 构建链路级性能指标
    3. 检测慢链路和瓶颈链路
    4. 分析通信效率
    """

    # 慢链路检测阈值
    SLOW_LINK_BANDWIDTH_THRESHOLD = 0.7    # 低于平均值 70% 为慢链路
    BOTTLENECK_UTIL_THRESHOLD = 0.9        # 利用率超过 90% 为瓶颈

    def __init__(
        self,
        world_size: int,
        npus_per_machine: int = 8,
        hccs_bandwidth: float = 56.0,      # GB/s
        rdma_bandwidth: float = 25.0,      # GB/s
    ):
        self.world_size = world_size
        self.npus_per_machine = npus_per_machine
        self.num_machines = (world_size + npus_per_machine - 1) // npus_per_machine

        self.hccs_bandwidth = hccs_bandwidth
        self.rdma_bandwidth = rdma_bandwidth

        self._topology_analyzer = TopologyAnalyzer(
            world_size=world_size,
            npus_per_machine=npus_per_machine,
            hccs_bandwidth=hccs_bandwidth,
            rdma_bandwidth=rdma_bandwidth,
        )
        self._topology: Optional[TopologyInfo] = None
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

        # 处理矩阵数据
        self._process_matrix_data(matrix_data)

        # 读取带宽数据
        bandwidth_data = self._read_bandwidth_data(db_path, step_ids)
        self._process_bandwidth_data(bandwidth_data)

        # 计算派生指标
        self._compute_derived_metrics()

        # 检测慢链路和瓶颈
        self._detect_anomalies()

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
        """
        self._matrix = CommunicationMatrix(
            world_size=self.world_size,
            num_machines=self.num_machines,
            npus_per_machine=self.npus_per_machine,
        )

        self._topology = self._topology_analyzer.build_topology()

        # 聚合链路数据
        link_data = defaultdict(lambda: {
            "total_size": 0,
            "total_time": 0,
            "op_count": 0,
        })

        for event in comm_events:
            src = event.get("src_rank")
            dst = event.get("dst_rank")
            data_size = event.get("data_size", 0)
            duration = event.get("duration", 0)  # us

            if src is None or dst is None:
                continue

            key = (min(src, dst), max(src, dst))
            link_data[key]["total_size"] += data_size
            link_data[key]["total_time"] += duration
            link_data[key]["op_count"] += 1

        # 构建 LinkMetrics
        for (src, dst), data in link_data.items():
            is_inter_node = not self._topology.is_same_machine(src, dst)
            link_type = LinkType.RDMA if is_inter_node else LinkType.HCCS
            theoretical_bw = self.rdma_bandwidth if is_inter_node else self.hccs_bandwidth

            metrics = LinkMetrics(
                src_rank=src,
                dst_rank=dst,
                transport_type=TransportType(link_type.value),
                theoretical_bandwidth_gbps=theoretical_bw,
                total_transit_size_mb=data["total_size"] / (1024 * 1024),
                total_transit_time_ms=data["total_time"] / 1000,
                op_count=data["op_count"],
            )

            # 计算带宽
            if metrics.total_transit_time_ms > 0:
                metrics.achieved_bandwidth_gbps = (
                    metrics.total_transit_size_mb / metrics.total_transit_time_ms
                )
                metrics.bandwidth_utilization = (
                    metrics.achieved_bandwidth_gbps / metrics.theoretical_bandwidth_gbps
                )

            self._matrix.link_metrics[(src, dst)] = metrics

        self._compute_derived_metrics()
        self._detect_anomalies()

        return self._matrix

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
                logger.warning(f"CommAnalyzerMatrix table not found in {db_path}")
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

    def _process_matrix_data(self, df: pd.DataFrame):
        """处理矩阵数据"""
        if df.empty:
            return

        # 按链路聚合
        grouped = df.groupby(['src_rank', 'dst_rank'])

        for (src, dst), group in grouped:
            # 判断链路类型
            is_inter_node = not self._topology.is_same_machine(src, dst)
            link_type = LinkType.RDMA if is_inter_node else LinkType.HCCS
            theoretical_bw = self.rdma_bandwidth if is_inter_node else self.hccs_bandwidth

            # 计算聚合指标
            total_size_mb = group['transit_size'].sum() / (1024 * 1024)  # 假设 transit_size 单位是 bytes
            total_time_ms = group['transit_time'].sum()  # 假设单位是 ms

            metrics = LinkMetrics(
                src_rank=int(src),
                dst_rank=int(dst),
                transport_type=TransportType(link_type.value),
                theoretical_bandwidth_gbps=theoretical_bw,
                total_transit_size_mb=total_size_mb,
                total_transit_time_ms=total_time_ms,
                op_count=len(group),
                large_packet_ratio=group.get('large_packet_ratio', pd.Series([0])).mean(),
            )

            self._matrix.link_metrics[(int(src), int(dst))] = metrics

    def _process_bandwidth_data(self, df: pd.DataFrame):
        """处理带宽数据"""
        if df.empty:
            return

        # 更新链路指标
        for _, row in df.iterrows():
            # 带宽数据可能没有 src_rank/dst_rank，跳过
            pass

    def _compute_derived_metrics(self):
        """计算派生指标"""
        if not self._matrix.link_metrics:
            return

        # 计算每条链路的带宽利用率
        for metrics in self._matrix.link_metrics.values():
            metrics.compute_derived_metrics()

        # 计算总体统计
        total_size = sum(m.total_transit_size_mb for m in self._matrix.link_metrics.values())
        total_time = sum(m.total_transit_time_ms for m in self._matrix.link_metrics.values())

        self._matrix.total_comm_data_mb = total_size
        self._matrix.total_comm_time_ms = total_time

        if total_time > 0:
            self._matrix.avg_bandwidth_gbps = total_size / total_time

        # 计算节点内/节点间通信比例
        intra_size = sum(
            m.total_transit_size_mb for m in self._matrix.link_metrics.values()
            if not self._topology.is_same_machine(m.src_rank, m.dst_rank) == False
        )
        inter_size = total_size - intra_size

        if total_size > 0:
            self._matrix.intra_node_ratio = intra_size / total_size
            self._matrix.inter_node_ratio = inter_size / total_size

    def _detect_anomalies(self):
        """检测慢链路和瓶颈链路"""
        if not self._matrix.link_metrics:
            return

        # 分别计算节点内和节点间的平均带宽
        intra_bws = []
        inter_bws = []

        for metrics in self._matrix.link_metrics.values():
            is_inter = not self._topology.is_same_machine(metrics.src_rank, metrics.dst_rank)
            if is_inter:
                inter_bws.append(metrics.achieved_bandwidth_gbps)
            else:
                intra_bws.append(metrics.achieved_bandwidth_gbps)

        intra_avg = sum(intra_bws) / len(intra_bws) if intra_bws else 0
        inter_avg = sum(inter_bws) / len(inter_bws) if inter_bws else 0

        # 检测慢链路和瓶颈
        for metrics in self._matrix.link_metrics.values():
            is_inter = not self._topology.is_same_machine(metrics.src_rank, metrics.dst_rank)
            avg_bw = inter_avg if is_inter else intra_avg

            # 慢链路：带宽低于平均值 70%
            if metrics.achieved_bandwidth_gbps < avg_bw * self.SLOW_LINK_BANDWIDTH_THRESHOLD:
                metrics.is_slow_link = True
                metrics.anomaly_type = "low_bandwidth"
                self._matrix.slow_links.append(metrics)

            # 瓶颈链路：利用率超过 90%
            if metrics.bandwidth_utilization > self.BOTTLENECK_UTIL_THRESHOLD:
                self._matrix.bottleneck_links.append(metrics)

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

        sorted_links = sorted(
            self._matrix.link_metrics.values(),
            key=lambda m: m.total_transit_size_mb,
            reverse=True
        )

        return [m.to_dict() for m in sorted_links[:top_n]]


# 便捷函数
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
```

---

## 3. 集成到 CommunicationAgent

### 3.1 增强 CommunicationAgent

```python
# src/agents/communication_agent.py (增强部分)

from src.analyzers.communication_matrix_analyzer import (
    CommunicationMatrixAnalyzer,
    CommunicationMatrix,
    LinkMetrics,
)

class CommunicationAgent(BaseAgent):
    """增强版 Communication Agent"""

    def __init__(self, llm: LLMInterface, config: Optional[Dict[str, Any]] = None):
        super().__init__(...)
        self._matrix_analyzer: Optional[CommunicationMatrixAnalyzer] = None

    async def analyze(self, data: Dict[str, Any]) -> AnalysisResult:
        """分析通信数据"""
        try:
            analysis_data = self._prepare_analysis_data(data)

            # 新增：通信矩阵分析
            if "db_path" in data or "comm_events" in data:
                matrix_result = await self._analyze_communication_matrix(data)
                if matrix_result:
                    analysis_data.metrics.comm_matrix = matrix_result

            # 原有逻辑...

        except Exception as e:
            ...

    async def _analyze_communication_matrix(
        self,
        data: Dict[str, Any]
    ) -> Optional[CommunicationMatrix]:
        """分析通信矩阵"""
        world_size = data.get("world_size", 1)
        npus_per_machine = data.get("npus_per_machine", 8)

        self._matrix_analyzer = CommunicationMatrixAnalyzer(
            world_size=world_size,
            npus_per_machine=npus_per_machine,
        )

        if "db_path" in data:
            return self._matrix_analyzer.analyze_from_db(data["db_path"])
        elif "comm_events" in data:
            return self._matrix_analyzer.analyze_from_events(data["comm_events"])

        return None
```

---

## 4. CLI 命令

```python
# src/cli/main.py (新增命令)

@app.command("analyze-comm-matrix")
def analyze_comm_matrix(
    profiling_path: str = typer.Argument(..., help="Profiling 数据路径"),
    world_size: int = typer.Option(8, "--world-size", "-w", help="总进程数"),
    npus_per_machine: int = typer.Option(8, "--npus-per-machine", help="每机器 NPU 数"),
    output: str = typer.Option("comm_matrix_report.md", "--output", "-o", help="输出文件"),
):
    """分析通信矩阵"""
    from src.analyzers.communication_matrix_analyzer import analyze_communication_matrix
    from src.data_loader.profiling_loader import ProfilingLoader

    loader = ProfilingLoader(profiling_path)
    info = loader.detect()

    # 查找 DB 文件
    db_path = None
    for pattern in ["**/*.db"]:
        import glob
        dbs = glob.glob(str(Path(profiling_path) / pattern), recursive=True)
        if dbs:
            db_path = dbs[0]
            break

    if not db_path:
        console.print("[red]未找到 DB 文件[/red]")
        raise typer.Exit(1)

    console.print(f"[blue]分析通信矩阵: {db_path}[/blue]")

    matrix = analyze_communication_matrix(
        db_path=db_path,
        world_size=info.rank_count or world_size,
        npus_per_machine=npus_per_machine,
    )

    # 输出报告
    report = matrix.to_prompt_text()

    with open(output, "w") as f:
        f.write(report)

    console.print(f"[green]报告已保存到: {output}[/green]")
```

---

## 5. 测试用例

```python
# tests/unit/test_communication_matrix_analyzer.py

import pytest
from src.analyzers.communication_matrix_analyzer import (
    CommunicationMatrixAnalyzer,
    CommunicationMatrix,
    LinkMetrics,
    TransportType,
)


class TestCommunicationMatrixAnalyzer:

    def test_analyze_from_events(self):
        """测试从事件列表分析"""
        analyzer = CommunicationMatrixAnalyzer(
            world_size=8,
            npus_per_machine=8,
        )

        # 模拟通信事件
        events = [
            {"src_rank": 0, "dst_rank": 1, "data_size": 100e6, "duration": 2000},  # 100MB, 2ms
            {"src_rank": 0, "dst_rank": 2, "data_size": 50e6, "duration": 1000},   # 50MB, 1ms
            {"src_rank": 1, "dst_rank": 3, "data_size": 80e6, "duration": 1500},   # 80MB, 1.5ms
        ]

        matrix = analyzer.analyze_from_events(events)

        assert matrix.world_size == 8
        assert len(matrix.link_metrics) == 3

        # 检查链路指标
        link_0_1 = matrix.get_link(0, 1)
        assert link_0_1 is not None
        assert link_0_1.transport_type == TransportType.HCCS  # 同节点内
        assert link_0_1.total_transit_size_mb > 0

    def test_slow_link_detection(self):
        """测试慢链路检测"""
        analyzer = CommunicationMatrixAnalyzer(
            world_size=8,
            npus_per_machine=8,
        )

        # 创建一条慢链路
        events = [
            {"src_rank": 0, "dst_rank": 1, "data_size": 100e6, "duration": 1000},   # 高带宽
            {"src_rank": 2, "dst_rank": 3, "data_size": 10e6, "duration": 5000},    # 低带宽 (慢链路)
        ]

        matrix = analyzer.analyze_from_events(events)

        # 应该检测到慢链路
        assert len(matrix.slow_links) > 0 or len(matrix.link_metrics) == 2
```

---

## 6. 后续扩展

### 6.1 Phase 2 能力

- **集群通信分析**：跨节点通信拓扑优化
- **集合通信效率**：AllReduce/AllGather 算法效率分析
- **通信掩盖优化**：计算-通信重叠分析

### 6.2 数据可视化

```python
# src/web/visualization.py

def plot_communication_matrix(matrix: CommunicationMatrix):
    """绘制通信矩阵热力图"""
    import matplotlib.pyplot as plt
    import numpy as np

    data = np.array(matrix.get_matrix_2d())

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(data, cmap='YlOrRd')

    ax.set_xticks(range(matrix.world_size))
    ax.set_yticks(range(matrix.world_size))
    ax.set_xlabel('Rank')
    ax.set_ylabel('Rank')
    ax.set_title('Communication Bandwidth Matrix (GB/s)')

    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    return fig
```
