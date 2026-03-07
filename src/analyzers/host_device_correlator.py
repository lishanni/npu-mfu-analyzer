"""
Host-Device 关联分析器

基于 msprof 的 connection_id 机制建立 Host-Device 调用链关联。
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
from pathlib import Path

from src.data_loader.stack_types import (
    HostDeviceChain,
    CorrelationStats,
    SourceAnalysisResult,
    StackFrame,
    HostStack,
    STACK_PATTERNS,
)
from src.data_loader.stack_parser import StackParser

logger = logging.getLogger(__name__)


# 关联相关的事件类型
HOST_EVENT_CATEGORIES = [
    "cpu_op",           # CPU 算子
    "user_annotation",  # 用户标注
    "python_function",  # Python 函数
    "Runtime",          # 运行时
]

DEVICE_EVENT_CATEGORIES = [
    "Kernel",           # GPU/NPU Kernel
    "dequeue",          # 出队事件
    "Device",           # 设备事件
]


@dataclass
class HostEvent:
    """Host 侧事件"""
    name: str
    ts: float
    dur: float
    cat: str
    pid: Any
    tid: Any
    connection_id: Optional[int] = None
    python_stack: List[str] = field(default_factory=list)
    cpp_stack: List[str] = field(default_factory=list)
    host_stack: Optional[HostStack] = None
    raw_event: Optional[Dict[str, Any]] = None


@dataclass
class DeviceEvent:
    """Device 侧事件"""
    name: str
    ts: float
    dur: float
    cat: str
    pid: Any
    tid: Any
    connection_id: Optional[int] = None
    raw_event: Optional[Dict[str, Any]] = None


class HostDeviceCorrelator:
    """
    Host-Device 关联分析器

    基于 connection_id 建立 Host 侧 Torch 算子与 Device 侧 NPU 算子的关联。

    Usage:
        correlator = HostDeviceCorrelator()
        chains = correlator.build_call_chains(trace_events)
        stats = correlator.build_stats(chains)
    """

    def __init__(self, max_stack_depth: int = 10):
        """
        Args:
            max_stack_depth: 堆栈提取深度
        """
        self.max_stack_depth = max_stack_depth
        self.stack_parser = StackParser()

    def build_call_chains(
        self,
        trace_events: List[Dict[str, Any]],
    ) -> List[HostDeviceChain]:
        """
        构建 Host-Device 调用链

        Args:
            trace_events: trace_view.json 中的事件列表

        Returns:
            HostDeviceChain 列表
        """
        # 1. 分离 Host 和 Device 事件
        host_events, device_events = self._separate_events(trace_events)

        logger.info(f"Separated {len(host_events)} host events and {len(device_events)} device events")

        # 2. 按 connection_id 关联
        chains = self._correlate_by_connection_id(host_events, device_events)

        logger.info(f"Built {len(chains)} host-device chains")

        return chains

    def _separate_events(
        self,
        events: List[Dict[str, Any]],
    ) -> Tuple[List[HostEvent], List[DeviceEvent]]:
        """分离 Host 和 Device 事件"""
        host_events = []
        device_events = []

        for event in events:
            cat = event.get("cat", "")
            name = event.get("name", "")
            ts = event.get("ts", 0)
            dur = event.get("dur", 0)

            # 转换时间戳
            try:
                ts = float(ts) if ts else 0.0
                dur = float(dur) if dur else 0.0
            except (ValueError, TypeError):
                continue

            # 提取 connection_id
            args = event.get("args", {})
            connection_id = None
            if isinstance(args, dict):
                # 尝试多种字段名
                connection_id = args.get("connection_id") or args.get("Connection Id") or args.get("correlation_id")

            if cat in HOST_EVENT_CATEGORIES or "aten::" in name:
                # 解析堆栈
                host_stack = self.stack_parser.parse_from_event(event)
                python_stack = []
                cpp_stack = []

                if host_stack:
                    python_stack = [f.function for f in host_stack.python_stack[:self.max_stack_depth]]
                    cpp_stack = [f.function for f in host_stack.cpp_stack[:self.max_stack_depth]]

                host_events.append(HostEvent(
                    name=name,
                    ts=ts,
                    dur=dur,
                    cat=cat,
                    pid=event.get("pid"),
                    tid=event.get("tid"),
                    connection_id=connection_id,
                    python_stack=python_stack,
                    cpp_stack=cpp_stack,
                    host_stack=host_stack,
                    raw_event=event,
                ))

            elif cat in DEVICE_EVENT_CATEGORIES or "Kernel" in cat:
                device_events.append(DeviceEvent(
                    name=name,
                    ts=ts,
                    dur=dur,
                    cat=cat,
                    pid=event.get("pid"),
                    tid=event.get("tid"),
                    connection_id=connection_id,
                    raw_event=event,
                ))

        return host_events, device_events

    def _correlate_by_connection_id(
        self,
        host_events: List[HostEvent],
        device_events: List[DeviceEvent],
    ) -> List[HostDeviceChain]:
        """
        按 connection_id 建立关联
        """
        chains = []

        # 建立 connection_id -> device_event 映射
        device_by_conn: Dict[int, List[DeviceEvent]] = defaultdict(list)
        device_no_conn: List[DeviceEvent] = []

        for de in device_events:
            if de.connection_id is not None:
                device_by_conn[de.connection_id].append(de)
            else:
                device_no_conn.append(de)

        # 建立 connection_id -> host_event 映射
        host_by_conn: Dict[int, List[HostEvent]] = defaultdict(list)
        host_no_conn: List[HostEvent] = []

        for he in host_events:
            if he.connection_id is not None:
                host_by_conn[he.connection_id].append(he)
            else:
                host_no_conn.append(he)

        # 按 connection_id 关联
        matched_conn_ids = set(host_by_conn.keys()) & set(device_by_conn.keys())

        for conn_id in matched_conn_ids:
            host_list = host_by_conn[conn_id]
            device_list = device_by_conn[conn_id]

            # 取最近的 host 和 device 事件
            for he in host_list:
                for de in device_list:
                    # 时间一致性检查：device 事件应在 host 事件之后
                    if de.ts >= he.ts:
                        chain = self._create_chain(he, de, conn_id)
                        chains.append(chain)

        # 对于没有 connection_id 的事件，尝试按时间关联
        if host_no_conn and device_no_conn:
            time_based_chains = self._correlate_by_time(host_no_conn, device_no_conn)
            chains.extend(time_based_chains)

        return chains

    def _correlate_by_time(
        self,
        host_events: List[HostEvent],
        device_events: List[DeviceEvent],
        time_window_us: float = 1000.0,  # 1ms 时间窗口
    ) -> List[HostDeviceChain]:
        """
        按时间戳关联（回退方案）

        当 connection_id 不可用时，使用时间窗口进行关联。
        """
        chains = []

        # 按时间排序
        sorted_hosts = sorted(host_events, key=lambda e: e.ts)
        sorted_devices = sorted(device_events, key=lambda e: e.ts)

        # 使用贪心匹配
        used_devices = set()

        for he in sorted_hosts:
            best_match = None
            best_gap = float('inf')

            for i, de in enumerate(sorted_devices):
                if i in used_devices:
                    continue

                # device 应在 host 之后
                if de.ts < he.ts:
                    continue

                gap = de.ts - he.ts - he.dur  # host 结束到 device 开始的间隔

                if gap >= 0 and gap < time_window_us and gap < best_gap:
                    best_match = i
                    best_gap = gap

            if best_match is not None:
                de = sorted_devices[best_match]
                used_devices.add(best_match)

                chain = self._create_chain(he, de, -1)  # -1 表示无 connection_id
                chains.append(chain)

        return chains

    def _create_chain(
        self,
        host_event: HostEvent,
        device_event: DeviceEvent,
        connection_id: int,
    ) -> HostDeviceChain:
        """创建 HostDeviceChain"""
        # 确定来源类型
        source_type = "unknown"
        if host_event.host_stack:
            source_type = host_event.host_stack.get_source_type()

        # 提取 rank_id
        rank_id = 0
        if host_event.pid:
            try:
                # pid 格式可能是 "Rank 0" 或数字
                pid_str = str(host_event.pid)
                if "Rank" in pid_str:
                    rank_id = int(pid_str.split()[-1])
                else:
                    rank_id = int(pid_str)
            except (ValueError, IndexError):
                pass

        return HostDeviceChain(
            torch_op_name=host_event.name,
            torch_op_ts=host_event.ts,
            torch_op_dur=host_event.dur,
            acl_api_name="",  # ACL API 信息需要额外提取
            acl_api_ts=0.0,
            acl_api_dur=0.0,
            device_op_name=device_event.name,
            device_op_ts=device_event.ts,
            device_op_dur=device_event.dur,
            connection_id=connection_id,
            rank_id=rank_id,
            python_stack=host_event.python_stack,
            cpp_stack=host_event.cpp_stack,
            host_stack=host_event.host_stack,
            source_type=source_type,
        )

    def build_stats(self, chains: List[HostDeviceChain]) -> CorrelationStats:
        """
        构建关联统计

        Args:
            chains: HostDeviceChain 列表

        Returns:
            CorrelationStats
        """
        stats = CorrelationStats()
        stats.total_chains = len(chains)

        # 按来源类型统计
        for chain in chains:
            source_type = chain.source_type
            stats.by_source_type[source_type] = stats.by_source_type.get(source_type, 0) + 1

            # 按算子名称统计
            op_name = chain.device_op_name or chain.torch_op_name
            stats.by_operator[op_name] = stats.by_operator.get(op_name, 0) + 1

            # 分类收集
            if source_type == "eager":
                stats.eager_ops.append(op_name)
            elif source_type == "torch_compile":
                stats.compile_ops.append(op_name)
            elif source_type == "fusion_op":
                stats.fusion_ops.append(op_name)
            elif source_type == "mindspeed":
                stats.mindspeed_ops.append(op_name)

            # 建立算子来源映射
            stats.operator_source_map[op_name] = source_type

        # 去重
        stats.eager_ops = list(set(stats.eager_ops))[:20]
        stats.compile_ops = list(set(stats.compile_ops))[:20]
        stats.fusion_ops = list(set(stats.fusion_ops))[:20]
        stats.mindspeed_ops = list(set(stats.mindspeed_ops))[:20]

        # 提取高频堆栈模式
        pattern_counter = defaultdict(int)
        for chain in chains:
            if chain.python_stack:
                top_frame = chain.python_stack[0] if chain.python_stack else ""
                if top_frame:
                    pattern_counter[top_frame] += 1

        stats.top_stack_patterns = [
            f"{pattern} ({count})"
            for pattern, count in sorted(pattern_counter.items(), key=lambda x: x[1], reverse=True)[:10]
        ]

        return stats

    def analyze(
        self,
        trace_events: List[Dict[str, Any]],
    ) -> SourceAnalysisResult:
        """
        执行完整的 Host-Device 关联分析

        Args:
            trace_events: trace_view.json 中的事件列表

        Returns:
            SourceAnalysisResult
        """
        chains = self.build_call_chains(trace_events)
        stats = self.build_stats(chains)

        # 构建结果
        result = SourceAnalysisResult(
            total_chains=stats.total_chains,
            by_source_type=stats.by_source_type,
            stats=stats,
        )

        # 获取各来源的 Top 算子
        result.top_source_operators = {
            "eager": stats.eager_ops[:10],
            "torch_compile": stats.compile_ops[:10],
            "fusion_op": stats.fusion_ops[:10],
            "mindspeed": stats.mindspeed_ops[:10],
        }

        result.stack_patterns = stats.top_stack_patterns

        # 识别潜在问题
        result.potential_issues = self._identify_potential_issues(stats, chains)

        return result

    def _identify_potential_issues(
        self,
        stats: CorrelationStats,
        chains: List[HostDeviceChain],
    ) -> List[str]:
        """识别潜在问题"""
        issues = []

        if stats.total_chains == 0:
            issues.append("未找到 Host-Device 关联数据，可能缺少 connection_id 信息")
            return issues

        # 检查 torch.compile 模式占比
        compile_ratio = stats.by_source_type.get("torch_compile", 0) / stats.total_chains
        if compile_ratio > 0.5:
            issues.append(f"检测到大量 torch.compile 模式算子 ({compile_ratio*100:.1f}%)，建议检查融合配置")

        # 检查是否有大量小算子
        small_op_threshold_us = 10  # 10us
        small_ops = [c for c in chains if c.device_op_dur < small_op_threshold_us]
        if len(small_ops) > stats.total_chains * 0.3:
            issues.append(f"检测到大量小算子 ({len(small_ops)} 个 < {small_op_threshold_us}us)，可能存在算子融合机会")

        # 检查 eager 和 compile 混合使用
        has_eager = stats.by_source_type.get("eager", 0) > 0
        has_compile = stats.by_source_type.get("torch_compile", 0) > 0
        if has_eager and has_compile:
            issues.append("检测到 eager 和 torch.compile 混合使用，建议统一执行模式")

        # 检查融合算子使用
        fusion_ratio = stats.by_source_type.get("fusion_op", 0) / stats.total_chains
        if fusion_ratio < 0.1 and stats.total_chains > 100:
            issues.append("融合算子使用率较低，建议检查是否可以启用更多算子融合")

        return issues


def correlate_host_device(
    trace_events: List[Dict[str, Any]],
) -> Tuple[List[HostDeviceChain], CorrelationStats]:
    """
    便捷函数：执行 Host-Device 关联分析

    Args:
        trace_events: trace_view.json 中的事件列表

    Returns:
        (chains, stats)
    """
    correlator = HostDeviceCorrelator()
    chains = correlator.build_call_chains(trace_events)
    stats = correlator.build_stats(chains)
    return chains, stats


def analyze_from_trace_file(
    trace_path: str,
) -> SourceAnalysisResult:
    """
    从 trace_view.json 文件执行分析

    Args:
        trace_path: trace_view.json 文件路径

    Returns:
        SourceAnalysisResult
    """
    from src.data_loader.stream_parser import StreamParser

    parser = StreamParser(trace_path)
    events = list(parser.iter_events())

    correlator = HostDeviceCorrelator()
    return correlator.analyze(events)


def build_call_chains_from_file(
    trace_path: str,
) -> List[HostDeviceChain]:
    """
    从 trace_view.json 文件构建调用链

    Args:
        trace_path: trace_view.json 文件路径

    Returns:
        HostDeviceChain 列表
    """
    from src.data_loader.stream_parser import StreamParser

    parser = StreamParser(trace_path)
    events = list(parser.iter_events())

    correlator = HostDeviceCorrelator()
    return correlator.build_call_chains(events)