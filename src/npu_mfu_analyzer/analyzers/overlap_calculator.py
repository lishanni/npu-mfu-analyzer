"""
Overlap 计算器

计算 HCCL 通信流与 Computing Stream 的重叠关系。
复用自 msprof-analyze/msprof_analyze/cluster_analyse/common_func/time_range_calculator.py
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Iterator, Optional
import logging

logger = logging.getLogger(__name__)

DEFAULT_INT_VALUE = -1


@dataclass
class TimeRange:
    """时间区间"""
    start_ts: int = DEFAULT_INT_VALUE
    end_ts: int = DEFAULT_INT_VALUE
    
    @property
    def duration(self) -> int:
        """区间持续时间"""
        if self.start_ts == DEFAULT_INT_VALUE or self.end_ts == DEFAULT_INT_VALUE:
            return 0
        return max(0, self.end_ts - self.start_ts)
    
    def overlaps(self, other: "TimeRange") -> bool:
        """判断是否与另一个区间重叠"""
        return self.start_ts < other.end_ts and other.start_ts < self.end_ts
    
    def __repr__(self) -> str:
        return f"TimeRange({self.start_ts}, {self.end_ts})"


class CommunicationTimeRange(TimeRange):
    """通信类型的时间区间（用于区分计算与通信）"""
    pass


class RangeCalculator:
    """时间区间计算器"""
    
    @staticmethod
    def generate_time_range(
        start: int, 
        end: int, 
        class_range: type = TimeRange
    ) -> TimeRange:
        """创建时间区间"""
        time_range = class_range()
        time_range.start_ts = start
        time_range.end_ts = end
        return time_range
    
    @staticmethod
    def merge_continuous_intervals(time_range_list: List[TimeRange]) -> List[TimeRange]:
        """合并连续/重叠的时间区间"""
        result = []
        if not time_range_list:
            return result
        
        sorted_ranges = sorted(time_range_list, key=lambda x: x.start_ts)
        current_range = sorted_ranges[0]
        
        for time_range in sorted_ranges[1:]:
            if time_range.start_ts <= current_range.end_ts:
                # 区间重叠或相邻，合并
                current_range.end_ts = max(current_range.end_ts, time_range.end_ts)
            else:
                # 区间不连续，保存当前区间，开始新区间
                result.append(current_range)
                current_range = time_range
        
        result.append(current_range)
        return result
    
    @staticmethod
    def compute_pipeline_overlap(
        communication_range: List[CommunicationTimeRange], 
        compute_range: List[TimeRange]
    ) -> tuple:
        """
        计算通信与计算的重叠关系
        
        Args:
            communication_range: HCCL 通信的时间区间列表
            compute_range: AICore 计算的时间区间列表
            
        Returns:
            (pure_communication_range, free_time_range):
            - pure_communication_range: 未被计算掩盖的纯通信时间区间
            - free_time_range: 空闲时间区间（无计算也无通信）
        """
        free_time_range = []
        pure_communication_range = []
        
        time_range_list = sorted(
            communication_range + compute_range, 
            key=lambda x: x.start_ts
        )
        
        if not time_range_list:
            return pure_communication_range, free_time_range
        
        min_range = time_range_list.pop(0)
        
        for time_range in time_range_list:
            # Case 1: 区间不重叠，存在 gap（free time）
            if min_range.end_ts - time_range.start_ts < 0:
                free_time_range.append(
                    RangeCalculator.generate_time_range(min_range.end_ts, time_range.start_ts)
                )
                if isinstance(min_range, CommunicationTimeRange):
                    pure_communication_range.append(
                        RangeCalculator.generate_time_range(min_range.start_ts, min_range.end_ts)
                    )
                min_range = time_range
                continue
            
            # Case 2: 区间重叠
            if min_range.end_ts - time_range.end_ts < 0:
                # min_range 结束早于 time_range
                if isinstance(min_range, CommunicationTimeRange):
                    # 通信区间在计算开始前的部分未被掩盖
                    pure_communication_range.append(
                        RangeCalculator.generate_time_range(min_range.start_ts, time_range.start_ts)
                    )
                    min_range = RangeCalculator.generate_time_range(min_range.end_ts, time_range.end_ts)
                if isinstance(time_range, CommunicationTimeRange):
                    min_range = RangeCalculator.generate_time_range(
                        min_range.end_ts, time_range.end_ts, 
                        class_range=CommunicationTimeRange
                    )
            else:
                # min_range 结束晚于或等于 time_range
                if isinstance(min_range, CommunicationTimeRange):
                    pure_communication_range.append(
                        RangeCalculator.generate_time_range(min_range.start_ts, time_range.start_ts)
                    )
                    min_range = RangeCalculator.generate_time_range(
                        time_range.end_ts, min_range.end_ts, 
                        class_range=CommunicationTimeRange
                    )
                if isinstance(time_range, CommunicationTimeRange):
                    min_range = RangeCalculator.generate_time_range(time_range.end_ts, min_range.end_ts)
        
        # 处理最后一个区间
        if isinstance(min_range, CommunicationTimeRange):
            pure_communication_range.append(min_range)
        
        return pure_communication_range, free_time_range


@dataclass
class OverlapMetrics:
    """Overlap 分析指标"""
    total_compute_time: float = 0.0      # 总计算时间 (us)
    total_comm_time: float = 0.0         # 总通信时间 (us)
    overlapped_time: float = 0.0         # 被掩盖的通信时间 (us)
    comm_not_overlapped: float = 0.0     # 未被掩盖的通信时间 (us)
    free_time: float = 0.0               # 空闲时间 (us)
    overlap_ratio: float = 0.0           # 通信掩盖率 (%)
    e2e_time: float = 0.0                # 端到端时间 (us)
    
    def to_dict(self) -> Dict[str, float]:
        return {
            "total_compute_time": self.total_compute_time,
            "total_comm_time": self.total_comm_time,
            "overlapped_time": self.overlapped_time,
            "comm_not_overlapped": self.comm_not_overlapped,
            "free_time": self.free_time,
            "overlap_ratio": self.overlap_ratio,
            "e2e_time": self.e2e_time,
        }
    
    def to_prompt_text(self) -> str:
        """转换为 LLM Prompt 格式"""
        return f"""## Overlap 分析指标
- 总计算时间: {self.total_compute_time / 1000:.2f} ms
- 总通信时间: {self.total_comm_time / 1000:.2f} ms
- 通信掩盖率: {self.overlap_ratio:.1f}%
- 已掩盖通信: {self.overlapped_time / 1000:.2f} ms
- 未掩盖通信: {self.comm_not_overlapped / 1000:.2f} ms（在关键路径上）
- 空闲时间: {self.free_time / 1000:.2f} ms
- 端到端时间: {self.e2e_time / 1000:.2f} ms"""


class OverlapCalculator:
    """
    Overlap 计算器
    
    计算 Computing Stream 与 Communication Stream 的重叠关系，
    用于分析通信掩盖率（通信是否被计算隐藏）。
    """
    
    def __init__(self):
        self._range_calculator = RangeCalculator()
    
    def calculate_from_events(
        self, 
        compute_events: List[Dict[str, Any]], 
        comm_events: List[Dict[str, Any]]
    ) -> OverlapMetrics:
        """
        从事件列表计算 Overlap 指标
        
        Args:
            compute_events: 计算事件列表，每个事件需包含 ts（开始时间）和 dur（持续时间）
            comm_events: 通信事件列表，格式同上
            
        Returns:
            OverlapMetrics: Overlap 分析指标
        """
        metrics = OverlapMetrics()
        
        if not compute_events and not comm_events:
            return metrics
        
        # 1. 构建时间区间
        compute_ranges = []
        for e in compute_events:
            ts = e.get("ts", 0)
            dur = e.get("dur", 0)
            if dur > 0:
                compute_ranges.append(
                    self._range_calculator.generate_time_range(ts, ts + dur)
                )
        
        comm_ranges = []
        for e in comm_events:
            ts = e.get("ts", 0)
            dur = e.get("dur", 0)
            if dur > 0:
                comm_ranges.append(
                    self._range_calculator.generate_time_range(
                        ts, ts + dur, 
                        class_range=CommunicationTimeRange
                    )
                )
        
        # 2. 合并连续区间
        compute_ranges = self._range_calculator.merge_continuous_intervals(compute_ranges)
        comm_ranges = self._range_calculator.merge_continuous_intervals(comm_ranges)
        
        # 3. 计算重叠
        pure_comm, free_ranges = self._range_calculator.compute_pipeline_overlap(
            comm_ranges, compute_ranges
        )
        
        # 4. 统计指标
        metrics.total_compute_time = sum(e.get("dur", 0) for e in compute_events)
        metrics.total_comm_time = sum(e.get("dur", 0) for e in comm_events)
        metrics.comm_not_overlapped = sum(r.duration for r in pure_comm)
        metrics.overlapped_time = metrics.total_comm_time - metrics.comm_not_overlapped
        metrics.free_time = sum(r.duration for r in free_ranges)
        
        # 5. 计算掩盖率
        if metrics.total_comm_time > 0:
            metrics.overlap_ratio = (metrics.overlapped_time / metrics.total_comm_time) * 100
        
        # 6. 计算端到端时间
        all_events = compute_events + comm_events
        if all_events:
            min_ts = min(e.get("ts", float("inf")) for e in all_events)
            max_ts = max(e.get("ts", 0) + e.get("dur", 0) for e in all_events)
            metrics.e2e_time = max_ts - min_ts
        
        return metrics
    
    def calculate_from_step_trace(
        self, 
        step_trace_row: Dict[str, Any]
    ) -> OverlapMetrics:
        """
        从 STEP_TRACE 行数据计算 Overlap 指标
        
        Args:
            step_trace_row: 包含 computing, communication, communication_not_overlapped, 
                           overlapped, free 等字段的字典
        
        Returns:
            OverlapMetrics: Overlap 分析指标
        """
        metrics = OverlapMetrics()
        
        metrics.total_compute_time = float(step_trace_row.get("computing", 0) or 0)
        metrics.total_comm_time = float(step_trace_row.get("communication", 0) or 0)
        metrics.comm_not_overlapped = float(step_trace_row.get("communication_not_overlapped", 0) or 0)
        metrics.overlapped_time = float(step_trace_row.get("overlapped", 0) or 0)
        metrics.free_time = float(step_trace_row.get("free", 0) or 0)
        
        # 如果没有 overlapped 字段，从 total_comm - not_overlapped 计算
        if metrics.overlapped_time == 0 and metrics.total_comm_time > 0:
            metrics.overlapped_time = max(0, metrics.total_comm_time - metrics.comm_not_overlapped)
        
        # 计算掩盖率
        if metrics.total_comm_time > 0:
            metrics.overlap_ratio = (metrics.overlapped_time / metrics.total_comm_time) * 100
        
        # 计算端到端时间
        stage_time = float(step_trace_row.get("stage", 0) or 0)
        if stage_time > 0:
            metrics.e2e_time = stage_time
        else:
            # 估算 E2E = compute + comm_not_overlapped + free
            metrics.e2e_time = (
                metrics.total_compute_time + 
                metrics.comm_not_overlapped + 
                metrics.free_time
            )
        
        return metrics
    
    def extract_overlap_events_from_trace(
        self, 
        trace_events: Iterator[Dict[str, Any]]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        从 trace_view.json 事件流中提取 Overlap 相关事件
        
        识别方式：
        1. 从 "Overlap Analysis" 进程/线程提取（msprof 预计算）
        2. 从原始事件分类（Computing/Communication）
        
        Args:
            trace_events: trace_view.json 事件迭代器
            
        Returns:
            {
                "compute": [...],          # 计算事件
                "comm_not_overlap": [...], # 未掩盖通信事件
                "free": [...],             # 空闲时间事件
                "hccl": [...],             # HCCL 通信事件
            }
        """
        result = {
            "compute": [],
            "comm_not_overlap": [],
            "free": [],
            "hccl": [],
        }
        
        for event in trace_events:
            name = event.get("name", "")
            cat = event.get("cat", "")
            args = event.get("args", {})
            
            # 方式 1：从 Overlap Analysis 泳道提取（推荐，msprof 预计算结果）
            if args.get("name") == "Overlap Analysis" or "Overlap Analysis" in str(event.get("pid", "")):
                if name == "Computing":
                    result["compute"].append(event)
                elif "Communication(Not Overlapped)" in name or name == "Communication(Not Overlapped)":
                    result["comm_not_overlap"].append(event)
                elif name == "Free":
                    result["free"].append(event)
                continue
            
            # 方式 2：从原始事件分类
            # 计算事件：Kernel 类型且在 AICore 上
            if cat == "Kernel" or "aicore" in name.lower() or "matmul" in name.lower():
                result["compute"].append(event)
            
            # 通信事件：HCCL 相关
            elif cat in ("Communication", "hccl", "HCCL") or "allreduce" in name.lower() or "allgather" in name.lower():
                result["hccl"].append(event)
        
        return result


def calculate_overlap_from_trace_file(json_path: str) -> OverlapMetrics:
    """
    从 trace_view.json 文件计算 Overlap 指标
    
    Args:
        json_path: trace_view.json 文件路径
        
    Returns:
        OverlapMetrics: Overlap 分析指标
    """
    from npu_mfu_analyzer.data_loader.stream_parser import StreamParser
    
    parser = StreamParser(json_path)
    calculator = OverlapCalculator()
    
    # 提取事件
    events = calculator.extract_overlap_events_from_trace(parser.iter_events())
    
    # 优先使用预计算的 Overlap Analysis 数据
    if events["compute"] or events["comm_not_overlap"]:
        # 从预计算数据直接统计
        metrics = OverlapMetrics()
        metrics.total_compute_time = sum(e.get("dur", 0) for e in events["compute"])
        metrics.comm_not_overlapped = sum(e.get("dur", 0) for e in events["comm_not_overlap"])
        metrics.free_time = sum(e.get("dur", 0) for e in events["free"])
        metrics.total_comm_time = sum(e.get("dur", 0) for e in events["hccl"])
        
        if metrics.total_comm_time > 0:
            metrics.overlapped_time = max(0, metrics.total_comm_time - metrics.comm_not_overlapped)
            metrics.overlap_ratio = (metrics.overlapped_time / metrics.total_comm_time) * 100
        
        # E2E 估算
        metrics.e2e_time = metrics.total_compute_time + metrics.comm_not_overlapped + metrics.free_time
        
        return metrics
    
    # 回退：从原始事件计算
    return calculator.calculate_from_events(
        events["compute"], 
        events["hccl"]
    )
