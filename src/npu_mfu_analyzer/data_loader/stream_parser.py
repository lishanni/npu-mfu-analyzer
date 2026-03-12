"""
流式 JSON 解析器 - 处理 GB 级 trace_view.json

使用 ijson 进行流式解析，避免内存溢出。
复用自 msprof-analyze 的设计模式。

集成数据验证和错误处理功能。
"""

import os
import logging
from typing import Iterator, Callable, Any, Optional, Dict, List
from collections import defaultdict
from decimal import Decimal
import heapq

try:
    import ijson
except ImportError:
    ijson = None

from tqdm import tqdm

from .data_validator import (
    ProfilingDataValidator,
    ProfilingDataSanitizer,
    DataQualityReport,
    DataQualityLevel,
    DataIssue,
    IssueType,
)

logger = logging.getLogger(__name__)


class StreamParser:
    """
    流式 JSON 解析器

    支持两种模式：
    1. 流式解析（默认）：使用 ijson，逐条处理事件
    2. 全量加载：小文件或调试场景

    集成数据验证和错误处理功能。

    Usage:
        parser = StreamParser(json_path)
        parser.set_strict_mode(False)  # 宽松模式，继续处理有效数据
        for event in parser.iter_events():
            process(event)
        quality_report = parser.get_quality_report()
    """

    def __init__(self, file_path: str, disable_streaming: bool = False):
        """
        Args:
            file_path: JSON 文件路径
            disable_streaming: 是否禁用流式解析（环境变量 DISABLE_STREAMING_READER=1 也可禁用）
        """
        self.file_path = file_path
        self.disable_streaming = disable_streaming or os.getenv("DISABLE_STREAMING_READER") == "1"

        if ijson is None and not self.disable_streaming:
            logger.warning("ijson not installed, falling back to full load mode")
            self.disable_streaming = True

        self._file_size = self._get_file_size()
        self._event_count = 0

        # 数据验证相关
        self._strict_mode = False
        self._quality_report: Optional[DataQualityReport] = None
        self._error_count = 0
        self._warning_count = 0

    def _get_file_size(self) -> int:
        """获取文件大小（字节）"""
        try:
            return os.path.getsize(self.file_path)
        except OSError:
            return 0

    @property
    def file_size_mb(self) -> float:
        """文件大小（MB）"""
        return self._file_size / (1024 * 1024)

    def set_strict_mode(self, strict: bool):
        """
        设置严格模式

        Args:
            strict: True 表示遇到错误立即停止，False 表示跳过错误继续处理
        """
        self._strict_mode = strict

    def get_quality_report(self) -> Optional[DataQualityReport]:
        """获取数据质量报告"""
        return self._quality_report

    def iter_events(self, show_progress: bool = True) -> Iterator[Dict[str, Any]]:
        """
        迭代返回 JSON 数组中的每个事件

        包含数据验证和清洗功能：
        - 自动跳过无效事件
        - 修复常见数据问题
        - 记录数据质量问题

        Args:
            show_progress: 是否显示进度条

        Yields:
            dict: 清洗后的事件对象
        """
        if not os.path.exists(self.file_path):
            logger.error(f"File not found: {self.file_path}")
            return

        logger.info(f"Parsing {self.file_path} ({self.file_size_mb:.1f} MB), streaming={not self.disable_streaming}")

        # 初始化质量报告
        self._quality_report = DataQualityReport(file_path=self.file_path)

        try:
            with open(self.file_path, "r", encoding="utf-8") as f:
                if self.disable_streaming:
                    # 全量加载模式
                    import json
                    data = json.load(f)
                    iterator = iter(data) if isinstance(data, list) else iter([data])
                else:
                    # 流式解析模式
                    iterator = ijson.items(f, "item")

                # 包装进度条
                if show_progress:
                    iterator = tqdm(
                        iterator,
                        desc="Parsing events",
                        unit=" events",
                        leave=False,
                    )

                for i, raw_event in enumerate(iterator):
                    # 验证事件
                    issues = ProfilingDataValidator.validate_timeline_event(
                        raw_event, i, self.file_path
                    )

                    # 记录问题
                    for issue in issues:
                        self._quality_report.add_issue(issue)
                        if issue.severity == "error":
                            self._error_count += 1
                        elif issue.severity == "warning":
                            self._warning_count += 1

                    # 严格模式下遇到错误停止
                    if self._strict_mode and any(i.severity == "error" for i in issues):
                        logger.error(f"Strict mode: stopping at event {i} due to error")
                        break

                    # 清洗事件
                    cleaned_event, is_valid = ProfilingDataSanitizer.sanitize_timeline_event(
                        raw_event, i
                    )

                    if is_valid:
                        self._event_count += 1
                        yield cleaned_event
                    else:
                        self._quality_report.skipped_events += 1

                # 更新质量报告统计
                self._quality_report.total_events = i + 1
                self._quality_report.valid_events = self._event_count
                self._quality_report.has_timeline_data = self._event_count > 0

        except Exception as e:
            logger.error(f"Parse error: {e}", exc_info=True)
            self._quality_report.add_issue(DataIssue(
                issue_type=IssueType.PARSE_ERROR,
                severity="error",
                location=self.file_path,
                message=f"解析异常: {e}",
                suggestion="检查文件权限和格式",
            ))

    def get_parse_summary(self) -> Dict[str, Any]:
        """
        获取解析摘要信息

        Returns:
            {
                "total_events": int,
                "valid_events": int,
                "skipped_events": int,
                "error_count": int,
                "warning_count": int,
                "quality_level": str,
            }
        """
        if self._quality_report is None:
            return {
                "total_events": 0,
                "valid_events": 0,
                "skipped_events": 0,
                "error_count": 0,
                "warning_count": 0,
                "quality_level": "unknown",
            }

        return {
            "total_events": self._quality_report.total_events,
            "valid_events": self._quality_report.valid_events,
            "skipped_events": self._quality_report.skipped_events,
            "error_count": self._error_count,
            "warning_count": self._warning_count,
            "quality_level": self._quality_report.quality_level.value,
        }

    def parse_with_filter(
        self,
        filter_func: Callable[[Dict[str, Any]], bool],
        transform_func: Optional[Callable[[Dict[str, Any]], Any]] = None,
        show_progress: bool = True
    ) -> List[Any]:
        """
        带过滤和转换的解析
        
        Args:
            filter_func: 过滤函数，返回 True 保留该事件
            transform_func: 转换函数，可选，用于提取关键信息减少内存
            show_progress: 是否显示进度条
            
        Returns:
            过滤后的事件列表
        """
        results = []
        
        for event in self.iter_events(show_progress):
            if filter_func(event):
                if transform_func:
                    results.append(transform_func(event))
                else:
                    results.append(event)
        
        return results
    
    def extract_by_category(
        self, 
        categories: List[str],
        show_progress: bool = True
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        按 category 分类提取事件
        
        Args:
            categories: 要提取的 category 列表
            show_progress: 是否显示进度条
            
        Returns:
            {category: [events]}
        """
        results = defaultdict(list)
        categories_set = set(categories)
        
        for event in self.iter_events(show_progress):
            cat = event.get("cat", "")
            if cat in categories_set:
                results[cat].append(event)
        
        return dict(results)


class TimelineSummarizer:
    """
    Timeline 数据摘要化
    
    将 GB 级原始数据转换为 KB 级摘要，适合 LLM 分析。
    """
    
    def __init__(self, max_top_events: int = 20):
        self.max_top_events = max_top_events
        self.stats = {
            "total_events": 0,
            "total_duration_us": 0,
            "by_category": defaultdict(lambda: {"count": 0, "duration": 0}),
            "by_process": defaultdict(lambda: {"count": 0, "duration": 0}),
            "top_by_duration": [],  # heap
        }
        self._heap = []
    
    def process_event(self, event: Dict[str, Any]):
        """处理单个事件，更新统计信息"""
        self.stats["total_events"] += 1

        dur = event.get("dur", 0)
        # 统一转换为 float 类型（处理 str、Decimal 等类型）
        if isinstance(dur, str):
            try:
                dur = float(dur)
            except ValueError:
                dur = 0.0
        elif isinstance(dur, Decimal):
            dur = float(dur)
        elif not isinstance(dur, (int, float)):
            dur = 0.0

        cat = event.get("cat", "unknown")
        pid = event.get("pid", "unknown")
        name = event.get("name", "unknown")

        self.stats["total_duration_us"] += dur
        self.stats["by_category"][cat]["count"] += 1
        self.stats["by_category"][cat]["duration"] += dur
        self.stats["by_process"][str(pid)]["count"] += 1
        self.stats["by_process"][str(pid)]["duration"] += dur
        
        # 维护 Top N
        if len(self._heap) < self.max_top_events:
            heapq.heappush(self._heap, (dur, name, cat))
        elif dur > self._heap[0][0]:
            heapq.heapreplace(self._heap, (dur, name, cat))
    
    def get_summary(self) -> Dict[str, Any]:
        """获取摘要结果"""
        # 排序 Top N
        self.stats["top_by_duration"] = sorted(self._heap, reverse=True)
        
        # 转换 defaultdict 为普通 dict
        self.stats["by_category"] = dict(self.stats["by_category"])
        self.stats["by_process"] = dict(self.stats["by_process"])
        
        return self.stats
    
    @classmethod
    def summarize_file(cls, file_path: str, max_top_events: int = 20) -> Dict[str, Any]:
        """
        一站式摘要化接口
        
        Args:
            file_path: trace_view.json 路径
            max_top_events: 保留的 Top N 事件数
            
        Returns:
            摘要统计信息
        """
        parser = StreamParser(file_path)
        summarizer = cls(max_top_events)
        
        for event in parser.iter_events():
            summarizer.process_event(event)
        
        return summarizer.get_summary()


def extract_overlap_events(file_path: str) -> Dict[str, List[Dict[str, Any]]]:
    """
    提取 Overlap Analysis 相关事件
    
    识别 Computing、Communication(Not Overlapped)、Free 等事件。
    
    Args:
        file_path: trace_view.json 路径
        
    Returns:
        {
            "compute": [...],
            "comm_not_overlap": [...],
            "free": [...],
            "hccl": [...]
        }
    """
    result = {
        "compute": [],
        "comm_not_overlap": [],
        "free": [],
        "hccl": [],
    }
    
    parser = StreamParser(file_path)
    
    for event in parser.iter_events():
        name = event.get("name", "")
        cat = event.get("cat", "")
        args = event.get("args", {})
        
        # Overlap Analysis 泳道的事件
        process_name = args.get("name", "") if isinstance(args, dict) else ""
        
        if process_name == "Overlap Analysis" or "Overlap" in str(event.get("pid", "")):
            if name == "Computing":
                result["compute"].append(_extract_time_info(event))
            elif name == "Communication(Not Overlapped)":
                result["comm_not_overlap"].append(_extract_time_info(event))
            elif name == "Free":
                result["free"].append(_extract_time_info(event))
        
        # HCCL 通信事件
        elif cat in ("Communication", "hccl") or "hccl" in name.lower():
            result["hccl"].append(_extract_time_info(event))
        
        # Kernel 计算事件
        elif cat == "Kernel":
            result["compute"].append(_extract_time_info(event))
    
    return result


def _extract_time_info(event: Dict[str, Any]) -> Dict[str, Any]:
    """提取事件的关键时间信息，减少内存占用"""
    ts = event.get("ts", 0)
    dur = event.get("dur", 0)

    # 统一转换为 float 类型（处理 str、Decimal 等类型）
    if isinstance(ts, str):
        try:
            ts = float(ts)
        except ValueError:
            ts = 0.0
    elif isinstance(ts, Decimal):
        ts = float(ts)
    elif not isinstance(ts, (int, float)):
        ts = 0.0

    if isinstance(dur, str):
        try:
            dur = float(dur)
        except ValueError:
            dur = 0.0
    elif isinstance(dur, Decimal):
        dur = float(dur)
    elif not isinstance(dur, (int, float)):
        dur = 0.0

    return {
        "name": event.get("name", ""),
        "ts": ts,
        "dur": dur,
        "cat": event.get("cat", ""),
        "pid": event.get("pid", ""),
        "tid": event.get("tid", ""),
    }
