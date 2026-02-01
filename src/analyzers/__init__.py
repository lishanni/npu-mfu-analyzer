"""
分析器模块

提供各种性能分析算法，复用 msprof-analyze 的核心逻辑。
"""

from src.analyzers.overlap_calculator import (
    TimeRange,
    CommunicationTimeRange,
    RangeCalculator,
    OverlapCalculator,
    OverlapMetrics,
    calculate_overlap_from_trace_file,
)

from src.analyzers.slow_rank_detector import (
    judge_dixon,
    judge_norm,
    judge_slow_rank,
    SlowRankDetector,
    SlowRankResult,
    SlowRankInfo,
    DIXON_TABLE_995,
)

from src.analyzers.bubble_analyzer import (
    BubbleAnalyzer,
    BubbleMetrics,
)

from src.analyzers.comm_splitter import (
    CommunicationSplitter,
    CommSplitResult,
    ParallelConfig,
    ParallelGroupBuilder,
)

from src.analyzers.mfu_calculator import (
    MFUCalculator,
    MFUMetrics,
    ChipInfo,
    OperatorMFU,
    DataType,
    OperatorType,
)

from src.analyzers.history_comparator import (
    HistoryComparator,
    ProfilingSnapshot,
    ComparisonResult,
)

__all__ = [
    # Overlap
    "TimeRange",
    "CommunicationTimeRange",
    "RangeCalculator",
    "OverlapCalculator",
    "OverlapMetrics",
    "calculate_overlap_from_trace_file",
    # Slow Rank
    "judge_dixon",
    "judge_norm",
    "judge_slow_rank",
    "SlowRankDetector",
    "SlowRankResult",
    "SlowRankInfo",
    "DIXON_TABLE_995",
    # Bubble
    "BubbleAnalyzer",
    "BubbleMetrics",
    # Communication Split
    "CommunicationSplitter",
    "CommSplitResult",
    "ParallelConfig",
    "ParallelGroupBuilder",
    # MFU
    "MFUCalculator",
    "MFUMetrics",
    "ChipInfo",
    "OperatorMFU",
    "DataType",
    "OperatorType",
    # History Comparison
    "HistoryComparator",
    "ProfilingSnapshot",
    "ComparisonResult",
]
