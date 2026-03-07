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
    PPScheduleType,
    PPScheduleAnalysis,
    BubbleBreakdown,
    RecomputationAnalysis,
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

from src.analyzers.similarity_checker import (
    SimilarityChecker,
    SimilarityResult,
    SimilarityDimension,
    ComparabilityLevel,
)

from src.analyzers.profiling_diff import (
    ProfilingDiffEngine,
    ProfilingDiff,
    SummaryDiff,
    OperatorDiff,
    OperatorChange,
    TimelineDiff,
    CommDiff,
    MemoryDiff,
    MetricChange,
)

from src.analyzers.comparison_orchestrator import (
    ComparisonOrchestrator,
    ComparisonReport,
    run_comparison,
)

from src.analyzers.communication_matrix_analyzer import (
    CommunicationMatrixAnalyzer,
    CommunicationMatrix,
    LinkMetrics,
    TransportType,
    CommOpType,
    CommOpStatistics,
    CommunicationMatrixReport,
)

from src.analyzers.link_performance_dashboard import (
    LinkPerformanceDashboard,
    DashboardData,
    generate_dashboard,
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
    "PPScheduleType",
    "PPScheduleAnalysis",
    "BubbleBreakdown",
    "RecomputationAnalysis",
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
    # Similarity Checker
    "SimilarityChecker",
    "SimilarityResult",
    "SimilarityDimension",
    "ComparabilityLevel",
    # Profiling Diff
    "ProfilingDiffEngine",
    "ProfilingDiff",
    "SummaryDiff",
    "OperatorDiff",
    "OperatorChange",
    "TimelineDiff",
    "CommDiff",
    "MemoryDiff",
    "MetricChange",
    # Comparison Orchestrator
    "ComparisonOrchestrator",
    "ComparisonReport",
    "run_comparison",
    # Communication Matrix
    "CommunicationMatrixAnalyzer",
    "CommunicationMatrix",
    "LinkMetrics",
    "TransportType",
    "CommOpType",
    "CommOpStatistics",
    "CommunicationMatrixReport",
    # Dashboard
    "LinkPerformanceDashboard",
    "DashboardData",
    "generate_dashboard",
]
