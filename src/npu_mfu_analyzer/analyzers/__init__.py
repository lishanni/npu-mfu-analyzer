"""
分析器模块

提供各种性能分析算法，复用 msprof-analyze 的核心逻辑。
"""

from npu_mfu_analyzer.analyzers.overlap_calculator import (
    TimeRange,
    CommunicationTimeRange,
    RangeCalculator,
    OverlapCalculator,
    OverlapMetrics,
    calculate_overlap_from_trace_file,
)

from npu_mfu_analyzer.analyzers.slow_rank_detector import (
    judge_dixon,
    judge_norm,
    judge_slow_rank,
    SlowRankDetector,
    SlowRankResult,
    SlowRankInfo,
    DIXON_TABLE_995,
)

from npu_mfu_analyzer.analyzers.bubble_analyzer import (
    BubbleAnalyzer,
    BubbleMetrics,
    PPScheduleType,
    PPScheduleAnalysis,
    BubbleBreakdown,
    RecomputationAnalysis,
)

from npu_mfu_analyzer.analyzers.comm_splitter import (
    CommunicationSplitter,
    CommSplitResult,
    ParallelConfig,
    ParallelGroupBuilder,
)

from npu_mfu_analyzer.analyzers.mfu_calculator import (
    MFUCalculator,
    MFUMetrics,
    ChipInfo,
    OperatorMFU,
    DataType,
    OperatorType,
)

from npu_mfu_analyzer.analyzers.history_comparator import (
    HistoryComparator,
    ProfilingSnapshot,
    ComparisonResult,
)

from npu_mfu_analyzer.analyzers.similarity_checker import (
    SimilarityChecker,
    SimilarityResult,
    SimilarityDimension,
    ComparabilityLevel,
)

from npu_mfu_analyzer.analyzers.profiling_diff import (
    ProfilingDiffEngine,
    ProfilingDiff,
    SummaryDiff,
    OperatorDiff,
    OperatorChange,
    TimelineDiff,
    CommDiff,
    MemoryDiff,
    MetricChange,
    SourceChange,
    RootCauseFinding,
)

from npu_mfu_analyzer.analyzers.comparison_orchestrator import (
    ComparisonOrchestrator,
    ComparisonReport,
    run_comparison,
)

from npu_mfu_analyzer.analyzers.communication_matrix_analyzer import (
    CommunicationMatrixAnalyzer,
    CommunicationMatrix,
    LinkMetrics,
    TransportType,
    CommOpType,
    CommOpStatistics,
    CommunicationMatrixReport,
)

from npu_mfu_analyzer.analyzers.link_performance_dashboard import (
    LinkPerformanceDashboard,
    DashboardData,
    generate_dashboard,
)

from npu_mfu_analyzer.analyzers.aic import (
    InstructionAnalyzer,
    MemoryHierarchyAnalyzer,
    PipelineAnalyzer,
    PMUDataParser,
    MicroarchReportGenerator,
)

# 尝试导入可能不存在的类
try:
    from npu_mfu_analyzer.analyzers.aic.instruction_analyzer import InstructionBottleneck
except ImportError:
    InstructionBottleneck = None

try:
    from npu_mfu_analyzer.analyzers.aic.memory_hierarchy_analyzer import MemoryHierarchyAnalysis
except ImportError:
    MemoryHierarchyAnalysis = None

try:
    from npu_mfu_analyzer.analyzers.aic.pipeline_analyzer import PipelineAnalysis
except ImportError:
    PipelineAnalysis = None

try:
    from npu_mfu_analyzer.analyzers.aic.microarch_report import generate_microarch_report
except ImportError:
    generate_microarch_report = None

from npu_mfu_analyzer.analyzers.host_device_correlator import (
    HostDeviceCorrelator,
    HostEvent,
    DeviceEvent,
    correlate_host_device,
    analyze_from_trace_file,
    build_call_chains_from_file,
)

from npu_mfu_analyzer.analyzers.operator_source_classifier import (
    OperatorSourceClassifier,
    ClassificationResult,
    SourceChange,
    StackPatternDiscovery,
    classify_operators,
    discover_new_patterns,
    EXTENDED_STACK_PATTERNS,
)

from npu_mfu_analyzer.analyzers.root_cause_engine import (
    RootCauseSkillEngine,
    RootCauseRule,
    RootCauseFinding,
    analyze_root_causes,
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
    "SourceChange",
    "RootCauseFinding",
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
    # AIC Microarchitecture
    "InstructionAnalyzer",
    "InstructionBottleneck",
    "MemoryHierarchyAnalyzer",
    "MemoryHierarchyAnalysis",
    "PipelineAnalyzer",
    "PipelineAnalysis",
    "PMUDataParser",
    "MicroarchReportGenerator",
    "generate_microarch_report",
    # Host-Device Correlation
    "HostDeviceCorrelator",
    "HostEvent",
    "DeviceEvent",
    "correlate_host_device",
    "analyze_from_trace_file",
    # Operator Source Classifier
    "OperatorSourceClassifier",
    "ClassificationResult",
    "SourceChange",
    "StackPatternDiscovery",
    "classify_operators",
    "discover_new_patterns",
    "EXTENDED_STACK_PATTERNS",
    # Root Cause Engine
    "RootCauseSkillEngine",
    "RootCauseRule",
    "RootCauseFinding",
    "analyze_root_causes",
]
