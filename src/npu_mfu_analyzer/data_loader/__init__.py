"""数据加载模块 - Profiling 数据解析"""

from npu_mfu_analyzer.data_loader.profiling_loader import ProfilingLoader
from npu_mfu_analyzer.data_loader.stream_parser import StreamParser
from npu_mfu_analyzer.data_loader.aic_metrics import (
    AICMetrics,
    AICAnalysisResult,
    ArithmeticUtilization,
    MemoryMetrics,
    PipelineMetrics,
    DetailedOperatorAnalysisData,
)
from npu_mfu_analyzer.data_loader.stack_types import (
    StackFrame,
    HostStack,
    OperatorWithStack,
    HostDeviceChain,
    CorrelationStats,
    SourceAnalysisResult,
    STACK_PATTERNS,
)
from npu_mfu_analyzer.data_loader.stack_parser import (
    StackParser,
    StackPatternDiscovery,
    extract_stack_from_events,
    analyze_stack_distribution,
)

__all__ = [
    "ProfilingLoader",
    "StreamParser",
    "AICMetrics",
    "AICAnalysisResult",
    "ArithmeticUtilization",
    "MemoryMetrics",
    "PipelineMetrics",
    "DetailedOperatorAnalysisData",
    # Stack types
    "StackFrame",
    "HostStack",
    "OperatorWithStack",
    "HostDeviceChain",
    "CorrelationStats",
    "SourceAnalysisResult",
    "STACK_PATTERNS",
    # Stack parser
    "StackParser",
    "StackPatternDiscovery",
    "extract_stack_from_events",
    "analyze_stack_distribution",
]
