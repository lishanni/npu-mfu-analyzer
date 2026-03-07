"""
AIC 微架构分析模块

提供指令级、内存层次、流水线等深度分析能力，匹配 msprof 框架采集的 PMU 数据。
"""

from src.analyzers.aic.instruction_analyzer import InstructionAnalyzer
from src.analyzers.aic.memory_hierarchy_analyzer import MemoryHierarchyAnalyzer
from src.analyzers.aic.pipeline_analyzer import PipelineAnalyzer
from src.analyzers.aic.pmu_data_parser import PMUDataParser
from src.analyzers.aic.microarch_report import MicroarchReportGenerator

__all__ = [
    "InstructionAnalyzer",
    "MemoryHierarchyAnalyzer",
    "PipelineAnalyzer",
    "PMUDataParser",
    "MicroarchReportGenerator",
]
