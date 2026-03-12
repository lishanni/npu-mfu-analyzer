"""Agent 模块 - Multi-Agent 核心实现"""

from npu_mfu_analyzer.agents.base_agent import BaseAgent, AgentMessage, AnalysisResult
from npu_mfu_analyzer.agents.orchestrator import Orchestrator, AnalysisReport
from npu_mfu_analyzer.agents.timeline_agent import TimelineAgent, EnhancedTimelineAgent, TimelineAnalysisData
from npu_mfu_analyzer.agents.operator_agent import OperatorAgent, OperatorAnalysisData
from npu_mfu_analyzer.agents.memory_agent import MemoryAgent, MemoryMetrics, MemoryAnalysisData
from npu_mfu_analyzer.agents.communication_agent import CommunicationAgent, CommunicationMetrics, CommunicationAnalysisData
from npu_mfu_analyzer.agents.advisor_agent import AdvisorAgent, AdvisorReport, OptimizationSuggestion, Priority
from npu_mfu_analyzer.agents.jitter_agent import JitterAgent, JitterMetrics, JitterDetector, detect_jitter_from_loader
from npu_mfu_analyzer.agents.detailed_operator_agent import DetailedOperatorAgent, DetailedOperatorAnalysisData
from npu_mfu_analyzer.agents.comparison_agent import ComparisonAdvisorAgent

__all__ = [
    # Base
    "BaseAgent",
    "AgentMessage",
    "AnalysisResult",
    "Orchestrator",
    "AnalysisReport",
    # Timeline
    "TimelineAgent",
    "EnhancedTimelineAgent",
    "TimelineAnalysisData",
    # Operator
    "OperatorAgent",
    "OperatorAnalysisData",
    # Memory
    "MemoryAgent",
    "MemoryMetrics",
    "MemoryAnalysisData",
    # Communication
    "CommunicationAgent",
    "CommunicationMetrics",
    "CommunicationAnalysisData",
    # Advisor
    "AdvisorAgent",
    "AdvisorReport",
    "OptimizationSuggestion",
    "Priority",
    # Jitter
    "JitterAgent",
    "JitterMetrics",
    "JitterDetector",
    "detect_jitter_from_loader",
    # Detailed Operator
    "DetailedOperatorAgent",
    "DetailedOperatorAnalysisData",
    # Comparison
    "ComparisonAdvisorAgent",
]
