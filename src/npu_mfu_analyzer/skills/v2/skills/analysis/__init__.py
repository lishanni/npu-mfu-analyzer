"""
分析技能模块

包含时间线分析、通信分析等深度分析技能。
"""

from .timeline_skill import TimelineAnalysisSkill
from .communication_skill import CommunicationAnalysisSkill

__all__ = [
    "TimelineAnalysisSkill",
    "CommunicationAnalysisSkill",
]