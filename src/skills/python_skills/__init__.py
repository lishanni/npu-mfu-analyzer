"""
Python Skills 模块

包含所有内置的 Python 精确计算技能
"""

from .mfu_skill import CalculateMFUSkill, EstimateModelFLOPsSkill
from .bandwidth_skill import CheckBandwidthEfficiencySkill, AnalyzeCollectiveOpsSkill
from .overlap_skill import CheckOverlapRatioSkill, VerifyOverlapStrategySkill
from .jitter_skill import DetectComputeJitterSkill, DetectCommJitterSkill, AnalyzeCrossRankJitterSkill
from .slow_rank_skill import DetectSlowRankSkill

__all__ = [
    # MFU 相关
    "CalculateMFUSkill",
    "EstimateModelFLOPsSkill",
    # 带宽相关
    "CheckBandwidthEfficiencySkill",
    "AnalyzeCollectiveOpsSkill",
    # 掩盖率相关
    "CheckOverlapRatioSkill",
    "VerifyOverlapStrategySkill",
    # 抖动检测
    "DetectComputeJitterSkill",
    "DetectCommJitterSkill",
    "AnalyzeCrossRankJitterSkill",
    # 慢卡检测
    "DetectSlowRankSkill",
]


def register_all_skills():
    """注册所有内置 Python 技能"""
    from ..registry import register_skill
    
    skills = [
        # MFU 相关
        CalculateMFUSkill(),
        EstimateModelFLOPsSkill(),
        # 带宽相关
        CheckBandwidthEfficiencySkill(),
        AnalyzeCollectiveOpsSkill(),
        # 掩盖率相关
        CheckOverlapRatioSkill(),
        VerifyOverlapStrategySkill(),
        # 抖动检测
        DetectComputeJitterSkill(),
        DetectCommJitterSkill(),
        AnalyzeCrossRankJitterSkill(),
        # 慢卡检测
        DetectSlowRankSkill(),
    ]
    
    for skill in skills:
        register_skill(skill)
    
    return len(skills)
