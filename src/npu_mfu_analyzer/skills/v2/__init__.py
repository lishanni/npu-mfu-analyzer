"""
Skills v2 - 统一技能系统

提供统一的技能接口、上下文和结果定义。
"""

from .base import (
    SkillType,
    SkillCategory,
    SkillMetadata,
    SkillContext,
    SkillResult,
    BaseSkill,
)
from .registry import SkillRegistry, get_registry
from .engine import SkillEngine, SkillChain
from .loader import SkillLoader

__all__ = [
    # 类型枚举
    "SkillType",
    "SkillCategory",
    # 数据结构
    "SkillMetadata",
    "SkillContext",
    "SkillResult",
    # 基类
    "BaseSkill",
    # 注册和执行
    "SkillRegistry",
    "get_registry",
    "SkillEngine",
    "SkillChain",
    "SkillLoader",
]