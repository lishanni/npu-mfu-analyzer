"""
Skill Engine 模块

提供专家技能引擎，支持 Python 精确计算技能和 Prompt 推理指导技能
"""

from .base_skill import (
    BaseSkill,
    PromptSkill,
    SkillCategory,
    SkillPriority,
    SkillInput,
    SkillOutput,
    SkillResult,
    SkillMetadata,
)

from .registry import (
    SkillRegistry,
    get_registry,
    register_skill,
    skill,
    auto_discover_skills,
)

from .engine import (
    SkillEngine,
    LogicChain,
    ChainStep,
    ChainResult,
    ChainStatus,
    get_engine,
    create_builtin_chains,
)

__all__ = [
    # Base Skill
    "BaseSkill",
    "PromptSkill",
    "SkillCategory",
    "SkillPriority",
    "SkillInput",
    "SkillOutput",
    "SkillResult",
    "SkillMetadata",
    # Registry
    "SkillRegistry",
    "get_registry",
    "register_skill",
    "skill",
    "auto_discover_skills",
    # Engine
    "SkillEngine",
    "LogicChain",
    "ChainStep",
    "ChainResult",
    "ChainStatus",
    "get_engine",
    "create_builtin_chains",
]
