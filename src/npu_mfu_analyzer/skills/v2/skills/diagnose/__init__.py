"""
诊断技能模块

包含各类性能诊断规则。
"""

from pathlib import Path

# 自动加载 Markdown 规则
RULES_DIR = Path(__file__).parent / "rules"

__all__ = [
    "RULES_DIR",
]