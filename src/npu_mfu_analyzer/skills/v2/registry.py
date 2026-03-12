"""
Skills v2 注册中心

管理所有技能的注册、发现和检索。
"""

from typing import Dict, List, Optional, Set, Type, Any
from dataclasses import dataclass, field
from pathlib import Path
import logging
import importlib
import json

from .base import (
    BaseSkill,
    SkillType,
    SkillCategory,
    SkillMetadata,
    SkillContext,
    SkillResult,
)

logger = logging.getLogger(__name__)


@dataclass
class SkillRegistry:
    """
    技能注册中心

    负责管理所有技能的注册、发现和检索。
    支持依赖管理和拓扑排序。
    """

    _skills: Dict[str, BaseSkill] = field(default_factory=dict)
    _skill_metadata: Dict[str, SkillMetadata] = field(default_factory=dict)
    _initialized: bool = False

    def register(self, skill: BaseSkill) -> None:
        """
        注册技能

        Args:
            skill: BaseSkill 实例
        """
        name = skill.metadata.name
        if name in self._skills:
            logger.warning(f"技能 {name} 已存在，将被覆盖")

        self._skills[name] = skill
        self._skill_metadata[name] = skill.metadata
        logger.debug(f"注册技能: {name} ({skill.skill_type.value})")

    def unregister(self, name: str) -> bool:
        """注销技能"""
        if name in self._skills:
            del self._skills[name]
            del self._skill_metadata[name]
            return True
        return False

    def get(self, name: str) -> Optional[BaseSkill]:
        """获取技能"""
        return self._skills.get(name)

    def get_metadata(self, name: str) -> Optional[SkillMetadata]:
        """获取技能元数据"""
        return self._skill_metadata.get(name)

    def has(self, name: str) -> bool:
        """检查技能是否存在"""
        return name in self._skills

    def list_all(self) -> List[BaseSkill]:
        """列出所有技能"""
        return list(self._skills.values())

    def list_by_type(self, skill_type: SkillType) -> List[BaseSkill]:
        """按类型列出技能"""
        return [
            s for s in self._skills.values()
            if s.skill_type == skill_type
        ]

    def list_by_category(self, category: SkillCategory) -> List[BaseSkill]:
        """按分类列出技能"""
        return [
            s for s in self._skills.values()
            if s.metadata.category == category
        ]

    def list_by_tags(self, tags: List[str], match_any: bool = True) -> List[BaseSkill]:
        """
        按标签列出技能

        Args:
            tags: 标签列表
            match_any: True 表示任一匹配，False 表示全部匹配
        """
        result = []
        for skill in self._skills.values():
            skill_tags = set(skill.metadata.tags)
            search_tags = set(tags)

            if match_any:
                if skill_tags & search_tags:  # 有交集
                    result.append(skill)
            else:
                if search_tags.issubset(skill_tags):  # 全部包含
                    result.append(skill)

        return result

    def search(self, query: str) -> List[BaseSkill]:
        """
        搜索技能

        在名称、描述、标签中搜索
        """
        query = query.lower()
        results = []

        for skill in self._skills.values():
            meta = skill.metadata
            if (
                query in meta.name.lower() or
                query in meta.display_name.lower() or
                query in meta.description.lower() or
                any(query in tag.lower() for tag in meta.tags)
            ):
                results.append(skill)

        return results

    def get_dependencies(self, name: str) -> List[str]:
        """获取技能的直接依赖"""
        meta = self._skill_metadata.get(name)
        return meta.dependencies if meta else []

    def get_all_dependencies(self, name: str) -> Set[str]:
        """
        获取技能的所有依赖（递归）

        Returns:
            所有依赖的技能名称集合
        """
        visited = set()
        self._collect_dependencies(name, visited)
        visited.discard(name)  # 移除自身
        return visited

    def _collect_dependencies(self, name: str, visited: Set[str]) -> None:
        """递归收集依赖"""
        if name in visited:
            return
        visited.add(name)

        deps = self.get_dependencies(name)
        for dep in deps:
            self._collect_dependencies(dep, visited)

    def get_execution_order(self, skill_names: List[str]) -> List[str]:
        """
        计算技能执行顺序（拓扑排序）

        确保依赖的技能先执行。

        Args:
            skill_names: 需要执行的技能列表

        Returns:
            排序后的技能名称列表
        """
        # 收集所有需要的技能（包括依赖）
        all_skills = set()
        for name in skill_names:
            all_skills.add(name)
            all_skills.update(self.get_all_dependencies(name))

        # 构建依赖图
        in_degree = {name: 0 for name in all_skills}
        graph = {name: [] for name in all_skills}

        for name in all_skills:
            deps = self.get_dependencies(name)
            for dep in deps:
                if dep in all_skills:
                    graph[dep].append(name)
                    in_degree[name] += 1

        # 拓扑排序（Kahn 算法）
        queue = [name for name, degree in in_degree.items() if degree == 0]
        result = []

        while queue:
            # 按优先级排序
            queue.sort(key=lambda n: self._skill_metadata.get(n, None) and
                       self._skill_metadata[n].priority.value or 99)
            current = queue.pop(0)
            result.append(current)

            for neighbor in graph[current]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        if len(result) != len(all_skills):
            # 存在循环依赖
            missing = all_skills - set(result)
            logger.warning(f"检测到循环依赖，跳过: {missing}")

        return result

    def validate_dependencies(self, skill_names: List[str]) -> List[str]:
        """
        验证依赖是否满足

        Returns:
            缺失的依赖列表
        """
        missing = []
        for name in skill_names:
            deps = self.get_all_dependencies(name)
            for dep in deps:
                if dep not in self._skills:
                    missing.append(f"{name} -> {dep}")
        return missing

    def get_skill_catalog(self) -> str:
        """
        生成技能目录（Markdown 格式）

        供 LLM 参考
        """
        lines = ["# 技能目录\n"]

        for category in SkillCategory:
            skills = self.list_by_category(category)
            if skills:
                lines.append(f"## {category.value.title()}\n")
                for skill in skills:
                    meta = skill.metadata
                    lines.append(f"### {meta.name}\n")
                    lines.append(f"{meta.description}\n")
                    if meta.dependencies:
                        lines.append(f"**依赖**: {', '.join(meta.dependencies)}\n")
                    lines.append("")
                    if meta.inputs:
                        lines.append("**输入**:")
                        for inp in meta.inputs:
                            req = "必需" if inp.required else "可选"
                            lines.append(f"- `{inp.name}` ({inp.type}, {req}): {inp.description}")
                        lines.append("")
                    if meta.outputs:
                        lines.append("**输出**:")
                        for out in meta.outputs:
                            lines.append(f"- `{out.name}` ({out.type}): {out.description}")
                        lines.append("")

        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """导出为字典"""
        return {
            "skills": {name: meta.to_dict() for name, meta in self._skill_metadata.items()},
            "count": len(self._skills),
        }

    def clear(self) -> None:
        """清空注册表"""
        self._skills.clear()
        self._skill_metadata.clear()
        self._initialized = False

    @property
    def count(self) -> int:
        """技能数量"""
        return len(self._skills)


# 全局注册表实例
_global_registry: Optional[SkillRegistry] = None


def get_registry() -> SkillRegistry:
    """获取全局技能注册表"""
    global _global_registry
    if _global_registry is None:
        _global_registry = SkillRegistry()
    return _global_registry


def reset_registry() -> None:
    """重置全局注册表"""
    global _global_registry
    _global_registry = None


def register_skill(skill: BaseSkill) -> None:
    """注册技能到全局注册表"""
    get_registry().register(skill)


def skill_decorator(cls: Type[BaseSkill]) -> Type[BaseSkill]:
    """
    技能注册装饰器

    用法:
        @skill_decorator
        class MySkill(BaseSkill):
            ...
    """
    instance = cls()
    register_skill(instance)
    return cls
