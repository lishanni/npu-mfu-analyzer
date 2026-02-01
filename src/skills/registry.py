"""
Skill 注册表

管理所有技能的注册、发现和检索
"""

from typing import Dict, List, Optional, Type, Union
from dataclasses import dataclass, field
import logging
import importlib
import pkgutil
from pathlib import Path

from .base_skill import (
    BaseSkill, 
    PromptSkill, 
    SkillCategory, 
    SkillMetadata,
)

logger = logging.getLogger(__name__)


@dataclass
class SkillRegistry:
    """
    技能注册表
    
    负责管理所有技能的注册和检索
    """
    
    _python_skills: Dict[str, BaseSkill] = field(default_factory=dict)
    _prompt_skills: Dict[str, PromptSkill] = field(default_factory=dict)
    _initialized: bool = False
    
    def register_python_skill(self, skill: BaseSkill) -> None:
        """
        注册 Python 技能
        
        Args:
            skill: BaseSkill 实例
        """
        name = skill.metadata.name
        if name in self._python_skills:
            logger.warning(f"技能 {name} 已存在，将被覆盖")
        
        self._python_skills[name] = skill
        logger.debug(f"注册 Python 技能: {name}")
    
    def register_prompt_skill(self, skill: PromptSkill) -> None:
        """
        注册 Prompt 技能
        
        Args:
            skill: PromptSkill 实例
        """
        name = skill.name
        if name in self._prompt_skills:
            logger.warning(f"Prompt 技能 {name} 已存在，将被覆盖")
        
        self._prompt_skills[name] = skill
        logger.debug(f"注册 Prompt 技能: {name}")
    
    def get_python_skill(self, name: str) -> Optional[BaseSkill]:
        """获取 Python 技能"""
        return self._python_skills.get(name)
    
    def get_prompt_skill(self, name: str) -> Optional[PromptSkill]:
        """获取 Prompt 技能"""
        return self._prompt_skills.get(name)
    
    def get_skill(self, name: str) -> Optional[Union[BaseSkill, PromptSkill]]:
        """获取任意类型的技能"""
        return self._python_skills.get(name) or self._prompt_skills.get(name)
    
    def list_python_skills(
        self, 
        category: Optional[SkillCategory] = None,
        tags: Optional[List[str]] = None,
    ) -> List[BaseSkill]:
        """
        列出 Python 技能
        
        Args:
            category: 按分类过滤
            tags: 按标签过滤（任一匹配）
            
        Returns:
            技能列表
        """
        skills = list(self._python_skills.values())
        
        if category:
            skills = [s for s in skills if s.metadata.category == category]
        
        if tags:
            skills = [
                s for s in skills 
                if any(tag in s.metadata.tags for tag in tags)
            ]
        
        # 按优先级排序
        skills.sort(key=lambda s: s.metadata.priority.value)
        
        return skills
    
    def list_prompt_skills(
        self,
        category: Optional[SkillCategory] = None,
        tags: Optional[List[str]] = None,
    ) -> List[PromptSkill]:
        """列出 Prompt 技能"""
        skills = list(self._prompt_skills.values())
        
        if category:
            skills = [s for s in skills if s.category == category]
        
        if tags:
            skills = [
                s for s in skills
                if any(tag in s.tags for tag in tags)
            ]
        
        return skills
    
    def list_all_skills(self) -> Dict[str, List]:
        """列出所有技能"""
        return {
            "python_skills": list(self._python_skills.values()),
            "prompt_skills": list(self._prompt_skills.values()),
        }
    
    def search_skills(
        self, 
        query: str,
        skill_type: Optional[str] = None,  # "python" or "prompt" or None (all)
    ) -> List[Union[BaseSkill, PromptSkill]]:
        """
        搜索技能
        
        Args:
            query: 搜索关键词
            skill_type: 技能类型过滤
            
        Returns:
            匹配的技能列表
        """
        query = query.lower()
        results = []
        
        # 搜索 Python 技能
        if skill_type in (None, "python"):
            for skill in self._python_skills.values():
                meta = skill.metadata
                if (
                    query in meta.name.lower() or
                    query in meta.display_name.lower() or
                    query in meta.description.lower() or
                    any(query in tag.lower() for tag in meta.tags)
                ):
                    results.append(skill)
        
        # 搜索 Prompt 技能
        if skill_type in (None, "prompt"):
            for skill in self._prompt_skills.values():
                if (
                    query in skill.name.lower() or
                    query in skill.display_name.lower() or
                    query in skill.description.lower() or
                    any(query in tag.lower() for tag in skill.tags)
                ):
                    results.append(skill)
        
        return results
    
    def get_skill_catalog(self) -> str:
        """
        生成技能目录文本（供 LLM 参考）
        
        Returns:
            技能目录的 Markdown 格式文本
        """
        lines = ["# 可用技能目录\n"]
        
        # Python 技能
        lines.append("## Python 技能（精确计算）\n")
        for category in SkillCategory:
            category_skills = self.list_python_skills(category=category)
            if category_skills:
                lines.append(f"### {category.value.title()}\n")
                for skill in category_skills:
                    meta = skill.metadata
                    lines.append(f"- **{meta.name}**: {meta.description}")
                    if meta.inputs:
                        inputs_str = ", ".join(
                            f"`{i.name}` ({i.type})" 
                            for i in meta.inputs
                        )
                        lines.append(f"  - 输入: {inputs_str}")
                lines.append("")
        
        # Prompt 技能
        lines.append("## Prompt 技能（推理指导）\n")
        for skill in self._prompt_skills.values():
            lines.append(f"- **{skill.name}**: {skill.description}")
        
        return "\n".join(lines)
    
    def clear(self) -> None:
        """清空注册表"""
        self._python_skills.clear()
        self._prompt_skills.clear()
        self._initialized = False
    
    @property
    def python_skill_count(self) -> int:
        return len(self._python_skills)
    
    @property
    def prompt_skill_count(self) -> int:
        return len(self._prompt_skills)
    
    @property
    def total_skill_count(self) -> int:
        return self.python_skill_count + self.prompt_skill_count


# 全局注册表实例
_global_registry: Optional[SkillRegistry] = None


def get_registry() -> SkillRegistry:
    """获取全局技能注册表"""
    global _global_registry
    if _global_registry is None:
        _global_registry = SkillRegistry()
    return _global_registry


def register_skill(skill: Union[BaseSkill, PromptSkill]) -> None:
    """
    注册技能到全局注册表
    
    装饰器或直接调用均可使用
    """
    registry = get_registry()
    if isinstance(skill, BaseSkill):
        registry.register_python_skill(skill)
    elif isinstance(skill, PromptSkill):
        registry.register_prompt_skill(skill)
    else:
        raise TypeError(f"不支持的技能类型: {type(skill)}")


def skill(cls: Type[BaseSkill]) -> Type[BaseSkill]:
    """
    技能注册装饰器
    
    用法:
        @skill
        class MySkill(BaseSkill):
            ...
    """
    instance = cls()
    register_skill(instance)
    return cls


def auto_discover_skills(package_path: str = None) -> int:
    """
    自动发现并注册技能
    
    Args:
        package_path: 技能包路径，默认为 src/skills/python_skills
        
    Returns:
        发现的技能数量
    """
    if package_path is None:
        package_path = Path(__file__).parent / "python_skills"
    else:
        package_path = Path(package_path)
    
    if not package_path.exists():
        logger.warning(f"技能路径不存在: {package_path}")
        return 0
    
    count = 0
    
    # 遍历所有 Python 文件
    for py_file in package_path.glob("*.py"):
        if py_file.name.startswith("_"):
            continue
        
        module_name = py_file.stem
        try:
            # 动态导入模块
            spec = importlib.util.spec_from_file_location(
                f"skills.python_skills.{module_name}",
                py_file,
            )
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                
                # 查找 BaseSkill 子类
                for name in dir(module):
                    obj = getattr(module, name)
                    if (
                        isinstance(obj, type) and
                        issubclass(obj, BaseSkill) and
                        obj is not BaseSkill
                    ):
                        try:
                            instance = obj()
                            register_skill(instance)
                            count += 1
                            logger.info(f"自动注册技能: {instance.metadata.name}")
                        except Exception as e:
                            logger.error(f"实例化技能 {name} 失败: {e}")
                            
        except Exception as e:
            logger.error(f"加载模块 {module_name} 失败: {e}")
    
    return count
