"""
Skills v2 加载器

支持从文件系统加载 Python Skills 和 Markdown Skills。
"""

from typing import Dict, List, Optional, Type, Any
from pathlib import Path
from dataclasses import dataclass, field
import logging
import importlib
import re
import yaml

from .base import (
    BaseSkill,
    SkillType,
    SkillCategory,
    SkillPriority,
    SkillMetadata,
    SkillInput,
    SkillOutput,
    SkillContext,
    SkillResult,
    PythonSkill,
    DiagnoseSkill,
    AnalysisSkill,
    ReasoningSkill,
    GenerateSkill,
)
from .registry import SkillRegistry

logger = logging.getLogger(__name__)


@dataclass
class MarkdownRule:
    """Markdown 规则定义"""
    name: str
    trigger_conditions: Dict[str, Any] = field(default_factory=dict)
    root_cause: str = ""
    evidence_patterns: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    priority: str = "P2"
    severity: str = "medium"
    metadata: Dict[str, Any] = field(default_factory=dict)


class MarkdownSkill(BaseSkill):
    """
    Markdown 技能

    从 Markdown 文件加载的规则型技能
    """

    def __init__(self, rule: MarkdownRule, file_path: Path):
        self.rule = rule
        self.file_path = file_path
        self._metadata = self._build_metadata()

    def _build_metadata(self) -> SkillMetadata:
        """构建元数据"""
        # 解析优先级
        priority_map = {
            "P0": SkillPriority.CRITICAL,
            "P1": SkillPriority.HIGH,
            "P2": SkillPriority.NORMAL,
            "critical": SkillPriority.CRITICAL,
            "high": SkillPriority.HIGH,
            "normal": SkillPriority.NORMAL,
            "low": SkillPriority.LOW,
        }
        priority = priority_map.get(self.rule.priority.lower() if isinstance(self.rule.priority, str) else self.rule.priority, SkillPriority.NORMAL)

        return SkillMetadata(
            name=self.rule.name,
            display_name=self.rule.name.replace("_", " ").title(),
            description=self.rule.root_cause[:100] if self.rule.root_cause else "",
            skill_type=SkillType.DIAGNOSE,
            category=SkillCategory.DIAGNOSIS,
            priority=priority,
            tags=self.rule.metadata.get("tags", []),
            dependencies=self.rule.metadata.get("dependencies", []),
        )

    @property
    def metadata(self) -> SkillMetadata:
        return self._metadata

    def check_trigger(self, context: SkillContext) -> bool:
        """检查触发条件"""
        conditions = self.rule.trigger_conditions
        if not conditions:
            return False

        # 检查各种触发条件
        for cond_type, cond_value in conditions.items():
            if cond_type == "small_operator_ratio":
                data = context.get_previous_data("classify_operators", "small_op_ratio", 0)
                if data < cond_value:
                    return False

            elif cond_type == "source_change":
                data = context.get_previous_data("classify_operators", "source_changes", [])
                if not data:
                    return False

            elif cond_type == "mfu_drop":
                mfu_a = context.get_previous_data("compute_mfu", "overall_mfu", 0)
                mfu_b = context.get_previous_data("compute_mfu_b", "overall_mfu", 0)
                if mfu_b >= mfu_a:
                    return False

        return True

    def execute(self, context: SkillContext) -> SkillResult:
        """执行诊断"""
        if not self.check_trigger(context):
            return SkillResult(
                skill_name=self.name,
                skill_type=SkillType.DIAGNOSE,
                success=False,
                error="触发条件不满足",
            )

        # 收集证据
        evidence = []
        for pattern in self.rule.evidence_patterns:
            # 尝试从上下文获取数据填充模式
            try:
                formatted = pattern.format(
                    context=context,
                    **context.previous_results,
                )
                evidence.append(formatted)
            except (KeyError, AttributeError):
                evidence.append(pattern)

        return SkillResult(
            skill_name=self.name,
            skill_type=SkillType.DIAGNOSE,
            success=True,
            root_cause=self.rule.root_cause,
            evidence=evidence,
            recommendations=self.rule.suggestions,
            priority=self.rule.priority,
            severity=self.rule.severity,
            confidence=0.8,  # 规则型诊断的默认置信度
        )


class SkillLoader:
    """
    技能加载器

    支持从文件系统自动发现和加载技能
    """

    def __init__(self, registry: SkillRegistry = None):
        self.registry = registry or SkillRegistry()

    def discover(self, skills_dir: Path) -> int:
        """
        自动发现并加载技能

        目录结构:
        skills/
        ├── compute/
        │   ├── mfu_skill.py
        │   └── bandwidth_skill.py
        ├── diagnose/
        │   ├── rules/
        │   │   ├── torch_compile_fusion.md
        │   │   └── communication_bottleneck.md
        │   └── diagnose_engine.py
        └── analysis/
            └── timeline_skill.py

        Returns:
            加载的技能数量
        """
        if not skills_dir.exists():
            logger.warning(f"技能目录不存在: {skills_dir}")
            return 0

        count = 0

        # 1. 加载 Python Skills
        for py_file in skills_dir.rglob("*_skill.py"):
            try:
                loaded = self._load_python_skill(py_file)
                count += loaded
            except Exception as e:
                logger.error(f"加载 Python 技能失败 {py_file}: {e}")

        # 2. 加载 Markdown Skills
        for md_file in skills_dir.rglob("*.md"):
            try:
                loaded = self._load_markdown_skill(md_file)
                count += loaded
            except Exception as e:
                logger.error(f"加载 Markdown 技能失败 {md_file}: {e}")

        logger.info(f"共加载 {count} 个技能")
        return count

    def _load_python_skill(self, py_file: Path) -> int:
        """加载 Python 技能文件"""
        count = 0

        # 动态导入模块
        spec = importlib.util.spec_from_file_location(
            f"skills.{py_file.stem}",
            py_file,
        )
        if not spec or not spec.loader:
            return 0

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # 查找 BaseSkill 子类
        for name in dir(module):
            obj = getattr(module, name)
            if (
                isinstance(obj, type) and
                issubclass(obj, BaseSkill) and
                obj is not BaseSkill and
                obj is not PythonSkill and
                obj is not DiagnoseSkill and
                obj is not AnalysisSkill and
                obj is not ReasoningSkill and
                obj is not GenerateSkill and
                obj is not MarkdownSkill
            ):
                try:
                    instance = obj()
                    self.registry.register(instance)
                    count += 1
                    logger.info(f"加载 Python 技能: {instance.name}")
                except Exception as e:
                    logger.error(f"实例化技能 {name} 失败: {e}")

        return count

    def _load_markdown_skill(self, md_file: Path) -> int:
        """加载 Markdown 技能文件"""
        content = md_file.read_text(encoding="utf-8")

        # 解析 YAML frontmatter
        frontmatter, body = self._parse_frontmatter(content)

        if not frontmatter:
            return 0

        # 构建 MarkdownRule
        rule = MarkdownRule(
            name=frontmatter.get("name", md_file.stem),
            trigger_conditions=self._parse_conditions(frontmatter.get("trigger_conditions", {})),
            root_cause=self._extract_section(body, "根因描述", "root_cause"),
            evidence_patterns=self._extract_list(body, "证据模式", "evidence"),
            suggestions=self._extract_list(body, "优化建议", "suggestions"),
            priority=frontmatter.get("priority", "P2"),
            severity=frontmatter.get("severity", "medium"),
            metadata=frontmatter,
        )

        # 创建并注册 MarkdownSkill
        skill = MarkdownSkill(rule, md_file)
        self.registry.register(skill)
        logger.info(f"加载 Markdown 技能: {skill.name}")

        return 1

    def _parse_frontmatter(self, content: str) -> tuple:
        """解析 YAML frontmatter"""
        pattern = r"^---\s*\n(.*?)\n---\s*\n(.*)$"
        match = re.match(pattern, content, re.DOTALL)

        if not match:
            return None, content

        try:
            frontmatter = yaml.safe_load(match.group(1))
            body = match.group(2)
            return frontmatter, body
        except yaml.YAMLError:
            return None, content

    def _parse_conditions(self, conditions: Any) -> Dict[str, Any]:
        """解析触发条件"""
        if isinstance(conditions, dict):
            return conditions
        if isinstance(conditions, str):
            # 尝试解析 YAML 格式
            try:
                return yaml.safe_load(conditions)
            except:
                return {}
        return {}

    def _extract_section(self, body: str, title: str, alt_key: str = None) -> str:
        """提取 Markdown 章节"""
        pattern = rf"##\s*{title}.*?\n(.*?)(?=\n##|\Z)"
        match = re.search(pattern, body, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return ""

    def _extract_list(self, body: str, title: str, alt_key: str = None) -> List[str]:
        """提取 Markdown 列表"""
        section = self._extract_section(body, title, alt_key)
        if not section:
            return []

        items = []
        for line in section.split("\n"):
            line = line.strip()
            if line.startswith("- ") or line.startswith("* "):
                items.append(line[2:].strip())
            elif re.match(r"^\d+\.\s", line):
                items.append(re.sub(r"^\d+\.\s*", "", line))

        return items


def load_skills_from_dir(skills_dir: Path, registry: SkillRegistry = None) -> int:
    """
    便捷函数：从目录加载技能

    Args:
        skills_dir: 技能目录
        registry: 注册表（可选）

    Returns:
        加载的技能数量
    """
    loader = SkillLoader(registry)
    return loader.discover(skills_dir)