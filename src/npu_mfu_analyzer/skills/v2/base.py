"""
Skills v2 基础定义

定义统一的技能接口、上下文和结果。
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Union
from enum import Enum
import logging
import time

logger = logging.getLogger(__name__)


class SkillType(Enum):
    """
    技能类型

    用于区分技能的执行方式和依赖关系
    """
    COMPUTE = "compute"        # 精确计算（不调用 LLM）
    DIAGNOSE = "diagnose"      # 规则诊断（规则引擎）
    ANALYSIS = "analysis"      # 深度分析（可能调用 LLM）
    REASONING = "reasoning"    # 推理（完全依赖 LLM）
    GENERATE = "generate"      # 代码生成


class SkillCategory(Enum):
    """
    技能分类

    用于组织技能和过滤
    """
    COMPUTE = "compute"              # 计算类：MFU、带宽效率等
    MEMORY = "memory"                # 内存类：碎片、OOM 诊断等
    COMMUNICATION = "communication"   # 通信类：集合操作、拓扑等
    DIAGNOSIS = "diagnosis"          # 诊断类：异常检测、根因分析等
    OPTIMIZATION = "optimization"     # 优化类：策略建议等
    ANALYSIS = "analysis"            # 分析类：Timeline、Operator 等
    GENERATION = "generation"        # 生成类：代码生成等


class SkillPriority(Enum):
    """技能优先级"""
    CRITICAL = 1    # 关键技能，优先执行
    HIGH = 2        # 高优先级
    NORMAL = 3      # 正常优先级
    LOW = 4         # 低优先级


@dataclass
class SkillInput:
    """技能输入参数定义"""
    name: str                     # 参数名
    type: str                     # 类型 (str, int, float, dict, list, DataFrame, Any)
    required: bool = True         # 是否必需
    default: Any = None           # 默认值
    description: str = ""         # 描述
    source: str = "context"       # 数据来源: context, previous_result, user_input


@dataclass
class SkillOutput:
    """技能输出定义"""
    name: str                     # 输出字段名
    type: str                     # 类型
    description: str = ""         # 描述


@dataclass
class SkillMetadata:
    """
    技能元数据

    描述技能的基本信息、输入输出规范
    """
    name: str                           # 技能名称（唯一标识）
    display_name: str                   # 显示名称
    description: str                    # 描述
    skill_type: SkillType               # 技能类型
    category: SkillCategory             # 技能分类
    priority: SkillPriority = SkillPriority.NORMAL
    version: str = "1.0.0"
    author: str = ""
    tags: List[str] = field(default_factory=list)
    inputs: List[SkillInput] = field(default_factory=list)
    outputs: List[SkillOutput] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)  # 依赖的其他技能

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        # 处理 inputs（可能是 SkillInput 或 dict）
        inputs_list = []
        for i in self.inputs:
            if isinstance(i, dict):
                inputs_list.append(i)
            else:
                inputs_list.append({
                    "name": i.name, "type": i.type, "required": i.required,
                    "default": i.default, "description": i.description, "source": i.source
                })

        # 处理 outputs（可能是 SkillOutput 或 dict）
        outputs_list = []
        for o in self.outputs:
            if isinstance(o, dict):
                outputs_list.append(o)
            else:
                outputs_list.append({"name": o.name, "type": o.type, "description": o.description})

        return {
            "name": self.name,
            "display_name": self.display_name,
            "description": self.description,
            "skill_type": self.skill_type.value,
            "category": self.category.value,
            "priority": self.priority.value,
            "version": self.version,
            "author": self.author,
            "tags": self.tags,
            "inputs": inputs_list,
            "outputs": outputs_list,
            "dependencies": self.dependencies,
        }


@dataclass
class SkillContext:
    """
    技能执行上下文

    包含技能执行所需的所有数据和环境信息
    """
    # === Profiling 数据 ===
    profiling_summary: Any = None       # ProfilingSummary
    hardware_spec: Any = None           # NPUSpec
    pattern: Any = None                 # UniversalPattern
    host_device_chains: List[Any] = field(default_factory=list)  # HostDeviceChain 列表

    # === 依赖数据 ===
    previous_results: Dict[str, "SkillResult"] = field(default_factory=dict)

    # === 用户意图 ===
    user_intent: str = ""
    user_inputs: Dict[str, Any] = field(default_factory=dict)

    # === 对比分析专用 ===
    profiling_summary_b: Optional[Any] = None
    hardware_spec_b: Optional[Any] = None
    diff_result: Optional[Any] = None

    # === 执行配置 ===
    config: Dict[str, Any] = field(default_factory=dict)

    # === LLM 接口（可选） ===
    llm: Any = None

    def get_input(self, name: str, default: Any = None) -> Any:
        """
        获取输入参数

        优先级：user_inputs > previous_results > context 属性
        """
        if name in self.user_inputs:
            return self.user_inputs[name]
        if name in self.previous_results:
            return self.previous_results[name].data
        if hasattr(self, name):
            return getattr(self, name)
        return default

    def get_previous_result(self, skill_name: str) -> Optional["SkillResult"]:
        """获取前序技能的结果"""
        return self.previous_results.get(skill_name)

    def get_previous_data(self, skill_name: str, field: str = None, default: Any = None) -> Any:
        """获取前序技能的输出数据"""
        result = self.previous_results.get(skill_name)
        if result is None:
            return default
        if field is None:
            return result.data
        return result.data.get(field, default)


@dataclass
class SkillResult:
    """
    技能执行结果（统一格式）

    所有技能的输出都必须符合此格式
    """
    # === 基本信息 ===
    skill_name: str
    skill_type: SkillType
    success: bool

    # === 结构化输出 ===
    data: Dict[str, Any] = field(default_factory=dict)

    # === 人类可读输出 ===
    summary: str = ""
    details: List[str] = field(default_factory=list)

    # === 建议和行动项 ===
    recommendations: List[str] = field(default_factory=list)
    actions: List[Dict[str, Any]] = field(default_factory=list)  # 可执行的操作

    # === 诊断专用字段 ===
    root_cause: Optional[str] = None
    evidence: List[str] = field(default_factory=list)
    affected_operators: List[str] = field(default_factory=list)

    # === 元数据 ===
    confidence: float = 1.0           # 置信度 [0, 1]
    priority: str = "P2"              # 优先级 P0/P1/P2
    severity: str = "info"            # 严重度 critical/high/medium/low/info
    execution_time_ms: float = 0
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "skill_name": self.skill_name,
            "skill_type": self.skill_type.value,
            "success": self.success,
            "data": self.data,
            "summary": self.summary,
            "details": self.details,
            "recommendations": self.recommendations,
            "actions": self.actions,
            "root_cause": self.root_cause,
            "evidence": self.evidence,
            "affected_operators": self.affected_operators,
            "confidence": self.confidence,
            "priority": self.priority,
            "severity": self.severity,
            "execution_time_ms": self.execution_time_ms,
            "error": self.error,
        }

    def to_markdown(self) -> str:
        """转换为 Markdown 格式"""
        lines = [
            f"## {self.skill_name}",
            "",
            f"**状态**: {'✅ 成功' if self.success else '❌ 失败'}",
            f"**类型**: {self.skill_type.value}",
            f"**置信度**: {self.confidence:.0%}",
        ]

        if self.summary:
            lines.extend(["", f"**摘要**: {self.summary}"])

        if self.data:
            lines.extend(["", "### 数据"])
            for key, value in self.data.items():
                if isinstance(value, float):
                    lines.append(f"- `{key}`: {value:.4f}")
                elif isinstance(value, dict):
                    lines.append(f"- `{key}`: {value}")
                else:
                    lines.append(f"- `{key}`: {value}")

        if self.root_cause:
            lines.extend(["", f"**根因**: {self.root_cause}"])

        if self.evidence:
            lines.extend(["", "### 证据"])
            for e in self.evidence:
                lines.append(f"- {e}")

        if self.recommendations:
            lines.extend(["", "### 建议"])
            for i, r in enumerate(self.recommendations, 1):
                lines.append(f"{i}. {r}")

        if self.actions:
            lines.extend(["", "### 可执行操作"])
            for action in self.actions:
                lines.append(f"- **{action.get('type', 'action')}**: {action.get('description', '')}")

        if self.error:
            lines.extend(["", f"**错误**: {self.error}"])

        return "\n".join(lines)

    def to_prompt_text(self) -> str:
        """转换为 LLM 可读的文本（兼容旧接口）"""
        return self.to_markdown()


class BaseSkill(ABC):
    """
    统一 Skill 基类

    所有技能（Python/Markdown/LLM）都必须继承此类。
    """

    def __init__(self):
        self._validate_metadata()

    @property
    @abstractmethod
    def metadata(self) -> SkillMetadata:
        """返回技能元数据"""
        pass

    @property
    def name(self) -> str:
        """技能名称"""
        return self.metadata.name

    @property
    def skill_type(self) -> SkillType:
        """技能类型"""
        return self.metadata.skill_type

    @property
    def dependencies(self) -> List[str]:
        """依赖的技能列表"""
        return self.metadata.dependencies

    @abstractmethod
    def execute(self, context: SkillContext) -> SkillResult:
        """
        执行技能

        Args:
            context: 执行上下文，包含所有必需数据

        Returns:
            SkillResult: 统一格式的结果
        """
        pass

    def can_execute(self, context: SkillContext) -> bool:
        """
        检查是否可以执行

        用于条件触发，如数据不足时跳过
        """
        return True

    def validate_inputs(self, context: SkillContext) -> Optional[str]:
        """
        验证输入

        Returns:
            错误信息，如果验证通过返回 None
        """
        for input_spec in self.metadata.inputs:
            if input_spec.required:
                value = context.get_input(input_spec.name)
                if value is None and input_spec.default is None:
                    return f"缺少必需参数: {input_spec.name}"
        return None

    def run(self, context: SkillContext) -> SkillResult:
        """
        运行技能（带验证和计时）

        这是外部调用的主入口
        """
        start_time = time.time()

        # 验证输入
        error = self.validate_inputs(context)
        if error:
            return SkillResult(
                skill_name=self.name,
                skill_type=self.skill_type,
                success=False,
                error=error,
            )

        # 检查是否可执行
        if not self.can_execute(context):
            return SkillResult(
                skill_name=self.name,
                skill_type=self.skill_type,
                success=False,
                error="执行条件不满足",
            )

        try:
            result = self.execute(context)
            result.execution_time_ms = (time.time() - start_time) * 1000
            return result
        except Exception as e:
            logger.exception(f"技能 {self.name} 执行失败")
            return SkillResult(
                skill_name=self.name,
                skill_type=self.skill_type,
                success=False,
                error=str(e),
                execution_time_ms=(time.time() - start_time) * 1000,
            )

    def _validate_metadata(self):
        """验证元数据"""
        if not self.metadata.name:
            raise ValueError("技能名称不能为空")
        if not isinstance(self.metadata.skill_type, SkillType):
            raise ValueError(f"无效的技能类型: {self.metadata.skill_type}")

    def __repr__(self) -> str:
        return f"<Skill: {self.name} ({self.skill_type.value})>"


class PythonSkill(BaseSkill):
    """
    Python 技能基类

    用于精确计算，不调用 LLM
    """

    @property
    def skill_type(self) -> SkillType:
        return SkillType.COMPUTE


class DiagnoseSkill(BaseSkill):
    """
    诊断技能基类

    用于规则引擎诊断
    """

    @property
    def skill_type(self) -> SkillType:
        return SkillType.DIAGNOSE


class AnalysisSkill(BaseSkill):
    """
    分析技能基类

    用于深度分析，可能需要调用 LLM
    """

    @property
    def skill_type(self) -> SkillType:
        return SkillType.ANALYSIS


class ReasoningSkill(BaseSkill):
    """
    推理技能基类

    完全依赖 LLM 进行推理
    """

    @property
    def skill_type(self) -> SkillType:
        return SkillType.REASONING


class GenerateSkill(BaseSkill):
    """
    生成技能基类

    用于代码生成
    """

    @property
    def skill_type(self) -> SkillType:
        return SkillType.GENERATE