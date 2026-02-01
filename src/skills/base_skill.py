"""
Skill 基类定义

定义所有 Python Skills 的基类和接口规范
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Type
from enum import Enum
import logging
import time

logger = logging.getLogger(__name__)


class SkillCategory(Enum):
    """技能分类"""
    COMPUTE = "compute"           # 计算类：MFU、带宽效率等
    MEMORY = "memory"             # 内存类：碎片、OOM 诊断等
    COMMUNICATION = "communication"  # 通信类：集合操作、拓扑等
    DIAGNOSIS = "diagnosis"       # 诊断类：异常检测、根因分析等
    OPTIMIZATION = "optimization"  # 优化类：策略建议等


class SkillPriority(Enum):
    """技能优先级"""
    CRITICAL = 1    # 关键技能，优先执行
    HIGH = 2        # 高优先级
    NORMAL = 3      # 正常优先级
    LOW = 4         # 低优先级


@dataclass
class SkillInput:
    """技能输入规范"""
    name: str                     # 输入参数名
    type: str                     # 参数类型 (str, int, float, dict, list, DataFrame)
    required: bool = True         # 是否必需
    default: Any = None           # 默认值
    description: str = ""         # 参数描述


@dataclass
class SkillOutput:
    """技能输出规范"""
    name: str                     # 输出字段名
    type: str                     # 输出类型
    description: str = ""         # 字段描述


@dataclass
class SkillResult:
    """技能执行结果"""
    skill_name: str               # 技能名称
    success: bool                 # 是否成功
    data: Dict[str, Any] = field(default_factory=dict)  # 结果数据
    summary: str = ""             # 结果摘要（人类可读）
    suggestions: List[str] = field(default_factory=list)  # 优化建议
    confidence: float = 1.0       # 结果置信度 [0, 1]
    execution_time_ms: float = 0  # 执行时间
    error: Optional[str] = None   # 错误信息
    
    def to_prompt_text(self) -> str:
        """转换为 LLM 可读的文本格式"""
        lines = [
            f"## {self.skill_name} 执行结果",
            f"- 状态: {'成功' if self.success else '失败'}",
            f"- 置信度: {self.confidence:.0%}",
        ]
        
        if self.summary:
            lines.append(f"- 摘要: {self.summary}")
        
        if self.data:
            lines.append("- 数据:")
            for key, value in self.data.items():
                if isinstance(value, float):
                    lines.append(f"  - {key}: {value:.4f}")
                else:
                    lines.append(f"  - {key}: {value}")
        
        if self.suggestions:
            lines.append("- 建议:")
            for i, suggestion in enumerate(self.suggestions, 1):
                lines.append(f"  {i}. {suggestion}")
        
        if self.error:
            lines.append(f"- 错误: {self.error}")
        
        return "\n".join(lines)


@dataclass
class SkillMetadata:
    """技能元数据"""
    name: str                     # 技能名称（唯一标识）
    display_name: str             # 显示名称
    description: str              # 技能描述
    category: SkillCategory       # 技能分类
    priority: SkillPriority = SkillPriority.NORMAL
    version: str = "1.0.0"
    author: str = ""
    inputs: List[SkillInput] = field(default_factory=list)
    outputs: List[SkillOutput] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)  # 用于搜索的标签
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "name": self.name,
            "display_name": self.display_name,
            "description": self.description,
            "category": self.category.value,
            "priority": self.priority.value,
            "version": self.version,
            "inputs": [
                {"name": i.name, "type": i.type, "required": i.required, "description": i.description}
                for i in self.inputs
            ],
            "outputs": [
                {"name": o.name, "type": o.type, "description": o.description}
                for o in self.outputs
            ],
            "tags": self.tags,
        }


class BaseSkill(ABC):
    """
    技能基类
    
    所有 Python Skills 必须继承此类并实现 execute 方法
    """
    
    @property
    @abstractmethod
    def metadata(self) -> SkillMetadata:
        """返回技能元数据"""
        pass
    
    @abstractmethod
    def execute(self, **kwargs) -> SkillResult:
        """
        执行技能
        
        Args:
            **kwargs: 根据 metadata.inputs 定义的输入参数
            
        Returns:
            SkillResult: 执行结果
        """
        pass
    
    def validate_inputs(self, **kwargs) -> Optional[str]:
        """
        验证输入参数
        
        Returns:
            错误信息，如果验证通过返回 None
        """
        for input_spec in self.metadata.inputs:
            if input_spec.required and input_spec.name not in kwargs:
                return f"缺少必需参数: {input_spec.name}"
            
            if input_spec.name in kwargs:
                value = kwargs[input_spec.name]
                # 基本类型检查
                if input_spec.type == "int" and not isinstance(value, int):
                    return f"参数 {input_spec.name} 应为 int 类型"
                elif input_spec.type == "float" and not isinstance(value, (int, float)):
                    return f"参数 {input_spec.name} 应为 float 类型"
                elif input_spec.type == "str" and not isinstance(value, str):
                    return f"参数 {input_spec.name} 应为 str 类型"
                elif input_spec.type == "dict" and not isinstance(value, dict):
                    return f"参数 {input_spec.name} 应为 dict 类型"
                elif input_spec.type == "list" and not isinstance(value, list):
                    return f"参数 {input_spec.name} 应为 list 类型"
        
        return None
    
    def run(self, **kwargs) -> SkillResult:
        """
        运行技能（带验证和计时）
        
        这是外部调用的主入口
        """
        start_time = time.time()
        
        # 验证输入
        error = self.validate_inputs(**kwargs)
        if error:
            return SkillResult(
                skill_name=self.metadata.name,
                success=False,
                error=error,
            )
        
        # 填充默认值
        for input_spec in self.metadata.inputs:
            if input_spec.name not in kwargs and input_spec.default is not None:
                kwargs[input_spec.name] = input_spec.default
        
        try:
            result = self.execute(**kwargs)
            result.execution_time_ms = (time.time() - start_time) * 1000
            return result
        except Exception as e:
            logger.exception(f"技能 {self.metadata.name} 执行失败")
            return SkillResult(
                skill_name=self.metadata.name,
                success=False,
                error=str(e),
                execution_time_ms=(time.time() - start_time) * 1000,
            )
    
    def __repr__(self) -> str:
        return f"<Skill: {self.metadata.name} ({self.metadata.category.value})>"


class PromptSkill:
    """
    Prompt 技能
    
    基于 Markdown/文本的指导性技能，用于引导 LLM 行为
    """
    
    def __init__(
        self,
        name: str,
        display_name: str,
        description: str,
        prompt_template: str,
        category: SkillCategory = SkillCategory.DIAGNOSIS,
        tags: List[str] = None,
    ):
        self.name = name
        self.display_name = display_name
        self.description = description
        self.prompt_template = prompt_template
        self.category = category
        self.tags = tags or []
    
    def render(self, **kwargs) -> str:
        """
        渲染 prompt 模板
        
        Args:
            **kwargs: 模板变量
            
        Returns:
            渲染后的 prompt 文本
        """
        try:
            return self.prompt_template.format(**kwargs)
        except KeyError as e:
            return self.prompt_template  # 如果缺少变量，返回原始模板
    
    @property
    def metadata(self) -> Dict[str, Any]:
        """返回元数据"""
        return {
            "name": self.name,
            "display_name": self.display_name,
            "description": self.description,
            "category": self.category.value,
            "type": "prompt",
            "tags": self.tags,
        }
    
    def __repr__(self) -> str:
        return f"<PromptSkill: {self.name}>"
