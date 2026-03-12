"""
Agent 基类

定义 Agent 的通用接口和行为。
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field

from npu_mfu_analyzer.llm.llm_interface import LLMInterface, Message

logger = logging.getLogger(__name__)


@dataclass
class AgentMessage:
    """Agent 消息"""
    role: str  # "user", "assistant", "system"
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_llm_message(self) -> Message:
        return Message(role=self.role, content=self.content)


@dataclass
class AnalysisResult:
    """分析结果"""
    agent_name: str
    success: bool
    summary: str
    details: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)
    raw_response: Optional[str] = None
    error: Optional[str] = None


class BaseAgent(ABC):
    """
    Agent 基类
    
    所有专业化 Agent 都继承此类。
    """
    
    def __init__(
        self, 
        name: str, 
        llm: LLMInterface,
        system_prompt: str = "",
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Args:
            name: Agent 名称
            llm: LLM 接口实例
            system_prompt: 系统提示词
            config: 额外配置
        """
        self.name = name
        self.llm = llm
        self.system_prompt = system_prompt
        self.config = config or {}
        self.context: List[AgentMessage] = []
        self._initialized = False
    
    async def initialize(self):
        """初始化 Agent（可选覆盖）"""
        self._initialized = True
        logger.info(f"Agent [{self.name}] initialized")
    
    @abstractmethod
    async def analyze(self, data: Dict[str, Any]) -> AnalysisResult:
        """
        执行分析任务
        
        Args:
            data: 输入数据
            
        Returns:
            AnalysisResult
        """
        pass
    
    @abstractmethod
    def get_prompt_template(self) -> str:
        """获取该 Agent 的 Prompt 模板"""
        pass
    
    async def call_llm(self, prompt: str, include_context: bool = True) -> str:
        """
        调用 LLM
        
        Args:
            prompt: 用户输入
            include_context: 是否包含上下文
            
        Returns:
            LLM 响应文本
        """
        messages = []
        
        # 添加系统提示
        if self.system_prompt:
            messages.append(Message(role="system", content=self.system_prompt))
        
        # 添加历史上下文
        if include_context:
            for msg in self.context[-10:]:  # 最多保留最近10条
                messages.append(msg.to_llm_message())
        
        # 添加当前输入
        messages.append(Message(role="user", content=prompt))
        
        # 调用 LLM
        response = await self.llm.complete(messages)
        
        # 更新上下文
        self.context.append(AgentMessage(role="user", content=prompt))
        self.context.append(AgentMessage(role="assistant", content=response.content))
        
        return response.content
    
    def format_prompt(self, template: str, **kwargs) -> str:
        """格式化 Prompt 模板"""
        try:
            return template.format(**kwargs)
        except KeyError as e:
            logger.warning(f"Missing key in prompt template: {e}")
            return template
    
    def clear_context(self):
        """清空上下文"""
        self.context.clear()
    
    def __repr__(self):
        return f"<{self.__class__.__name__}(name={self.name})>"

    def _extract_recommendations(self, response: str) -> List[str]:
        """
        从 LLM 响应中提取建议

        Args:
            response: LLM 响应文本

        Returns:
            建议列表
        """
        recommendations = []

        # 简单的建议提取逻辑
        lines = response.split("\n")
        in_recommendation = False

        for line in lines:
            if "建议" in line or "优化" in line:
                in_recommendation = True
            if in_recommendation and line.strip().startswith(("-", "*", "1", "2", "3", "4", "5")):
                recommendations.append(line.strip().lstrip("-*0123456789. "))

        return recommendations[:10]  # 最多10条


class TimelineAgent(BaseAgent):
    """Timeline 分析 Agent"""
    
    PROMPT_TEMPLATE = """
分析以下 Timeline 数据：

## 数据摘要
{data_summary}

## 分析任务
1. 计算时间组成占比（计算/通信/空闲）
2. 分析通信掩盖率是否合理
3. 识别主要性能瓶颈
4. 给出优化建议

请给出详细的分析结论和建议。
"""
    
    def __init__(self, llm: LLMInterface, config: Optional[Dict[str, Any]] = None):
        from npu_mfu_analyzer.llm.prompts import TIMELINE_ANALYSIS_SYSTEM
        super().__init__(
            name="TimelineAgent",
            llm=llm,
            system_prompt=TIMELINE_ANALYSIS_SYSTEM,
            config=config
        )
    
    def get_prompt_template(self) -> str:
        return self.PROMPT_TEMPLATE
    
    async def analyze(self, data: Dict[str, Any]) -> AnalysisResult:
        """分析 Timeline 数据"""
        try:
            # 构建数据摘要
            data_summary = self._format_data_summary(data)
            
            # 生成 Prompt
            prompt = self.format_prompt(self.PROMPT_TEMPLATE, data_summary=data_summary)
            
            # 调用 LLM
            response = await self.call_llm(prompt)
            
            return AnalysisResult(
                agent_name=self.name,
                success=True,
                summary="Timeline 分析完成",
                details=data,
                recommendations=self._extract_recommendations(response),
                raw_response=response,
            )
            
        except Exception as e:
            logger.error(f"Timeline analysis failed: {e}")
            return AnalysisResult(
                agent_name=self.name,
                success=False,
                summary="Timeline 分析失败",
                error=str(e),
            )
    
    def _format_data_summary(self, data: Dict[str, Any]) -> str:
        """格式化数据摘要"""
        lines = []
        
        if "avg_compute_time" in data:
            lines.append(f"- 平均计算时间: {data['avg_compute_time'] / 1000:.2f} ms")
        if "avg_comm_time" in data:
            lines.append(f"- 平均通信时间: {data['avg_comm_time'] / 1000:.2f} ms")
        if "avg_free_time" in data:
            lines.append(f"- 平均空闲时间: {data['avg_free_time'] / 1000:.2f} ms")
        if "overlap_metrics" in data:
            om = data["overlap_metrics"]
            if isinstance(om, dict):
                lines.append(f"- 通信掩盖率: {om.get('overlap_ratio', 0):.1f}%")
        if "top_operators" in data:
            lines.append("\n### Top 耗时算子")
            for op in data["top_operators"][:5]:
                lines.append(f"- {op.get('name', 'unknown')}: {op.get('dur', 0) / 1000:.2f} ms")
        
        return "\n".join(lines) if lines else "无数据"


class OperatorAgent(BaseAgent):
    """算子分析 Agent"""
    
    PROMPT_TEMPLATE = """
分析以下算子性能数据：

## 数据摘要
{data_summary}

## 分析任务
1. 计算整体 MFU
2. 识别低效算子
3. 分析算子性能瓶颈
4. 给出算子优化建议

请给出详细的分析结论和建议。
"""
    
    def __init__(self, llm: LLMInterface, config: Optional[Dict[str, Any]] = None):
        from npu_mfu_analyzer.llm.prompts import MFU_ANALYSIS_SYSTEM
        super().__init__(
            name="OperatorAgent",
            llm=llm,
            system_prompt=MFU_ANALYSIS_SYSTEM,
            config=config
        )
    
    def get_prompt_template(self) -> str:
        return self.PROMPT_TEMPLATE
    
    async def analyze(self, data: Dict[str, Any]) -> AnalysisResult:
        """分析算子数据"""
        try:
            data_summary = self._format_data_summary(data)
            prompt = self.format_prompt(self.PROMPT_TEMPLATE, data_summary=data_summary)
            response = await self.call_llm(prompt)
            
            return AnalysisResult(
                agent_name=self.name,
                success=True,
                summary="算子分析完成",
                details=data,
                raw_response=response,
            )
            
        except Exception as e:
            logger.error(f"Operator analysis failed: {e}")
            return AnalysisResult(
                agent_name=self.name,
                success=False,
                summary="算子分析失败",
                error=str(e),
            )
    
    def _format_data_summary(self, data: Dict[str, Any]) -> str:
        """格式化数据摘要"""
        if "to_prompt_text" in dir(data):
            return data.to_prompt_text()
        return str(data)
