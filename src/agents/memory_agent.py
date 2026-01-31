"""
Memory Agent

分析内存使用，识别峰值、碎片和 OOM 风险。
"""

import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field

import pandas as pd

from src.agents.base_agent import BaseAgent, AnalysisResult
from src.llm.llm_interface import LLMInterface
from src.llm.prompts import MEMORY_ANALYSIS_SYSTEM

logger = logging.getLogger(__name__)


@dataclass
class MemoryMetrics:
    """内存分析指标"""
    # 峰值内存（字节）
    peak_memory_bytes: float = 0.0
    peak_memory_mb: float = 0.0
    peak_memory_gb: float = 0.0
    
    # 内存使用分布
    model_memory_mb: float = 0.0       # 模型参数
    optimizer_memory_mb: float = 0.0   # 优化器状态
    activation_memory_mb: float = 0.0  # 激活值
    gradient_memory_mb: float = 0.0    # 梯度
    temp_memory_mb: float = 0.0        # 临时内存
    
    # 内存碎片
    fragmentation_ratio: float = 0.0   # 碎片率 (%)
    
    # 设备信息
    device_memory_gb: float = 64.0     # 设备总内存（默认 64GB）
    memory_utilization: float = 0.0    # 内存利用率 (%)
    
    # 风险评估
    oom_risk: str = "low"  # low, medium, high
    
    def calculate_utilization(self):
        """计算内存利用率"""
        if self.device_memory_gb > 0:
            self.memory_utilization = (self.peak_memory_gb / self.device_memory_gb) * 100
            
            # OOM 风险评估
            if self.memory_utilization > 90:
                self.oom_risk = "high"
            elif self.memory_utilization > 75:
                self.oom_risk = "medium"
            else:
                self.oom_risk = "low"
    
    def to_prompt_text(self) -> str:
        """转换为 LLM Prompt 格式"""
        lines = [
            "## 内存分析",
            "",
            "### 内存使用概况",
            f"- 峰值内存: {self.peak_memory_gb:.2f} GB ({self.peak_memory_mb:.0f} MB)",
            f"- 设备总内存: {self.device_memory_gb:.0f} GB",
            f"- 内存利用率: {self.memory_utilization:.1f}%",
            f"- OOM 风险: {self.oom_risk.upper()}",
            "",
            "### 内存分布",
        ]
        
        if self.model_memory_mb > 0:
            lines.append(f"- 模型参数: {self.model_memory_mb:.0f} MB")
        if self.optimizer_memory_mb > 0:
            lines.append(f"- 优化器状态: {self.optimizer_memory_mb:.0f} MB")
        if self.activation_memory_mb > 0:
            lines.append(f"- 激活值: {self.activation_memory_mb:.0f} MB")
        if self.gradient_memory_mb > 0:
            lines.append(f"- 梯度: {self.gradient_memory_mb:.0f} MB")
        if self.temp_memory_mb > 0:
            lines.append(f"- 临时内存: {self.temp_memory_mb:.0f} MB")
        
        if self.fragmentation_ratio > 0:
            lines.append("")
            lines.append(f"### 内存碎片")
            lines.append(f"- 碎片率: {self.fragmentation_ratio:.1f}%")
        
        return "\n".join(lines)


@dataclass
class MemoryEvent:
    """内存事件"""
    timestamp: float = 0.0
    operation: str = ""  # "allocate", "free"
    size_bytes: float = 0.0
    total_allocated: float = 0.0
    operator_name: str = ""


@dataclass
class MemoryAnalysisData:
    """内存分析数据"""
    metrics: Optional[MemoryMetrics] = None
    memory_events: List[MemoryEvent] = field(default_factory=list)
    top_memory_operators: List[Dict[str, Any]] = field(default_factory=list)
    memory_timeline: List[Dict[str, float]] = field(default_factory=list)
    
    def to_prompt_text(self) -> str:
        """转换为 LLM Prompt 格式"""
        lines = []
        
        if self.metrics:
            lines.append(self.metrics.to_prompt_text())
        
        if self.top_memory_operators:
            lines.append("")
            lines.append("### Top 内存消耗算子")
            for i, op in enumerate(self.top_memory_operators[:10], 1):
                name = op.get("name", "unknown")
                mem_mb = op.get("memory_mb", 0)
                lines.append(f"{i}. {name}: {mem_mb:.0f} MB")
        
        return "\n".join(lines)


class MemoryAgent(BaseAgent):
    """
    Memory Agent
    
    功能：
    1. 峰值内存分析
    2. 内存分布分析（模型/优化器/激活值/梯度）
    3. 内存碎片检测
    4. OOM 风险评估
    5. 优化建议（重计算、梯度累积、混合精度等）
    """
    
    PROMPT_TEMPLATE = """
你是昇腾 NPU 内存优化专家。分析以下内存使用数据：

{data_summary}

## 分析任务
1. **内存使用评估**：
   - 内存利用率是否合理？
   - 是否存在 OOM 风险？
   - 内存峰值出现在哪个阶段？

2. **内存分布分析**：
   - 各部分内存占比是否正常？
   - 激活值内存是否过大？（大模型常见问题）
   - 是否有内存泄漏迹象？

3. **内存碎片分析**：
   - 碎片率是否过高？（> 20% 需要关注）
   - 是否需要内存整理？

4. **优化建议**：
   - 激活值重计算（Activation Checkpointing）
   - 梯度累积（Gradient Accumulation）
   - 混合精度训练（AMP）
   - 优化器状态卸载（Optimizer Offload）
   - ZeRO 优化

请给出详细的分析结论和具体优化建议。
"""
    
    def __init__(
        self, 
        llm: LLMInterface, 
        config: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            name="MemoryAgent",
            llm=llm,
            system_prompt=MEMORY_ANALYSIS_SYSTEM,
            config=config
        )
    
    def get_prompt_template(self) -> str:
        return self.PROMPT_TEMPLATE
    
    async def analyze(self, data: Dict[str, Any]) -> AnalysisResult:
        """
        分析内存数据
        
        Args:
            data: 包含以下可选字段：
                - memory_df: 内存数据 DataFrame
                - peak_memory_mb: 峰值内存（MB）
                - memory_events: 内存事件列表
                - device_memory_gb: 设备总内存（GB）
                
        Returns:
            AnalysisResult
        """
        try:
            # 1. 准备分析数据
            analysis_data = self._prepare_analysis_data(data)
            
            # 2. 生成 Prompt 并调用 LLM
            prompt = self.format_prompt(
                self.PROMPT_TEMPLATE,
                data_summary=analysis_data.to_prompt_text()
            )
            response = await self.call_llm(prompt)
            
            # 3. 构建结果
            metrics = analysis_data.metrics
            peak_gb = metrics.peak_memory_gb if metrics else 0
            oom_risk = metrics.oom_risk if metrics else "unknown"
            
            return AnalysisResult(
                agent_name=self.name,
                success=True,
                summary=f"峰值内存: {peak_gb:.2f} GB, OOM 风险: {oom_risk}",
                details={
                    "peak_memory_gb": peak_gb,
                    "memory_utilization": metrics.memory_utilization if metrics else 0,
                    "oom_risk": oom_risk,
                    "fragmentation_ratio": metrics.fragmentation_ratio if metrics else 0,
                },
                recommendations=self._extract_recommendations(response),
                raw_response=response,
            )
            
        except Exception as e:
            logger.error(f"Memory analysis failed: {e}", exc_info=True)
            return AnalysisResult(
                agent_name=self.name,
                success=False,
                summary="内存分析失败",
                error=str(e),
            )
    
    def _prepare_analysis_data(self, data: Dict[str, Any]) -> MemoryAnalysisData:
        """准备分析数据"""
        analysis_data = MemoryAnalysisData()
        metrics = MemoryMetrics()
        
        # 设备内存
        metrics.device_memory_gb = float(data.get("device_memory_gb", 64.0))
        
        # 峰值内存
        if "peak_memory_mb" in data:
            metrics.peak_memory_mb = float(data["peak_memory_mb"])
            metrics.peak_memory_gb = metrics.peak_memory_mb / 1024
            metrics.peak_memory_bytes = metrics.peak_memory_mb * 1024 * 1024
        
        if "peak_memory_bytes" in data:
            metrics.peak_memory_bytes = float(data["peak_memory_bytes"])
            metrics.peak_memory_mb = metrics.peak_memory_bytes / (1024 * 1024)
            metrics.peak_memory_gb = metrics.peak_memory_mb / 1024
        
        # 内存分布
        if "model_memory_mb" in data:
            metrics.model_memory_mb = float(data["model_memory_mb"])
        if "optimizer_memory_mb" in data:
            metrics.optimizer_memory_mb = float(data["optimizer_memory_mb"])
        if "activation_memory_mb" in data:
            metrics.activation_memory_mb = float(data["activation_memory_mb"])
        if "gradient_memory_mb" in data:
            metrics.gradient_memory_mb = float(data["gradient_memory_mb"])
        
        # 碎片率
        if "fragmentation_ratio" in data:
            metrics.fragmentation_ratio = float(data["fragmentation_ratio"])
        
        # 计算利用率和风险
        metrics.calculate_utilization()
        
        analysis_data.metrics = metrics
        
        # Top 内存算子
        if "top_memory_operators" in data:
            analysis_data.top_memory_operators = data["top_memory_operators"]
        
        return analysis_data
    
    def _extract_recommendations(self, response: str) -> List[str]:
        """从 LLM 响应中提取优化建议"""
        recommendations = []
        
        lines = response.split("\n")
        in_recommendation = False
        
        for line in lines:
            line_lower = line.lower()
            if "建议" in line or "优化" in line or "suggestion" in line_lower:
                in_recommendation = True
            if in_recommendation and line.strip().startswith(("-", "*", "•", "1", "2", "3", "4", "5")):
                clean_line = line.strip().lstrip("-*•0123456789. )")
                if clean_line and len(clean_line) > 5:
                    recommendations.append(clean_line)
        
        return recommendations[:10]
