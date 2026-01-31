"""
Operator Agent

分析算子性能，计算 MFU，识别低效算子。
"""

import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field

import pandas as pd

from src.agents.base_agent import BaseAgent, AnalysisResult
from src.llm.llm_interface import LLMInterface
from src.llm.prompts import MFU_ANALYSIS_SYSTEM
from src.analyzers.mfu_calculator import MFUCalculator, MFUMetrics, ChipInfo

logger = logging.getLogger(__name__)


@dataclass
class OperatorAnalysisData:
    """算子分析数据"""
    # MFU 指标
    mfu_metrics: Optional[MFUMetrics] = None
    
    # 算子统计
    total_operators: int = 0
    top_operators: List[Dict[str, Any]] = field(default_factory=list)
    low_efficiency_operators: List[Dict[str, Any]] = field(default_factory=list)
    
    # 芯片信息
    chip_name: str = ""
    peak_flops_tflops: float = 0.0
    
    def to_prompt_text(self) -> str:
        """转换为 LLM Prompt 格式"""
        lines = [
            "## 算子分析数据摘要",
            "",
            f"### 芯片信息",
            f"- 芯片型号: {self.chip_name or '未知'}",
            f"- 理论峰值: {self.peak_flops_tflops:.1f} TFLOPS",
            "",
        ]
        
        if self.mfu_metrics:
            lines.append(self.mfu_metrics.to_prompt_text())
        else:
            lines.append("### MFU 分析")
            lines.append("- 无 MFU 数据（需要算子形状信息）")
        
        if self.top_operators:
            lines.append("")
            lines.append("### Top 10 耗时算子")
            for i, op in enumerate(self.top_operators[:10], 1):
                name = op.get("name", "unknown")
                dur = op.get("dur", 0) / 1e6  # ns -> ms
                lines.append(f"{i}. {name}: {dur:.2f} ms")
        
        if self.low_efficiency_operators:
            lines.append("")
            lines.append("### 低效算子（需优化）")
            for op in self.low_efficiency_operators[:5]:
                name = op.get("name", "unknown")
                mfu = op.get("mfu", 0) * 100
                lines.append(f"- {name}: MFU={mfu:.1f}%")
        
        return "\n".join(lines)


class OperatorAgent(BaseAgent):
    """
    Operator Agent
    
    功能：
    1. MFU 计算（Model FLOPS Utilization）
    2. 算子耗时分析
    3. 低效算子识别
    4. 优化建议生成
    """
    
    PROMPT_TEMPLATE = """
你是昇腾 NPU 算子优化专家。分析以下算子性能数据：

{data_summary}

## 分析任务
1. **MFU 评估**：
   - MFU > 50%：算子效率良好
   - MFU 30-50%：有优化空间
   - MFU < 30%：需要重点优化

2. **瓶颈识别**：
   - 哪些算子占用时间最多？
   - 哪些算子 MFU 最低？
   - 是否有冗余计算？

3. **优化建议**：
   - 算子融合（Operator Fusion）
   - 数据类型优化（FP16/BF16/INT8）
   - FlashAttention 替换普通 Attention
   - 减少不必要的 reshape/transpose

请给出详细的分析结论和具体优化建议。
"""
    
    def __init__(
        self, 
        llm: LLMInterface, 
        config: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            name="OperatorAgent",
            llm=llm,
            system_prompt=MFU_ANALYSIS_SYSTEM,
            config=config
        )
        self._mfu_calculator: Optional[MFUCalculator] = None
    
    def get_prompt_template(self) -> str:
        return self.PROMPT_TEMPLATE
    
    async def analyze(self, data: Dict[str, Any]) -> AnalysisResult:
        """
        分析算子数据
        
        Args:
            data: 包含以下可选字段：
                - operators_df: 算子数据 DataFrame
                - top_operators: Top 耗时算子列表
                - profiling_path: Profiling 数据路径（用于加载芯片信息）
                - chip_info: ChipInfo 对象
                
        Returns:
            AnalysisResult
        """
        try:
            # 1. 准备分析数据
            analysis_data = self._prepare_analysis_data(data)
            
            # 2. 计算 MFU（如果有算子数据）
            if "operators_df" in data:
                analysis_data.mfu_metrics = self._calculate_mfu(
                    data["operators_df"], 
                    data.get("chip_info")
                )
            
            # 3. 生成 Prompt 并调用 LLM
            prompt = self.format_prompt(
                self.PROMPT_TEMPLATE,
                data_summary=analysis_data.to_prompt_text()
            )
            response = await self.call_llm(prompt)
            
            # 4. 构建结果
            mfu_value = analysis_data.mfu_metrics.overall_mfu if analysis_data.mfu_metrics else 0
            
            return AnalysisResult(
                agent_name=self.name,
                success=True,
                summary=f"MFU: {mfu_value*100:.1f}%, 分析 {analysis_data.total_operators} 个算子",
                details={
                    "overall_mfu": mfu_value,
                    "matmul_mfu": analysis_data.mfu_metrics.matmul_mfu if analysis_data.mfu_metrics else 0,
                    "attention_mfu": analysis_data.mfu_metrics.attention_mfu if analysis_data.mfu_metrics else 0,
                    "operator_count": analysis_data.total_operators,
                    "low_efficiency_count": len(analysis_data.low_efficiency_operators),
                },
                recommendations=self._extract_recommendations(response),
                raw_response=response,
            )
            
        except Exception as e:
            logger.error(f"Operator analysis failed: {e}", exc_info=True)
            return AnalysisResult(
                agent_name=self.name,
                success=False,
                summary="算子分析失败",
                error=str(e),
            )
    
    def _prepare_analysis_data(self, data: Dict[str, Any]) -> OperatorAnalysisData:
        """准备分析数据"""
        analysis_data = OperatorAnalysisData()
        
        # 芯片信息
        if "chip_info" in data:
            chip_info = data["chip_info"]
            if isinstance(chip_info, ChipInfo):
                analysis_data.chip_name = chip_info.chip_name
                analysis_data.peak_flops_tflops = chip_info.get_peak_flops() / 1e12
        elif "profiling_path" in data:
            chip_info = ChipInfo.from_profiling_path(data["profiling_path"])
            analysis_data.chip_name = chip_info.chip_name
            analysis_data.peak_flops_tflops = chip_info.get_peak_flops() / 1e12
        
        # Top 算子
        if "top_operators" in data:
            analysis_data.top_operators = data["top_operators"]
            analysis_data.total_operators = len(data["top_operators"])
        
        return analysis_data
    
    def _calculate_mfu(
        self, 
        operators_df: pd.DataFrame,
        chip_info: Optional[ChipInfo] = None,
    ) -> Optional[MFUMetrics]:
        """计算 MFU"""
        if operators_df.empty:
            return None
        
        if chip_info is None:
            chip_info = ChipInfo.default_ascend_910b()
        
        calculator = MFUCalculator(chip_info)
        return calculator.analyze_operators(operators_df)
    
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
