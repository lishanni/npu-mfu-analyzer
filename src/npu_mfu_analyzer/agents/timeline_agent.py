"""
Timeline Agent

分析 Timeline 数据，包括：
- 时间组成分析（计算/通信/空闲）
- Overlap（通信掩盖）分析
- 慢卡检测
- 瓶颈识别
"""

import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field

from npu_mfu_analyzer.agents.base_agent import BaseAgent, AnalysisResult, AgentMessage
from npu_mfu_analyzer.llm.llm_interface import LLMInterface
from npu_mfu_analyzer.llm.prompts import TIMELINE_ANALYSIS_SYSTEM
from npu_mfu_analyzer.analyzers.overlap_calculator import OverlapCalculator, OverlapMetrics
from npu_mfu_analyzer.analyzers.slow_rank_detector import SlowRankDetector, SlowRankResult

logger = logging.getLogger(__name__)


@dataclass
class TimelineAnalysisData:
    """Timeline 分析数据"""
    # 基本信息
    rank_count: int = 0
    step_count: int = 0
    framework: str = ""
    
    # 时间指标（平均值，单位 us）
    avg_step_time: float = 0.0
    avg_compute_time: float = 0.0
    avg_comm_time: float = 0.0
    avg_free_time: float = 0.0
    
    # Overlap 分析
    overlap_metrics: Optional[OverlapMetrics] = None
    
    # 慢卡检测
    slow_rank_result: Optional[SlowRankResult] = None
    
    # Bubble 分析（PP 并行）
    avg_bubble_time: float = 0.0
    bubble_ratio: float = 0.0
    
    # Top 算子
    top_operators: List[Dict[str, Any]] = field(default_factory=list)
    
    # 原始数据
    raw_summary: Optional[Any] = None
    
    def compute_time_breakdown(self) -> Dict[str, float]:
        """计算时间占比"""
        total = self.avg_compute_time + self.avg_comm_time + self.avg_free_time
        if total == 0:
            return {"compute": 0, "communication": 0, "free": 0}
        return {
            "compute": self.avg_compute_time / total * 100,
            "communication": self.avg_comm_time / total * 100,
            "free": self.avg_free_time / total * 100,
        }
    
    def to_prompt_text(self) -> str:
        """转换为 LLM Prompt 格式"""
        breakdown = self.compute_time_breakdown()
        
        lines = [
            "## Timeline 数据摘要",
            f"- Rank 数量: {self.rank_count}",
            f"- Step 数量: {self.step_count}",
            f"- 平均 Step 时间: {self.avg_step_time / 1000:.2f} ms",
            "",
            "### 时间组成",
            f"- 计算时间: {self.avg_compute_time / 1000:.2f} ms ({breakdown['compute']:.1f}%)",
            f"- 通信时间: {self.avg_comm_time / 1000:.2f} ms ({breakdown['communication']:.1f}%)",
            f"- 空闲时间: {self.avg_free_time / 1000:.2f} ms ({breakdown['free']:.1f}%)",
        ]
        
        # Overlap 分析
        if self.overlap_metrics:
            lines.append("")
            lines.append(self.overlap_metrics.to_prompt_text())
        
        # Bubble 分析
        if self.avg_bubble_time > 0:
            lines.append("")
            lines.append("### PP Bubble 分析")
            lines.append(f"- Bubble 时间: {self.avg_bubble_time / 1000:.2f} ms")
            lines.append(f"- Bubble 占比: {self.bubble_ratio:.1f}%")
        
        # 慢卡检测
        if self.slow_rank_result and self.slow_rank_result.has_slow_ranks():
            lines.append("")
            lines.append(self.slow_rank_result.to_prompt_text())
        
        # Top 算子
        if self.top_operators:
            lines.append("")
            lines.append("### Top 耗时算子")
            for i, op in enumerate(self.top_operators[:5], 1):
                name = op.get("name", "unknown")
                dur = op.get("dur", 0) / 1000  # us -> ms
                lines.append(f"{i}. {name}: {dur:.2f} ms")
        
        return "\n".join(lines)


class TimelineAgent(BaseAgent):
    """
    Timeline 分析 Agent
    
    功能：
    1. 时间组成分析（计算/通信/空闲占比）
    2. Overlap（通信掩盖）分析
    3. 慢卡检测（Dixon/三倍标准差）
    4. PP Bubble 分析
    5. 瓶颈识别和优化建议
    """
    
    PROMPT_TEMPLATE = """
你是昇腾 NPU 性能优化专家。分析以下 Timeline 数据：

{data_summary}

## 分析任务
1. **时间组成评估**：计算/通信/空闲的占比是否合理？
2. **通信掩盖率分析**：
   - 掩盖率 > 80%：优秀，通信被计算完全隐藏
   - 掩盖率 50-80%：良好，仍有优化空间
   - 掩盖率 < 50%：较差，通信成为瓶颈
3. **瓶颈识别**：主要瓶颈在哪里（计算/通信/数据加载）？
4. **慢卡分析**：如果有慢卡，分析根因并给出建议
5. **优化建议**：给出具体、可操作的优化方案

请给出详细的分析结论和建议。
"""
    
    def __init__(
        self, 
        llm: LLMInterface, 
        config: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            name="TimelineAgent",
            llm=llm,
            system_prompt=TIMELINE_ANALYSIS_SYSTEM,
            config=config
        )
        self._overlap_calculator = OverlapCalculator()
        self._slow_rank_detector = SlowRankDetector()
    
    def get_prompt_template(self) -> str:
        return self.PROMPT_TEMPLATE
    
    async def analyze(self, data: Dict[str, Any]) -> AnalysisResult:
        """
        分析 Timeline 数据
        
        Args:
            data: 包含以下可选字段：
                - profiling_summary: ProfilingSummary 对象
                - step_trace_df: Step Trace DataFrame
                - trace_events: trace_view.json 事件列表
                - compute_events: 计算事件列表
                - comm_events: 通信事件列表
                
        Returns:
            AnalysisResult
        """
        try:
            # 1. 提取/构建分析数据
            analysis_data = self._prepare_analysis_data(data)
            
            # 2. 计算 Overlap（如果有原始事件）
            if "compute_events" in data and "comm_events" in data:
                analysis_data.overlap_metrics = self._overlap_calculator.calculate_from_events(
                    data["compute_events"],
                    data["comm_events"]
                )
            
            # 3. 慢卡检测（如果有多 Rank 数据）
            if "step_trace_df" in data and analysis_data.rank_count > 1:
                import pandas as pd
                step_trace_df = data["step_trace_df"]
                if isinstance(step_trace_df, pd.DataFrame) and not step_trace_df.empty:
                    analysis_data.slow_rank_result = self._slow_rank_detector.detect_from_step_trace(
                        step_trace_df
                    )
            
            # 4. 生成 Prompt 并调用 LLM
            prompt = self.format_prompt(
                self.PROMPT_TEMPLATE, 
                data_summary=analysis_data.to_prompt_text()
            )
            response = await self.call_llm(prompt)
            
            # 5. 构建结果
            return AnalysisResult(
                agent_name=self.name,
                success=True,
                summary=self._generate_summary(analysis_data),
                details={
                    "time_breakdown": analysis_data.compute_time_breakdown(),
                    "overlap_ratio": analysis_data.overlap_metrics.overlap_ratio if analysis_data.overlap_metrics else 0,
                    "slow_ranks": analysis_data.slow_rank_result.get_all_slow_ranks() if analysis_data.slow_rank_result else [],
                    "bubble_ratio": analysis_data.bubble_ratio,
                },
                recommendations=self._extract_recommendations(response),
                raw_response=response,
            )
            
        except Exception as e:
            logger.error(f"Timeline analysis failed: {e}", exc_info=True)
            return AnalysisResult(
                agent_name=self.name,
                success=False,
                summary="Timeline 分析失败",
                error=str(e),
            )
    
    def _prepare_analysis_data(self, data: Dict[str, Any]) -> TimelineAnalysisData:
        """从输入数据提取分析数据"""
        analysis_data = TimelineAnalysisData()
        
        # 从 ProfilingSummary 提取
        if "profiling_summary" in data:
            summary = data["profiling_summary"]
            if hasattr(summary, "to_dict"):
                summary = summary.to_dict()
            
            analysis_data.rank_count = summary.get("rank_count", 1)
            analysis_data.step_count = summary.get("step_count", 0)
            analysis_data.framework = summary.get("framework", "")
            analysis_data.avg_step_time = summary.get("avg_step_time", 0)
            analysis_data.avg_compute_time = summary.get("avg_compute_time", 0)
            analysis_data.avg_comm_time = summary.get("avg_comm_time", 0)
            analysis_data.avg_free_time = summary.get("avg_free_time", 0)
            analysis_data.avg_bubble_time = summary.get("avg_bubble_time", 0)
            analysis_data.bubble_ratio = summary.get("bubble_ratio", 0)
            analysis_data.top_operators = summary.get("top_operators", [])
            
            # Overlap 指标
            om = summary.get("overlap_metrics", {})
            if om:
                analysis_data.overlap_metrics = OverlapMetrics(
                    total_compute_time=om.get("total_compute_time", 0),
                    total_comm_time=om.get("total_comm_time", 0),
                    overlapped_time=om.get("overlapped_time", 0),
                    comm_not_overlapped=om.get("comm_not_overlapped", 0),
                    free_time=om.get("free_time", 0),
                    overlap_ratio=om.get("overlap_ratio", 0),
                    e2e_time=om.get("e2e_time", 0),
                )
            
            analysis_data.raw_summary = summary
        
        # 直接传入的字段
        for key in ["rank_count", "step_count", "avg_compute_time", "avg_comm_time", 
                    "avg_free_time", "avg_step_time", "top_operators"]:
            if key in data:
                setattr(analysis_data, key, data[key])
        
        return analysis_data
    
    def _generate_summary(self, data: TimelineAnalysisData) -> str:
        """生成分析摘要"""
        breakdown = data.compute_time_breakdown()
        
        parts = [f"计算 {breakdown['compute']:.0f}%"]
        if breakdown['communication'] > 0:
            parts.append(f"通信 {breakdown['communication']:.0f}%")
        if breakdown['free'] > 0:
            parts.append(f"空闲 {breakdown['free']:.0f}%")
        
        summary = f"时间组成: {', '.join(parts)}"
        
        if data.overlap_metrics and data.overlap_metrics.overlap_ratio > 0:
            summary += f"; 通信掩盖率 {data.overlap_metrics.overlap_ratio:.0f}%"
        
        if data.slow_rank_result and data.slow_rank_result.has_slow_ranks():
            slow_ranks = data.slow_rank_result.get_all_slow_ranks()
            summary += f"; 检测到慢卡 Rank {slow_ranks}"
        
        return summary
    
    def _extract_recommendations(self, response: str) -> List[str]:
        """从 LLM 响应中提取优化建议"""
        recommendations = []
        
        lines = response.split("\n")
        in_recommendation = False
        
        for line in lines:
            line_lower = line.lower()
            if "建议" in line or "优化" in line or "recommendation" in line_lower:
                in_recommendation = True
            if in_recommendation and line.strip().startswith(("-", "*", "•", "1", "2", "3", "4", "5")):
                # 清理前缀
                clean_line = line.strip().lstrip("-*•0123456789. )")
                if clean_line and len(clean_line) > 5:  # 过滤太短的内容
                    recommendations.append(clean_line)
        
        return recommendations[:10]  # 最多 10 条


class EnhancedTimelineAgent(TimelineAgent):
    """
    增强版 Timeline Agent
    
    支持更详细的分析模式：
    - 多 Step 对比分析
    - 时间序列异常检测
    - 自动生成优化代码示例
    """
    
    DETAILED_PROMPT_TEMPLATE = """
你是昇腾 NPU 性能优化专家。请对以下 Timeline 数据进行深度分析：

{data_summary}

## 深度分析任务

### 1. 时间组成诊断
- 评估计算/通信/空闲的比例是否健康
- 计算密集型模型期望：计算 > 70%, 通信 < 20%
- 如果空闲时间过多（>10%），识别原因

### 2. 通信掩盖率深度分析
- 当前掩盖率: {overlap_ratio:.1f}%
- 评估是否达到理论最优
- 分析未掩盖通信的构成（TP/PP/DP）

### 3. 慢卡根因分析（如有）
{slow_rank_analysis}

### 4. Pipeline Bubble 分析（如有）
{bubble_analysis}

### 5. 优化建议
请按优先级给出优化建议，并说明预期收益：
1. 高优先级（预期收益 > 10%）
2. 中优先级（预期收益 5-10%）
3. 低优先级（预期收益 < 5%）

请给出详细、可操作的分析结论。
"""
    
    async def analyze_detailed(self, data: Dict[str, Any]) -> AnalysisResult:
        """详细分析模式"""
        analysis_data = self._prepare_analysis_data(data)
        
        # 构建详细 Prompt
        slow_rank_analysis = "未检测到慢卡"
        if analysis_data.slow_rank_result and analysis_data.slow_rank_result.has_slow_ranks():
            slow_rank_analysis = analysis_data.slow_rank_result.to_prompt_text()
        
        bubble_analysis = "无 PP Bubble 数据"
        if analysis_data.avg_bubble_time > 0:
            bubble_analysis = f"Bubble 时间: {analysis_data.avg_bubble_time / 1000:.2f}ms, 占比: {analysis_data.bubble_ratio:.1f}%"
        
        overlap_ratio = 0
        if analysis_data.overlap_metrics:
            overlap_ratio = analysis_data.overlap_metrics.overlap_ratio
        
        prompt = self.DETAILED_PROMPT_TEMPLATE.format(
            data_summary=analysis_data.to_prompt_text(),
            overlap_ratio=overlap_ratio,
            slow_rank_analysis=slow_rank_analysis,
            bubble_analysis=bubble_analysis,
        )
        
        response = await self.call_llm(prompt)
        
        return AnalysisResult(
            agent_name=self.name,
            success=True,
            summary="Timeline 深度分析完成",
            details={
                "mode": "detailed",
                "time_breakdown": analysis_data.compute_time_breakdown(),
                "overlap_ratio": overlap_ratio,
            },
            recommendations=self._extract_recommendations(response),
            raw_response=response,
        )
