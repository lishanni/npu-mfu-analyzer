"""
Communication Agent

分析集群通信，识别通信瓶颈和负载不均。
集成通信拆分（TP/DP/PP）和慢卡检测。
"""

import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field

import pandas as pd

from src.agents.base_agent import BaseAgent, AnalysisResult
from src.llm.llm_interface import LLMInterface
from src.llm.prompts import COMMUNICATION_ANALYSIS_SYSTEM
from src.analyzers.comm_splitter import CommunicationSplitter, CommSplitResult, ParallelConfig
from src.analyzers.slow_rank_detector import SlowRankDetector, SlowRankResult

logger = logging.getLogger(__name__)


@dataclass
class CommunicationMetrics:
    """通信分析指标"""
    # 总通信时间
    total_comm_time_ms: float = 0.0
    avg_comm_time_per_step_ms: float = 0.0
    
    # 通信占比
    comm_ratio: float = 0.0  # 通信时间 / Step 时间
    
    # 通信拆分
    comm_split: Optional[CommSplitResult] = None
    
    # 通信带宽
    total_comm_bytes: float = 0.0
    avg_bandwidth_gbps: float = 0.0
    
    # 慢卡检测
    slow_rank_result: Optional[SlowRankResult] = None
    
    # 通信拓扑
    world_size: int = 1
    parallel_config: Optional[ParallelConfig] = None
    
    def to_prompt_text(self) -> str:
        """转换为 LLM Prompt 格式"""
        lines = [
            "## 通信分析",
            "",
            "### 通信概况",
            f"- 总通信时间: {self.total_comm_time_ms:.2f} ms",
            f"- 平均每 Step 通信时间: {self.avg_comm_time_per_step_ms:.2f} ms",
            f"- 通信占比: {self.comm_ratio * 100:.1f}%",
            f"- World Size: {self.world_size}",
        ]
        
        if self.avg_bandwidth_gbps > 0:
            lines.append(f"- 平均带宽: {self.avg_bandwidth_gbps:.1f} GB/s")
        
        # 并行配置
        if self.parallel_config:
            lines.append("")
            lines.append("### 并行配置")
            lines.append(f"- TP (Tensor Parallel): {self.parallel_config.tensor_parallel_size}")
            lines.append(f"- PP (Pipeline Parallel): {self.parallel_config.pipeline_parallel_size}")
            lines.append(f"- DP (Data Parallel): {self.parallel_config.data_parallel_size}")
        
        # 通信拆分
        if self.comm_split and self.comm_split.total_comm_time > 0:
            lines.append("")
            lines.append(self.comm_split.to_prompt_text())
        
        # 慢卡检测
        if self.slow_rank_result and self.slow_rank_result.has_slow_ranks():
            lines.append("")
            lines.append(self.slow_rank_result.to_prompt_text())
        
        return "\n".join(lines)


@dataclass
class CommunicationAnalysisData:
    """通信分析数据"""
    metrics: Optional[CommunicationMetrics] = None
    top_comm_operators: List[Dict[str, Any]] = field(default_factory=list)
    comm_by_group: Dict[str, float] = field(default_factory=dict)
    
    def to_prompt_text(self) -> str:
        """转换为 LLM Prompt 格式"""
        lines = []
        
        if self.metrics:
            lines.append(self.metrics.to_prompt_text())
        
        if self.top_comm_operators:
            lines.append("")
            lines.append("### Top 通信算子")
            for i, op in enumerate(self.top_comm_operators[:10], 1):
                name = op.get("name", "unknown")
                time_ms = op.get("time_ms", 0)
                group = op.get("group", "")
                lines.append(f"{i}. {name}: {time_ms:.2f} ms ({group})")
        
        return "\n".join(lines)


class CommunicationAgent(BaseAgent):
    """
    Communication Agent
    
    功能：
    1. 通信时间分析
    2. TP/DP/PP 通信拆分
    3. 通信带宽分析
    4. 慢卡检测
    5. 通信优化建议
    """
    
    PROMPT_TEMPLATE = """
你是昇腾 NPU 集群通信优化专家。分析以下通信数据：

{data_summary}

## 分析任务
1. **通信时间评估**：
   - 通信占比是否合理？（通常 < 30% 为健康）
   - 哪种并行策略通信开销最大？

2. **TP/DP/PP 通信分析**：
   - TP 通信：AllReduce/AllGather 是否高效？
   - PP 通信：P2P Send/Recv 是否存在阻塞？
   - DP 通信：梯度同步是否可以优化？

3. **慢卡分析**（如果有）：
   - 慢卡根因是什么？（计算/通信/网络）
   - 如何减少慢卡影响？

4. **带宽利用分析**：
   - 带宽利用率如何？
   - 是否存在网络拥塞？

5. **优化建议**：
   - 通信-计算重叠（Overlap）
   - 梯度累积减少 DP 通信
   - TP-SP Overlap
   - 调整 PP schedule

请给出详细的分析结论和具体优化建议。
"""
    
    def __init__(
        self, 
        llm: LLMInterface, 
        config: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            name="CommunicationAgent",
            llm=llm,
            system_prompt=COMMUNICATION_ANALYSIS_SYSTEM,
            config=config
        )
        self._comm_splitter = CommunicationSplitter()
        self._slow_rank_detector = SlowRankDetector()
    
    def get_prompt_template(self) -> str:
        return self.PROMPT_TEMPLATE
    
    async def analyze(self, data: Dict[str, Any]) -> AnalysisResult:
        """
        分析通信数据
        
        Args:
            data: 包含以下可选字段：
                - comm_events: 通信事件列表
                - comm_df: 通信数据 DataFrame
                - step_trace_df: Step Trace DataFrame（用于慢卡检测）
                - parallel_config: ParallelConfig 对象
                - world_size: 总卡数
                - total_step_time_ms: 总 Step 时间
                
        Returns:
            AnalysisResult
        """
        try:
            # 1. 准备分析数据
            analysis_data = self._prepare_analysis_data(data)
            
            # 2. 通信拆分
            if "comm_events" in data:
                analysis_data.metrics.comm_split = self._comm_splitter.split_from_events(
                    data["comm_events"]
                )
            elif "comm_df" in data:
                analysis_data.metrics.comm_split = self._comm_splitter.split_from_dataframe(
                    data["comm_df"]
                )
            
            # 3. 慢卡检测
            if "step_trace_df" in data and analysis_data.metrics.world_size > 1:
                step_trace_df = data["step_trace_df"]
                if isinstance(step_trace_df, pd.DataFrame) and not step_trace_df.empty:
                    analysis_data.metrics.slow_rank_result = self._slow_rank_detector.detect_from_step_trace(
                        step_trace_df
                    )
            
            # 4. 生成 Prompt 并调用 LLM
            prompt = self.format_prompt(
                self.PROMPT_TEMPLATE,
                data_summary=analysis_data.to_prompt_text()
            )
            response = await self.call_llm(prompt)
            
            # 5. 构建结果
            metrics = analysis_data.metrics
            comm_split = metrics.comm_split if metrics else None
            
            details = {
                "total_comm_time_ms": metrics.total_comm_time_ms if metrics else 0,
                "comm_ratio": metrics.comm_ratio if metrics else 0,
                "world_size": metrics.world_size if metrics else 1,
            }
            
            if comm_split:
                details.update({
                    "tp_comm_time_ms": comm_split.tp_comm_time / 1000,
                    "pp_comm_time_ms": comm_split.pp_comm_time / 1000,
                    "dp_comm_time_ms": comm_split.dp_comm_time / 1000,
                })
            
            slow_ranks = []
            if metrics and metrics.slow_rank_result:
                slow_ranks = metrics.slow_rank_result.get_all_slow_ranks()
                details["slow_ranks"] = slow_ranks
            
            summary = f"通信占比: {metrics.comm_ratio*100:.1f}%" if metrics else "通信分析"
            if slow_ranks:
                summary += f", 检测到慢卡: Rank {slow_ranks}"
            
            return AnalysisResult(
                agent_name=self.name,
                success=True,
                summary=summary,
                details=details,
                recommendations=self._extract_recommendations(response),
                raw_response=response,
            )
            
        except Exception as e:
            logger.error(f"Communication analysis failed: {e}", exc_info=True)
            return AnalysisResult(
                agent_name=self.name,
                success=False,
                summary="通信分析失败",
                error=str(e),
            )
    
    def _prepare_analysis_data(self, data: Dict[str, Any]) -> CommunicationAnalysisData:
        """准备分析数据"""
        analysis_data = CommunicationAnalysisData()
        metrics = CommunicationMetrics()
        
        # 基本信息
        metrics.world_size = int(data.get("world_size", 1))
        
        # 并行配置
        if "parallel_config" in data:
            metrics.parallel_config = data["parallel_config"]
        elif "tp_size" in data or "pp_size" in data or "dp_size" in data:
            metrics.parallel_config = ParallelConfig(
                world_size=metrics.world_size,
                tensor_parallel_size=int(data.get("tp_size", 1)),
                pipeline_parallel_size=int(data.get("pp_size", 1)),
                data_parallel_size=int(data.get("dp_size", 1)),
            )
        
        # 通信时间
        if "total_comm_time_ms" in data:
            metrics.total_comm_time_ms = float(data["total_comm_time_ms"])
        if "avg_comm_time_per_step_ms" in data:
            metrics.avg_comm_time_per_step_ms = float(data["avg_comm_time_per_step_ms"])
        
        # 通信占比
        if "total_step_time_ms" in data and data["total_step_time_ms"] > 0:
            metrics.comm_ratio = metrics.total_comm_time_ms / float(data["total_step_time_ms"])
        elif "comm_ratio" in data:
            metrics.comm_ratio = float(data["comm_ratio"])
        
        # 带宽
        if "total_comm_bytes" in data and "total_comm_time_ms" in data:
            metrics.total_comm_bytes = float(data["total_comm_bytes"])
            if metrics.total_comm_time_ms > 0:
                # GB/s = bytes / (ms * 1e6)
                metrics.avg_bandwidth_gbps = (
                    metrics.total_comm_bytes / 1e9 / (metrics.total_comm_time_ms / 1000)
                )
        
        # Top 通信算子
        if "top_comm_operators" in data:
            analysis_data.top_comm_operators = data["top_comm_operators"]
        
        analysis_data.metrics = metrics
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
