"""
Orchestrator - Agent 编排器

负责任务分发、流程控制、结果整合。
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field

from src.llm.llm_interface import LLMInterface, LLMConfig, LLMFactory, Message
from src.llm.prompts import ADVISOR_SYSTEM
from src.agents.base_agent import BaseAgent, AnalysisResult, TimelineAgent, OperatorAgent
from src.data_loader.profiling_loader import ProfilingLoader
from src.data_loader.data_summarizer import DataSummarizer, ProfilingSummary

logger = logging.getLogger(__name__)


@dataclass
class AnalysisReport:
    """分析报告"""
    success: bool
    summary: str
    profiling_summary: Optional[ProfilingSummary] = None
    agent_results: Dict[str, AnalysisResult] = field(default_factory=dict)
    final_report: str = ""
    recommendations: List[str] = field(default_factory=list)
    error: Optional[str] = None
    
    def to_markdown(self) -> str:
        """转换为 Markdown 格式"""
        lines = [
            "# NPU MFU 性能分析报告",
            "",
            f"## 概述",
            f"{self.summary}",
            "",
        ]
        
        if self.profiling_summary:
            lines.append("## 数据摘要")
            lines.append(self.profiling_summary.to_prompt_text())
            lines.append("")
        
        if self.final_report:
            lines.append("## 详细分析")
            lines.append(self.final_report)
            lines.append("")
        
        if self.recommendations:
            lines.append("## 优化建议")
            for i, rec in enumerate(self.recommendations, 1):
                lines.append(f"{i}. {rec}")
            lines.append("")
        
        return "\n".join(lines)


class Orchestrator:
    """
    Agent 编排器
    
    协调多个 Agent 执行分析任务，整合结果生成最终报告。
    
    Usage:
        orchestrator = Orchestrator(profiling_path="/path/to/profiling")
        report = await orchestrator.run()
    """
    
    def __init__(
        self,
        profiling_path: str,
        llm_config: Optional[LLMConfig] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Args:
            profiling_path: Profiling 数据路径
            llm_config: LLM 配置
            config: 额外配置
        """
        self.profiling_path = profiling_path
        self.llm_config = llm_config or LLMConfig()
        self.config = config or {}
        
        # 初始化组件
        self.loader = ProfilingLoader(profiling_path)
        self.summarizer = DataSummarizer(self.loader)
        self.llm = LLMFactory.create(self.llm_config)
        
        # 初始化 Agents
        self.agents: Dict[str, BaseAgent] = {}
        self._init_agents()
    
    def _init_agents(self):
        """初始化所有 Agent"""
        self.agents["timeline"] = TimelineAgent(self.llm, self.config)
        self.agents["operator"] = OperatorAgent(self.llm, self.config)
        # 后续添加更多 Agent
        # self.agents["memory"] = MemoryAgent(self.llm, self.config)
        # self.agents["communication"] = CommunicationAgent(self.llm, self.config)
    
    async def run(self) -> AnalysisReport:
        """
        执行完整的分析流程
        
        Returns:
            AnalysisReport
        """
        logger.info(f"Starting analysis for {self.profiling_path}")
        
        try:
            # 1. 检测数据
            info = self.loader.detect()
            logger.info(f"Detected: {info.data_type} data, {info.rank_count} ranks")
            
            if info.data_type == "unknown":
                return AnalysisReport(
                    success=False,
                    summary="未找到有效的 Profiling 数据",
                    error="No valid profiling data found"
                )
            
            # 2. 生成数据摘要
            profiling_summary = self.summarizer.summarize()
            logger.info(f"Generated summary: {profiling_summary.step_count} steps")
            
            # 3. 并行执行各 Agent 分析
            agent_results = await self._run_agents(profiling_summary)
            
            # 4. 整合结果生成最终报告
            final_report, recommendations = await self._generate_final_report(
                profiling_summary, agent_results
            )
            
            return AnalysisReport(
                success=True,
                summary=f"分析完成：{info.rank_count} 卡，{profiling_summary.step_count} 步",
                profiling_summary=profiling_summary,
                agent_results=agent_results,
                final_report=final_report,
                recommendations=recommendations,
            )
            
        except Exception as e:
            logger.error(f"Analysis failed: {e}", exc_info=True)
            return AnalysisReport(
                success=False,
                summary="分析失败",
                error=str(e)
            )
    
    async def _run_agents(self, summary: ProfilingSummary) -> Dict[str, AnalysisResult]:
        """并行运行各 Agent"""
        results = {}
        
        # 准备数据
        data = summary.to_dict()
        
        # 并行执行
        tasks = []
        agent_names = []
        
        for name, agent in self.agents.items():
            tasks.append(agent.analyze(data))
            agent_names.append(name)
        
        if tasks:
            responses = await asyncio.gather(*tasks, return_exceptions=True)
            
            for name, response in zip(agent_names, responses):
                if isinstance(response, Exception):
                    logger.error(f"Agent {name} failed: {response}")
                    results[name] = AnalysisResult(
                        agent_name=name,
                        success=False,
                        summary=f"{name} 分析失败",
                        error=str(response)
                    )
                else:
                    results[name] = response
        
        return results
    
    async def _generate_final_report(
        self,
        summary: ProfilingSummary,
        agent_results: Dict[str, AnalysisResult]
    ) -> tuple:
        """整合各 Agent 结果，生成最终报告"""
        
        # 构建综合 Prompt
        prompt_parts = [
            "## Profiling 数据摘要",
            summary.to_prompt_text(),
            "",
            "## 各维度分析结果",
        ]
        
        for name, result in agent_results.items():
            prompt_parts.append(f"\n### {name} 分析")
            if result.success and result.raw_response:
                prompt_parts.append(result.raw_response[:2000])  # 限制长度
            else:
                prompt_parts.append(f"分析失败: {result.error}")
        
        prompt_parts.append("""
## 任务
请综合以上分析结果，生成最终的性能优化报告，包括：
1. 性能概览（当前 MFU 估计、主要瓶颈）
2. 瓶颈分析（按影响程度排序）
3. 优化建议（具体、可操作，包含代码示例）
4. 预期收益（优化后的预期提升）
""")
        
        prompt = "\n".join(prompt_parts)
        
        # 调用 LLM 生成最终报告
        messages = [
            Message(role="system", content=ADVISOR_SYSTEM),
            Message(role="user", content=prompt)
        ]
        
        response = await self.llm.complete(messages)
        final_report = response.content
        
        # 提取建议
        recommendations = self._extract_recommendations(final_report)
        
        return final_report, recommendations
    
    def _extract_recommendations(self, report: str) -> List[str]:
        """从报告中提取建议列表"""
        recommendations = []
        
        lines = report.split("\n")
        in_rec_section = False
        
        for line in lines:
            if "优化建议" in line or "建议" in line:
                in_rec_section = True
                continue
            
            if in_rec_section:
                if line.strip().startswith(("#", "##")):
                    in_rec_section = False
                elif line.strip().startswith(("-", "*", "1", "2", "3", "4", "5")):
                    rec = line.strip().lstrip("-*0123456789. ")
                    if rec:
                        recommendations.append(rec)
        
        return recommendations[:20]  # 最多20条


async def run_analysis(profiling_path: str, llm_backend: str = "openai") -> AnalysisReport:
    """
    便捷的分析入口
    
    Args:
        profiling_path: Profiling 数据路径
        llm_backend: LLM 后端
        
    Returns:
        AnalysisReport
    """
    config = LLMConfig(backend=llm_backend)
    orchestrator = Orchestrator(profiling_path, llm_config=config)
    return await orchestrator.run()
