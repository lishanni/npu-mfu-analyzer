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
from src.agents.base_agent import BaseAgent, AnalysisResult
from src.agents.timeline_agent import TimelineAgent
from src.agents.operator_agent import OperatorAgent
from src.agents.memory_agent import MemoryAgent
from src.agents.communication_agent import CommunicationAgent
from src.agents.jitter_agent import JitterAgent
from src.agents.advisor_agent import AdvisorAgent
from src.agents.detailed_operator_agent import DetailedOperatorAgent
from src.agents.detailed_operator_agent_v2 import DetailedOperatorAgentV2
from src.data_loader.profiling_loader import ProfilingLoader
from src.data_loader.data_summarizer import DataSummarizer, ProfilingSummary
from src.report.report_generator import ReportGenerator, ReportFormat
from src.analyzers.communication_matrix_analyzer import CommunicationMatrixAnalyzer, CommunicationMatrix

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
    mfu_metrics: Any = None  # MFU 计算结果
    roofline_analysis: Any = None  # Roofline 分析结果
    communication_matrix: Optional[CommunicationMatrix] = None  # 通信矩阵分析结果
    comm_matrix_html: Optional[str] = None  # 通信矩阵可视化 HTML
    dashboard_html: Optional[str] = None  # 链路性能仪表板 HTML

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

        # 通信矩阵分析摘要
        if self.communication_matrix:
            lines.append("## 通信矩阵分析")
            lines.append(self._format_comm_matrix_summary())
            lines.append("")

        if self.recommendations:
            lines.append("## 优化建议")
            for i, rec in enumerate(self.recommendations, 1):
                lines.append(f"{i}. {rec}")
            lines.append("")

        return "\n".join(lines)

    def _format_comm_matrix_summary(self) -> str:
        """格式化通信矩阵摘要"""
        if not self.communication_matrix:
            return ""

        matrix = self.communication_matrix
        lines = [
            f"- 总通信量: {matrix.total_comm_data_mb:.2f} MB",
            f"- 总通信时间: {matrix.total_comm_time_ms:.2f} ms",
            f"- 平均带宽: {matrix.avg_bandwidth_gbps:.2f} GB/s",
            f"- 活跃链路数: {len(matrix.link_metrics)}",
        ]

        if matrix.slow_links:
            lines.append(f"- ⚠️ 慢链路数: {len(matrix.slow_links)}")

        if matrix.bottleneck_links:
            lines.append(f"- ⚠️ 瓶颈链路数: {len(matrix.bottleneck_links)}")

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
        config: Optional[Dict[str, Any]] = None,
        enable_comm_matrix: bool = True,
        enable_dashboard: bool = True,
        enable_deep_operator_analysis: bool = True,
    ):
        """
        Args:
            profiling_path: Profiling 数据路径
            llm_config: LLM 配置
            config: 额外配置
            enable_comm_matrix: 是否启用通信矩阵分析
            enable_dashboard: 是否启用链路性能仪表板
            enable_deep_operator_analysis: 是否启用深度算子分析 V2
        """
        self.profiling_path = profiling_path
        self.llm_config = llm_config or LLMConfig()
        self.config = config or {}
        self.enable_comm_matrix = enable_comm_matrix
        self.enable_dashboard = enable_dashboard
        self.enable_deep_operator_analysis = enable_deep_operator_analysis

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
        self.agents["memory"] = MemoryAgent(self.llm, self.config)
        self.agents["communication"] = CommunicationAgent(self.llm, self.config)
        self.agents["jitter"] = JitterAgent(self.llm, self.config)

        # 详细算子分析 Agent V1（仅在检测到 AIC metrics 时启用）
        if self._check_aic_metrics_available():
            self.agents["detailed_operator"] = DetailedOperatorAgent(self.llm, self.config)
            logger.info("DetailedOperatorAgent enabled (AIC metrics detected)")

            # 详细算子分析 Agent V2（深度分析）
            if self.enable_deep_operator_analysis:
                self.agents["detailed_operator_v2"] = DetailedOperatorAgentV2(self.llm, self.config)
                logger.info("DetailedOperatorAgentV2 enabled (deep analysis)")

        # Advisor Agent 单独保存，用于最终综合分析
        self.advisor = AdvisorAgent(self.llm, self.config)

        # 报告生成器
        self.report_generator = ReportGenerator()
    
    async def run(self, output_format: ReportFormat = ReportFormat.MARKDOWN) -> AnalysisReport:
        """
        执行完整的分析流程
        
        Args:
            output_format: 报告输出格式
        
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

            # 2.5. 计算 MFU 和 Roofline 分析
            mfu_metrics = self._calculate_mfu()
            roofline_analysis = self._analyze_roofline()

            # 2.6. 通信矩阵分析
            comm_matrix = None
            comm_matrix_html = None
            dashboard_html = None
            if self.enable_comm_matrix:
                comm_matrix = self._analyze_communication_matrix()
                if comm_matrix:
                    # 生成通信矩阵可视化
                    from src.analyzers.communication_matrix_visualizer import CommunicationMatrixVisualizer
                    visualizer = CommunicationMatrixVisualizer(comm_matrix)
                    comm_matrix_html = visualizer.generate_html()
                    logger.info(f"Communication matrix analysis complete: {len(comm_matrix.link_metrics)} links")

                    # 生成链路性能仪表板
                    if self.enable_dashboard:
                        from src.analyzers.link_performance_dashboard import generate_dashboard
                        dashboard_html = generate_dashboard(comm_matrix)
                        logger.info("Link performance dashboard generated")

            # 3. 检查是否有 AIC metrics 数据
            has_aic_metrics = self._check_aic_metrics_available()

            # 4. 准备 Agent 数据
            agent_data = profiling_summary.to_dict()
            agent_data["profiling_path"] = self.profiling_path

            # 如果有 AIC metrics，添加 top operators 用于详细分析
            if has_aic_metrics:
                top_kernels = self.loader.get_top_kernels(top_n=20)
                if top_kernels:
                    agent_data["top_operators"] = top_kernels
                    logger.info(f"AIC metrics detected, enabling detailed operator analysis for {len(top_kernels)} operators")

            # 5. 并行执行各 Agent 分析
            agent_results = await self._run_agents(agent_data)

            # 6. 使用 Advisor Agent 生成综合分析
            advisor_result = await self._run_advisor(profiling_summary, agent_results)

            # 5. 生成最终报告
            final_report = self.report_generator.generate_from_analysis(
                profiling_path=self.profiling_path,
                profiling_summary=profiling_summary,
                agent_results=agent_results,
                advisor_report=advisor_result.details.get("advisor_report") if advisor_result.success else None,
                mfu_metrics=mfu_metrics,
                roofline_analysis=roofline_analysis,
                format=output_format,
            )
            
            # 提取建议
            recommendations = []
            if advisor_result.success:
                recommendations = advisor_result.recommendations
            else:
                # 从各 Agent 结果中提取建议
                for name, result in agent_results.items():
                    if hasattr(result, "recommendations"):
                        recommendations.extend(result.recommendations[:5])
            
            return AnalysisReport(
                success=True,
                summary=f"分析完成：{info.rank_count} 卡，{profiling_summary.step_count} 步",
                profiling_summary=profiling_summary,
                agent_results=agent_results,
                final_report=final_report,
                recommendations=recommendations[:20],
                mfu_metrics=mfu_metrics,
                roofline_analysis=roofline_analysis,
                communication_matrix=comm_matrix,
                comm_matrix_html=comm_matrix_html,
                dashboard_html=dashboard_html,
            )
            
        except Exception as e:
            logger.error(f"Analysis failed: {e}", exc_info=True)
            return AnalysisReport(
                success=False,
                summary="分析失败",
                error=str(e)
            )
    
    async def _run_advisor(
        self, 
        summary: ProfilingSummary, 
        agent_results: Dict[str, AnalysisResult]
    ) -> AnalysisResult:
        """运行 Advisor Agent 生成综合分析"""
        try:
            advisor_data = {
                "profiling_summary": summary,
                "agent_results": agent_results,
            }
            result = await self.advisor.analyze(advisor_data)
            return result
        except Exception as e:
            logger.error(f"Advisor analysis failed: {e}")
            return AnalysisResult(
                agent_name="AdvisorAgent",
                success=False,
                summary="综合分析失败",
                error=str(e)
            )
    
    async def _run_agents(self, data: Dict[str, Any]) -> Dict[str, AnalysisResult]:
        """
        并行运行各 Agent

        Args:
            data: Agent 数据字典，包含 profiling_summary 和其他可选字段（如 top_operators）

        Returns:
            各 Agent 的分析结果字典
        """
        results = {}

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

    def _calculate_mfu(self):
        """计算 MFU 指标"""
        try:
            from src.analyzers.mfu_calculator import MFUCalculator, ChipInfo, MFUMetrics
            import pandas as pd

            # 获取芯片信息
            chip_info = ChipInfo.from_profiling_path(self.profiling_path)
            if not chip_info.is_valid():
                chip_info = ChipInfo.default_ascend_910b()

            calculator = MFUCalculator(chip_info=chip_info)

            # 获取算子数据 - 使用 get_kernel_details_for_mfu 获取带形状信息的算子
            all_kernels = self.loader.get_kernel_details_for_mfu(top_n=1000)

            # 尝试精确计算 MFU（如果有足够带形状的算子）
            if all_kernels and len(all_kernels) >= 10:
                # 转换为 DataFrame，添加 MFUCalculator 需要的列
                # 注意：MFUCalculator 期望 duration_ns 是纳秒，而 kernel_details.csv 中的 duration(us) 是微秒
                operators_df = pd.DataFrame([
                    {
                        "name": k["name"],
                        "dur": k.get("dur", 0) * 1000,  # 微秒转纳秒 (MFUCalculator 期望纳秒)
                        "input_shapes": k.get("input_shapes", ""),
                        "input_types": k.get("input_types", ""),
                        "output_shapes": k.get("output_shapes", ""),
                    }
                    for k in all_kernels
                ])

                # 执行 MFU 分析
                mfu_metrics = calculator.analyze_operators(operators_df)
                logger.info(f"MFU Analysis (detailed): overall={mfu_metrics.overall_mfu*100:.1f}%, "
                           f"peak={mfu_metrics.peak_flops/1e12:.1f} TFLOPS")

                # 如果 MFU 过低，可能是大部分算子缺少形状信息
                # 使用计算时间占比作为补充指标
                if mfu_metrics.overall_mfu < 0.01:
                    summary_dict = self.summarizer.summarize().to_dict()
                    compute_time_us = summary_dict.get("avg_compute_time", 0)
                    total_time_us = summary_dict.get("avg_step_time", compute_time_us)

                    if total_time_us > 0:
                        compute_ratio = compute_time_us / total_time_us
                        # 假设计算时间内能达到 40% 峰值（保守估计）
                        estimated_mfu = compute_ratio * 0.4
                        logger.info(f"Low detailed MFU, using estimate based on compute ratio: "
                                   f"compute_ratio={compute_ratio*100:.1f}%, estimated_mfu={estimated_mfu*100:.1f}%")

                        # 如果估算值更高，使用估算值
                        if estimated_mfu > mfu_metrics.overall_mfu:
                            mfu_metrics.overall_mfu = estimated_mfu

                return mfu_metrics

            # 回退到基于总计算时间的简化 MFU 计算
            logger.info("Using simplified MFU calculation based on total compute time")
            summary_dict = self.summarizer.summarize().to_dict()

            # 获取计算时间和总时间（微秒）
            compute_time_us = summary_dict.get("avg_compute_time", 0)
            total_time_us = summary_dict.get("avg_step_time", compute_time_us)

            if compute_time_us <= 0:
                logger.warning("No compute time available for MFU calculation")
                return None

            # 计算时间占比
            compute_ratio = compute_time_us / total_time_us if total_time_us > 0 else 0

            # 估算 MFU = 计算时间占比 × 估算的算力利用率
            # 假设计算密集型操作能达到 40% 峰值
            peak_flops = chip_info.get_peak_flops()
            estimated_mfu = compute_ratio * 0.4

            # 创建简化的 MFU 指标
            mfu_metrics = MFUMetrics()
            mfu_metrics.peak_flops = peak_flops
            mfu_metrics.actual_flops = peak_flops * estimated_mfu
            mfu_metrics.total_duration_ns = total_time_us * 1000  # 转纳秒
            mfu_metrics.overall_mfu = estimated_mfu

            logger.info(f"MFU Analysis (simplified): overall={mfu_metrics.overall_mfu*100:.1f}%, "
                       f"peak={mfu_metrics.peak_flops/1e12:.1f} TFLOPS, "
                       f"compute_ratio={compute_ratio*100:.1f}%")
            return mfu_metrics

        except Exception as e:
            logger.error(f"MFU calculation failed: {e}")
            return None

    def _analyze_roofline(self):
        """执行 Roofline 分析"""
        try:
            from src.roofline.roofline_model import RooflineModeler, PrecisionType

            # 使用默认 Atlas A2 280T 规格
            modeler = RooflineModeler(hardware_name="atlas_a2_280t")

            # 简化版 Roofline 分析 - 基于整体统计数据
            summary_dict = self.summarizer.summarize().to_dict()

            # avg_step_time 单位是微秒，转换为秒
            # 数据来源: step_trace_time.csv 中的时间是微秒
            step_time_s = summary_dict.get("avg_step_time", 0) / 1_000_000

            # 尝试从算子数据估算真实的 FLOPS
            # 如果无法获取，则使用基于计算时间的保守估算
            compute_time_us = summary_dict.get("avg_compute_time", 0)
            compute_time_s = compute_time_us / 1_000_000

            # 简化的 FLOPS 估算: 假设计算时间内的算力利用率
            # 使用芯片峰值算力 * 计算时间占比来估算
            if compute_time_s > 0:
                # 峰值算力: 280 TFLOPS * 20 AICores = 5600 TFLOPS (Atlas A2 280T)
                peak_tflops = 5600  # TFLOPS
                # 假设计算密集型算子能达到 80% 峰值效率
                efficiency = 0.8
                estimated_flops = peak_tflops * efficiency * compute_time_s
            else:
                # 回退到保守估算
                estimated_flops = step_time_s * 1e12  # 1 TFLOPS per second

            # 简化假设: 大模型训练的典型计算强度
            # 假设每操作 16 FLOPs (FP16 MAC: 16 * 2 = 32, 这里简化)
            model_memory_bytes = step_time_s * 16 * 1e9  # 基于时间估算

            # 执行 Roofline 分析 (注意：这里传入的是秒，不需要再转换)
            result = modeler.estimate_theoretical_mfu(
                model_flops=estimated_flops,
                model_memory_bytes=model_memory_bytes,
                step_time_ms=step_time_s * 1000,  # roofline_model.py 中会除以1000转回秒
                num_devices=summary_dict.get("rank_count", 1),
                precision=PrecisionType.FP16,
            )

            logger.info(f"Roofline Analysis: bound={result['bound_type']}, "
                       f"actual_mfu={result['actual_mfu_percent']:.1f}%, "
                       f"theoretical_max={result['theoretical_max_mfu_percent']:.1f}%")
            return result

        except Exception as e:
            logger.error(f"Roofline analysis failed: {e}")
            return None

    def _check_aic_metrics_available(self) -> bool:
        """
        检查是否有 AIC metrics 数据可用

        通过查找 OPPROF_* 目录来判断是否包含 msprof op --aic-metrics 生成的数据。

        Returns:
            True 如果找到 AIC metrics 数据，否则 False
        """
        import glob
        from pathlib import Path

        # 查找所有 OPPROF_* 目录
        opprof_dirs = glob.glob(str(Path(self.profiling_path) / "OPPROF_*"), recursive=False)

        if not opprof_dirs:
            # 尝试更深层的搜索
            opprof_dirs = glob.glob(str(Path(self.profiling_path) / "**" / "OPPROF_*"), recursive=True)

        if opprof_dirs:
            logger.debug(f"Found {len(opprof_dirs)} AIC metrics directories")
            return True

        return False

    def _analyze_communication_matrix(self) -> Optional[CommunicationMatrix]:
        """执行通信矩阵分析"""
        try:
            from src.analyzers.communication_matrix_analyzer import CommunicationMatrixAnalyzer

            # 获取 world_size
            info = self.loader.detect()
            world_size = info.rank_count

            # 查找 DB 文件
            db_files = list(Path(self.profiling_path).rglob("*.db"))
            if not db_files:
                logger.warning("No DB files found for communication matrix analysis")
                return None

            # 使用第一个 DB 文件
            db_path = str(db_files[0])

            # 创建分析器并执行分析
            analyzer = CommunicationMatrixAnalyzer(
                world_size=world_size,
                npus_per_machine=8,  # 可配置
            )
            matrix = analyzer.analyze_from_db(db_path)

            logger.info(f"Communication matrix analysis complete: {len(matrix.link_metrics)} links")
            return matrix

        except Exception as e:
            logger.error(f"Communication matrix analysis failed: {e}")
            return None


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
