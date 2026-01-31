"""
报告生成器

统一管理报告生成，支持多种输出格式。
"""

import logging
from typing import Dict, Any, List, Optional, Union
from enum import Enum
from pathlib import Path
from datetime import datetime

from src.report.templates import ReportData, MarkdownTemplate, HTMLTemplate

logger = logging.getLogger(__name__)


class ReportFormat(Enum):
    """报告格式"""
    MARKDOWN = "markdown"
    HTML = "html"
    TEXT = "text"
    JSON = "json"


class ReportGenerator:
    """
    报告生成器
    
    功能：
    1. 从分析结果生成报告
    2. 支持多种输出格式（Markdown/HTML/Text/JSON）
    3. 支持保存到文件
    """
    
    def __init__(self, format: ReportFormat = ReportFormat.MARKDOWN):
        """
        Args:
            format: 默认报告格式
        """
        self.default_format = format
    
    def generate(
        self,
        data: Union[ReportData, Dict[str, Any]],
        format: Optional[ReportFormat] = None,
    ) -> str:
        """
        生成报告
        
        Args:
            data: 报告数据（ReportData 或字典）
            format: 报告格式（可选，默认使用初始化时的格式）
            
        Returns:
            报告内容字符串
        """
        format = format or self.default_format
        
        # 转换为 ReportData
        if isinstance(data, dict):
            data = self._dict_to_report_data(data)
        
        # 根据格式生成报告
        if format == ReportFormat.MARKDOWN:
            return MarkdownTemplate.render(data)
        elif format == ReportFormat.HTML:
            return HTMLTemplate.render(data)
        elif format == ReportFormat.TEXT:
            return self._generate_text(data)
        elif format == ReportFormat.JSON:
            return self._generate_json(data)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def generate_from_analysis(
        self,
        profiling_path: str,
        profiling_summary: Any,
        agent_results: Dict[str, Any],
        advisor_report: Optional[Any] = None,
        format: Optional[ReportFormat] = None,
    ) -> str:
        """
        从分析结果生成报告
        
        Args:
            profiling_path: Profiling 数据路径
            profiling_summary: ProfilingSummary 对象
            agent_results: 各 Agent 分析结果
            advisor_report: AdvisorReport 对象（可选）
            format: 报告格式
            
        Returns:
            报告内容字符串
        """
        # 提取摘要数据
        summary_dict = {}
        if hasattr(profiling_summary, "to_dict"):
            summary_dict = profiling_summary.to_dict()
        elif isinstance(profiling_summary, dict):
            summary_dict = profiling_summary
        
        # 计算时间占比
        compute_time = summary_dict.get("avg_compute_time", 0)
        comm_time = summary_dict.get("avg_comm_time", 0)
        free_time = summary_dict.get("avg_free_time", 0)
        total_time = compute_time + comm_time + free_time
        
        compute_ratio = compute_time / total_time if total_time > 0 else 0
        comm_ratio = comm_time / total_time if total_time > 0 else 0
        idle_ratio = free_time / total_time if total_time > 0 else 0
        
        # 估算 MFU
        estimated_mfu = compute_ratio * 0.8  # 简化估算
        
        # 确定瓶颈
        main_bottleneck = ""
        bottleneck_impact = 0.0
        if idle_ratio > compute_ratio:
            main_bottleneck = "空闲时间过长（数据加载或调度问题）"
            bottleneck_impact = idle_ratio * 100
        elif comm_ratio > 0.3:
            main_bottleneck = "通信开销过大"
            bottleneck_impact = comm_ratio * 100
        
        # 提取各 Agent 分析文本
        timeline_analysis = ""
        operator_analysis = ""
        memory_analysis = ""
        communication_analysis = ""
        
        for name, result in agent_results.items():
            if hasattr(result, "raw_response"):
                text = result.raw_response or result.summary
            elif isinstance(result, dict):
                text = result.get("raw_response", result.get("summary", ""))
            else:
                text = str(result)
            
            if "timeline" in name.lower():
                timeline_analysis = text
            elif "operator" in name.lower():
                operator_analysis = text
            elif "memory" in name.lower():
                memory_analysis = text
            elif "communication" in name.lower() or "comm" in name.lower():
                communication_analysis = text
        
        # 提取建议
        suggestions = []
        if advisor_report and hasattr(advisor_report, "suggestions"):
            for s in advisor_report.suggestions:
                if hasattr(s, "title"):
                    suggestions.append({
                        "title": s.title,
                        "description": s.description,
                        "priority": s.priority.value if hasattr(s.priority, "value") else str(s.priority),
                        "expected_benefit": s.expected_benefit,
                        "code_example": s.code_example,
                    })
        
        # 构建 ReportData
        report_data = ReportData(
            title="昇腾 NPU 性能分析报告",
            generated_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            profiling_path=profiling_path,
            rank_count=summary_dict.get("rank_count", 1),
            step_count=summary_dict.get("step_count", 0),
            avg_step_time_ms=summary_dict.get("avg_step_time", 0) / 1000,
            estimated_mfu=estimated_mfu,
            compute_ratio=compute_ratio,
            comm_ratio=comm_ratio,
            idle_ratio=idle_ratio,
            overlap_ratio=summary_dict.get("overlap_ratio", 0),
            main_bottleneck=main_bottleneck,
            bottleneck_impact=bottleneck_impact,
            timeline_analysis=timeline_analysis[:500] if timeline_analysis else "",
            operator_analysis=operator_analysis[:500] if operator_analysis else "",
            memory_analysis=memory_analysis[:500] if memory_analysis else "",
            communication_analysis=communication_analysis[:500] if communication_analysis else "",
            suggestions=suggestions,
        )
        
        return self.generate(report_data, format)
    
    def save(
        self,
        content: str,
        output_path: str,
        format: Optional[ReportFormat] = None,
    ):
        """
        保存报告到文件
        
        Args:
            content: 报告内容
            output_path: 输出路径
            format: 报告格式（用于确定文件扩展名）
        """
        path = Path(output_path)
        
        # 确保目录存在
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # 写入文件
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        
        logger.info(f"Report saved to: {path}")
    
    def _dict_to_report_data(self, data: Dict[str, Any]) -> ReportData:
        """将字典转换为 ReportData"""
        return ReportData(
            title=data.get("title", "昇腾 NPU 性能分析报告"),
            generated_at=data.get("generated_at", ""),
            profiling_path=data.get("profiling_path", ""),
            rank_count=data.get("rank_count", 1),
            step_count=data.get("step_count", 0),
            avg_step_time_ms=data.get("avg_step_time_ms", 0),
            estimated_mfu=data.get("estimated_mfu", 0),
            compute_ratio=data.get("compute_ratio", 0),
            comm_ratio=data.get("comm_ratio", 0),
            idle_ratio=data.get("idle_ratio", 0),
            overlap_ratio=data.get("overlap_ratio", 0),
            main_bottleneck=data.get("main_bottleneck", ""),
            bottleneck_impact=data.get("bottleneck_impact", 0),
            timeline_analysis=data.get("timeline_analysis", ""),
            operator_analysis=data.get("operator_analysis", ""),
            memory_analysis=data.get("memory_analysis", ""),
            communication_analysis=data.get("communication_analysis", ""),
            suggestions=data.get("suggestions", []),
        )
    
    def _generate_text(self, data: ReportData) -> str:
        """生成纯文本报告"""
        lines = [
            "=" * 60,
            f"  {data.title}",
            "=" * 60,
            "",
            f"生成时间: {data.generated_at}",
            f"数据路径: {data.profiling_path}",
            "",
            "-" * 60,
            "性能概览",
            "-" * 60,
            f"  Rank 数量: {data.rank_count}",
            f"  Step 数量: {data.step_count}",
            f"  平均 Step 时间: {data.avg_step_time_ms:.2f} ms",
            f"  估算 MFU: {data.estimated_mfu*100:.1f}%",
            "",
            "时间分布:",
            f"  计算: {data.compute_ratio*100:.1f}%",
            f"  通信: {data.comm_ratio*100:.1f}%",
            f"  空闲: {data.idle_ratio*100:.1f}%",
            "",
        ]
        
        if data.main_bottleneck:
            lines.extend([
                "-" * 60,
                "主要瓶颈",
                "-" * 60,
                f"  {data.main_bottleneck}",
                f"  影响: {data.bottleneck_impact:.1f}%",
                "",
            ])
        
        if data.suggestions:
            lines.extend([
                "-" * 60,
                "优化建议",
                "-" * 60,
            ])
            for i, s in enumerate(data.suggestions, 1):
                lines.append(f"  {i}. [{s.get('priority', 'medium').upper()}] {s.get('title', '')}")
            lines.append("")
        
        lines.append("=" * 60)
        
        return "\n".join(lines)
    
    def _generate_json(self, data: ReportData) -> str:
        """生成 JSON 报告"""
        import json
        
        report_dict = {
            "title": data.title,
            "generated_at": data.generated_at,
            "profiling_path": data.profiling_path,
            "overview": {
                "rank_count": data.rank_count,
                "step_count": data.step_count,
                "avg_step_time_ms": data.avg_step_time_ms,
                "estimated_mfu": data.estimated_mfu,
            },
            "time_breakdown": {
                "compute_ratio": data.compute_ratio,
                "comm_ratio": data.comm_ratio,
                "idle_ratio": data.idle_ratio,
                "overlap_ratio": data.overlap_ratio,
            },
            "bottleneck": {
                "main": data.main_bottleneck,
                "impact": data.bottleneck_impact,
            },
            "suggestions": data.suggestions,
        }
        
        return json.dumps(report_dict, ensure_ascii=False, indent=2)
