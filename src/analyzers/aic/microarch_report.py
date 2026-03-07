"""
AIC 微架构分析报告生成器

生成 HTML 格式的 AIC 微架构深度分析报告，包括：
- 综合仪表板
- 指令级分析
- 内存层次分析
- 流水线分析
- 优化建议
"""

import json
import logging
from typing import Dict, List, Optional, Any
from pathlib import Path
from dataclasses import dataclass, field

from src.analyzers.aic.instruction_analyzer import InstructionBottleneck, InstructionAnalyzer
from src.analyzers.aic.memory_hierarchy_analyzer import MemoryHierarchyAnalysis, MemoryHierarchyAnalyzer
from src.analyzers.aic.pipeline_analyzer import PipelineAnalysis, PipelineAnalyzer
from src.data_loader.aic_metrics import ExtendedAICMetrics

logger = logging.getLogger(__name__)


@dataclass
class MicroarchReportData:
    """微架构报告数据结构"""
    summary: Dict[str, Any]
    instruction_analysis: Dict[str, Any]
    memory_analysis: Dict[str, Any]
    pipeline_analysis: Dict[str, Any]
    top_bottlenecks: List[Dict[str, Any]]
    recommendations: List[str]


class MicroarchReportGenerator:
    """
    AIC 微架构报告生成器

    生成交互式 HTML 报告，整合指令、内存、流水线三大分析维度。
    """

    def __init__(self, profiling_path: str):
        """
        初始化报告生成器

        Args:
            profiling_path: Profiling 数据路径
        """
        self.profiling_path = Path(profiling_path)

        # 初始化分析器
        self.instruction_analyzer = InstructionAnalyzer()
        self.memory_analyzer = MemoryHierarchyAnalyzer()
        self.pipeline_analyzer = PipelineAnalyzer()

    def generate_report(
        self,
        metrics_list: List[ExtendedAICMetrics],
        output_path: Optional[str] = None,
        title: str = "AIC 微架构深度分析报告",
    ) -> str:
        """
        生成 HTML 报告

        Args:
            metrics_list: 扩展 AIC 指标列表
            output_path: 输出路径（可选）
            title: 报告标题

        Returns:
            HTML 内容字符串
        """
        # 准备数据
        report_data = self._prepare_report_data(metrics_list)

        # 加载模板
        template = self._load_template()

        # 嵌入数据
        data_json = json.dumps(report_data.__dict__, ensure_ascii=False, default=str)
        html = template.replace("/* DATA_PLACEHOLDER */", f"const REPORT_DATA = {data_json};")
        html = html.replace("{{TITLE}}", title)

        # 保存
        if output_path:
            Path(output_path).write_text(html, encoding="utf-8")
            logger.info(f"Microarchitecture report saved to: {output_path}")

        return html

    def _prepare_report_data(
        self,
        metrics_list: List[ExtendedAICMetrics],
    ) -> MicroarchReportData:
        """准备报告数据"""
        # 批量分析
        instruction_bottlenecks, instruction_summary = self.instruction_analyzer.analyze_batch(metrics_list)
        memory_analyses, memory_summary = self.memory_analyzer.analyze_batch(metrics_list)
        pipeline_analyses, pipeline_summary = self.pipeline_analyzer.analyze_batch(metrics_list)

        # 汇总摘要
        summary = {
            "total_operators": len(metrics_list),
            "analyzed_operators": len(instruction_bottlenecks),
            **instruction_summary,
            **memory_summary,
            **pipeline_summary,
        }

        # Top 瓶颈
        top_bottlenecks = self._get_top_bottlenecks(
            instruction_bottlenecks,
            memory_analyses,
            pipeline_analyses,
        )

        # 综合优化建议
        recommendations = self._generate_comprehensive_recommendations(
            instruction_bottlenecks,
            memory_analyses,
            pipeline_analyses,
        )

        return MicroarchReportData(
            summary=summary,
            instruction_analysis={
                "bottlenecks": [b.to_dict() for b in instruction_bottlenecks[:10]],
                "summary": instruction_summary,
            },
            memory_analysis={
                "analyses": [a.to_dict() for a in memory_analyses[:10]],
                "summary": memory_summary,
            },
            pipeline_analysis={
                "analyses": [a.to_dict() for a in pipeline_analyses[:10]],
                "summary": pipeline_summary,
            },
            top_bottlenecks=top_bottlenecks,
            recommendations=recommendations,
        )

    def _get_top_bottlenecks(
        self,
        instruction_bottlenecks: List[InstructionBottleneck],
        memory_analyses: List[MemoryHierarchyAnalysis],
        pipeline_analyses: List[PipelineAnalysis],
    ) -> List[Dict[str, Any]]:
        """获取 Top 瓶颈"""
        bottlenecks = []

        for ib in instruction_bottlenecks:
            if ib.severity in ("critical", "high"):
                bottlenecks.append({
                    "type": "instruction",
                    "operator": "算子",
                    "bottleneck": ib.bottleneck_type.value,
                    "severity": ib.severity,
                    "score": ib.score,
                })

        for ma in memory_analyses:
            if ma.severity in ("critical", "high"):
                bottlenecks.append({
                    "type": "memory",
                    "operator": "算子",
                    "bottleneck": ma.bottleneck_type.value,
                    "severity": ma.severity,
                    "score": ma.score,
                })

        for pa in pipeline_analyses:
            if pa.severity in ("critical", "high"):
                bottlenecks.append({
                    "type": "pipeline",
                    "operator": "算子",
                    "bottleneck": pa.bottleneck_type.value,
                    "severity": pa.severity,
                    "score": pa.score,
                })

        # 按评分排序
        bottlenecks.sort(key=lambda x: x["score"], reverse=True)
        return bottlenecks[:10]

    def _generate_comprehensive_recommendations(
        self,
        instruction_bottlenecks: List[InstructionBottleneck],
        memory_analyses: List[MemoryHierarchyAnalysis],
        pipeline_analyses: List[PipelineAnalysis],
    ) -> List[str]:
        """生成综合优化建议"""
        recommendations = []

        # 统计主要瓶颈类型
        instruction_critical = sum(1 for b in instruction_bottlenecks if b.severity == "critical")
        memory_critical = sum(1 for a in memory_analyses if a.severity == "critical")
        pipeline_critical = sum(1 for a in pipeline_analyses if a.severity == "critical")

        # 优先级排序
        if instruction_critical >= 3:
            recommendations.append(
                f"【关键】检测到 {instruction_critical} 个算子存在严重指令级瓶颈，"
                "优先优化 Cube 利用率和指令发射率"
            )

        if memory_critical >= 3:
            recommendations.append(
                f"【关键】检测到 {memory_critical} 个算子存在严重内存瓶颈，"
                "优先优化 L2 缓存命中率和 UB 使用率"
            )

        if pipeline_critical >= 3:
            recommendations.append(
                f"【关键】检测到 {pipeline_critical} 个算子存在严重流水线瓶颈，"
                "优先解决依赖停顿和资源冲突问题"
            )

        # 收集各分析器的建议
        seen = set()
        for b in instruction_bottlenecks[:5]:
            for rec in b.recommendations[:2]:
                if rec not in seen:
                    recommendations.append(f"• {rec}")
                    seen.add(rec)

        for a in memory_analyses[:5]:
            for rec in a.recommendations[:2]:
                if rec not in seen:
                    recommendations.append(f"• {rec}")
                    seen.add(rec)

        for a in pipeline_analyses[:5]:
            for rec in a.recommendations[:2]:
                if rec not in seen:
                    recommendations.append(f"• {rec}")
                    seen.add(rec)

        return recommendations[:20]

    def _load_template(self) -> str:
        """加载 HTML 模板"""
        template_path = (
            self.profiling_path.parent / "src" / "web" / "static" / "aic_dashboard" / "index.html"
        )

        if template_path.exists():
            return template_path.read_text(encoding="utf-8")

        # 使用内嵌模板
        return self._get_embedded_template()

    def _get_embedded_template(self) -> str:
        """获取内嵌的 HTML 模板"""
        return """<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{TITLE}}</title>
    <style>
        :root {
            --primary-color: #673AB7;
            --success-color: #4CAF50;
            --warning-color: #FF9800;
            --danger-color: #F44336;
            --bg-color: #FAFAFA;
            --card-bg: #FFFFFF;
            --text-color: #212121;
        }
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: var(--bg-color);
            color: var(--text-color);
            line-height: 1.6;
            padding: 20px;
        }
        .container { max-width: 1400px; margin: 0 auto; }
        .header {
            background: linear-gradient(135deg, var(--primary-color), #9C27B0);
            color: white;
            padding: 30px;
            border-radius: 12px;
            margin-bottom: 20px;
        }
        .header h1 { font-size: 24px; margin-bottom: 8px; }
        .summary-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 16px;
            margin: 20px 0;
        }
        .summary-card {
            background: var(--card-bg);
            border-radius: 8px;
            padding: 16px;
            text-align: center;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .summary-value { font-size: 28px; font-weight: bold; color: var(--primary-color); }
        .summary-label { font-size: 13px; color: #666; margin-top: 4px; }
        .section {
            background: var(--card-bg);
            border-radius: 12px;
            padding: 24px;
            margin-bottom: 20px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }
        .section h2 {
            font-size: 18px;
            margin-bottom: 16px;
            padding-bottom: 8px;
            border-bottom: 2px solid var(--primary-color);
        }
        .bottleneck-list { list-style: none; }
        .bottleneck-item {
            padding: 12px;
            margin: 8px 0;
            border-radius: 8px;
            border-left: 4px solid var(--primary-color);
            background: #F5F5F5;
        }
        .bottleneck-item.critical { border-left-color: var(--danger-color); background: #FFEBEE; }
        .bottleneck-item.high { border-left-color: var(--warning-color); background: #FFF3E0; }
        .recommendations { list-style: none; }
        .recommendations li {
            padding: 8px 12px;
            margin: 4px 0;
            border-radius: 4px;
            background: #E8EAF6;
        }
        .badge {
            display: inline-block;
            padding: 2px 8px;
            border-radius: 12px;
            font-size: 11px;
            font-weight: bold;
        }
        .badge.critical { background: #FFEBEE; color: #C62828; }
        .badge.high { background: #FFF3E0; color: #EF6C00; }
        .badge.medium { background: #E3F2FD; color: #1565C0; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🔬 AIC 微架构深度分析报告</h1>
            <p>昇腾 AI Core 硬件级性能分析</p>
        </div>

        <div class="section">
            <h2>📊 分析概览</h2>
            <div class="summary-grid" id="summary-grid"></div>
        </div>

        <div class="section">
            <h2>⚠️ Top 瓶颈</h2>
            <ul class="bottleneck-list" id="bottleneck-list"></ul>
        </div>

        <div class="section">
            <h2>💡 优化建议</h2>
            <ul class="recommendations" id="recommendations"></ul>
        </div>
    </div>

    <script>
        /* DATA_PLACEHOLDER */

        // 渲染摘要卡片
        function renderSummary() {
            const summary = REPORT_DATA.summary;
            const grid = document.getElementById('summary-grid');

            const metrics = [
                { label: '分析算子数', value: summary.total_operators },
                { label: '严重瓶颈', value: summary.critical_count, critical: true },
                { label: '高优先级', value: summary.high_count },
                { label: 'Cube 利用率', value: summary.avg_cube_util?.toFixed(1) + '%' },
                { label: 'L2 命中率', value: summary.avg_l2_hit_rate?.toFixed(1) + '%' },
                { label: '流水线利用率', value: summary.avg_pipe_util?.toFixed(1) + '%' },
            ];

            grid.innerHTML = metrics.map(m => `
                <div class="summary-card">
                    <div class="summary-value" ${m.critical ? 'style="color: var(--danger-color)"' : ''}>${m.value}</div>
                    <div class="summary-label">${m.label}</div>
                </div>
            `).join('');
        }

        // 渲染瓶颈列表
        function renderBottlenecks() {
            const list = document.getElementById('bottleneck-list');
            list.innerHTML = REPORT_DATA.top_bottlenecks.map(b => `
                <li class="bottleneck-item ${b.severity}">
                    <strong>${b.type} 瓶颈</strong> - ${b.bottleneck}
                    <span class="badge ${b.severity}">${b.severity}</span>
                    <span>评分: ${b.score.toFixed(0)}</span>
                </li>
            `).join('');
        }

        // 渲染建议
        function renderRecommendations() {
            const list = document.getElementById('recommendations');
            list.innerHTML = REPORT_DATA.recommendations.map(r => `
                <li>${r}</li>
            `).join('');
        }

        // 初始化
        renderSummary();
        renderBottlenecks();
        renderRecommendations();
    </script>
</body>
</html>"""


def generate_microarch_report(
    profiling_path: str,
    metrics_list: List[ExtendedAICMetrics],
    output_path: Optional[str] = None,
) -> str:
    """
    便捷函数：生成微架构报告

    Args:
        profiling_path: Profiling 数据路径
        metrics_list: 扩展 AIC 指标列表
        output_path: 输出路径（可选）

    Returns:
        HTML 内容字符串
    """
    generator = MicroarchReportGenerator(profiling_path)
    return generator.generate_report(metrics_list, output_path=output_path)