"""
Comparison Orchestrator

协调完整的 Profiling 对比分析流程：
1. 加载两份 Profiling 数据
2. 相似度检测
3. 多层级差异分析
4. LLM 根因分析
5. 生成对比报告
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field

from src.data_loader.profiling_loader import ProfilingLoader
from src.data_loader.data_summarizer import DataSummarizer, ProfilingSummary
from src.analyzers.similarity_checker import SimilarityChecker, SimilarityResult, ComparabilityLevel
from src.analyzers.profiling_diff import ProfilingDiffEngine, ProfilingDiff
from src.agents.comparison_agent import ComparisonAdvisorAgent
from src.llm.llm_interface import LLMConfig, LLMFactory
from src.report.report_generator import ReportFormat

logger = logging.getLogger(__name__)


@dataclass
class ComparisonReport:
    """对比分析报告"""
    success: bool
    summary: str

    # 输入信息
    path_a: str = ""
    path_b: str = ""
    label_a: str = "基准版本 (A)"
    label_b: str = "当前版本 (B)"

    # 分析结果
    similarity: Optional[SimilarityResult] = None
    diff: Optional[ProfilingDiff] = None
    profiling_summary_a: Optional[ProfilingSummary] = None
    profiling_summary_b: Optional[ProfilingSummary] = None

    # LLM 分析
    advisor_analysis: str = ""
    recommendations: List[str] = field(default_factory=list)

    # 最终报告
    final_report: str = ""

    # 错误
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "summary": self.summary,
            "path_a": self.path_a,
            "path_b": self.path_b,
            "label_a": self.label_a,
            "label_b": self.label_b,
            "similarity": self.similarity.to_dict() if self.similarity else None,
            "diff": self.diff.to_dict() if self.diff else None,
            "advisor_analysis": self.advisor_analysis,
            "recommendations": self.recommendations,
            "error": self.error,
        }


class ComparisonOrchestrator:
    """
    Profiling 对比分析编排器

    Usage:
        orchestrator = ComparisonOrchestrator(
            path_a="/path/to/profiling_a",
            path_b="/path/to/profiling_b",
            label_a="v2.0 升级前",
            label_b="v2.1 升级后",
        )
        report = await orchestrator.run()
    """

    def __init__(
        self,
        path_a: str,
        path_b: str,
        label_a: str = "基准版本 (A)",
        label_b: str = "当前版本 (B)",
        llm_config: Optional[LLMConfig] = None,
        force: bool = False,
    ):
        """
        Args:
            path_a: 基准 Profiling 路径
            path_b: 当前 Profiling 路径
            label_a: A 的标签
            label_b: B 的标签
            llm_config: LLM 配置
            force: 是否跳过相似度检查强制对比
        """
        self.path_a = path_a
        self.path_b = path_b
        self.label_a = label_a
        self.label_b = label_b
        self.llm_config = llm_config or LLMConfig()
        self.force = force

        # 初始化组件
        self.loader_a = ProfilingLoader(path_a)
        self.loader_b = ProfilingLoader(path_b)
        self.summarizer_a = DataSummarizer(self.loader_a)
        self.summarizer_b = DataSummarizer(self.loader_b)
        self.similarity_checker = SimilarityChecker()
        self.diff_engine = ProfilingDiffEngine()
        self.llm = LLMFactory.create(self.llm_config)
        self.advisor = ComparisonAdvisorAgent(self.llm)

    async def run(self, output_format: ReportFormat = ReportFormat.MARKDOWN) -> ComparisonReport:
        """
        执行完整的对比分析

        Args:
            output_format: 输出格式

        Returns:
            ComparisonReport
        """
        logger.info(f"Starting comparison: {self.path_a} vs {self.path_b}")

        report = ComparisonReport(
            success=False,
            summary="",
            path_a=self.path_a,
            path_b=self.path_b,
            label_a=self.label_a,
            label_b=self.label_b,
        )

        try:
            # -------------------------------------------------------
            # Step 1: 检测并加载数据
            # -------------------------------------------------------
            logger.info("Step 1: Detecting and loading profiling data...")
            info_a = self.loader_a.detect()
            info_b = self.loader_b.detect()

            if info_a.data_type == "unknown":
                report.error = f"无法识别 Profiling A 的数据格式: {self.path_a}"
                report.summary = report.error
                return report

            if info_b.data_type == "unknown":
                report.error = f"无法识别 Profiling B 的数据格式: {self.path_b}"
                report.summary = report.error
                return report

            logger.info(
                f"Profiling A: {info_a.data_type}, {info_a.rank_count} ranks | "
                f"Profiling B: {info_b.data_type}, {info_b.rank_count} ranks"
            )

            # -------------------------------------------------------
            # Step 2: 生成摘要
            # -------------------------------------------------------
            logger.info("Step 2: Generating summaries...")
            summary_a = self.summarizer_a.summarize()
            summary_b = self.summarizer_b.summarize()
            report.profiling_summary_a = summary_a
            report.profiling_summary_b = summary_b

            # 获取算子数据用于深度比较
            operators_a = self.loader_a.get_top_kernels(top_n=100)
            operators_b = self.loader_b.get_top_kernels(top_n=100)

            # -------------------------------------------------------
            # Step 3: 相似度检测
            # -------------------------------------------------------
            logger.info("Step 3: Checking similarity...")
            similarity = self.similarity_checker.check(
                info_a, info_b,
                summary_a, summary_b,
                operators_a, operators_b,
            )
            report.similarity = similarity

            logger.info(
                f"Similarity score: {similarity.overall_score:.2f} "
                f"({similarity.level.value})"
            )

            if not self.force and not similarity.is_comparable():
                report.summary = (
                    f"两个 Profiling 数据不适合对比 "
                    f"(相似度: {similarity.overall_score * 100:.0f}%)。"
                    f"\n{similarity.summary}"
                )
                report.error = "NOT_COMPARABLE"
                report.final_report = self._generate_not_comparable_report(
                    similarity, info_a, info_b, summary_a, summary_b
                )
                return report

            # -------------------------------------------------------
            # Step 4: 多层级差异分析
            # -------------------------------------------------------
            logger.info("Step 4: Computing multi-level diff...")
            diff = self.diff_engine.compute(
                summary_a, summary_b,
                loader_a=self.loader_a,
                loader_b=self.loader_b,
                operators_a=operators_a,
                operators_b=operators_b,
            )
            report.diff = diff

            logger.info(f"Diff verdict: {diff.overall_verdict}")

            # -------------------------------------------------------
            # Step 5: LLM 根因分析
            # -------------------------------------------------------
            logger.info("Step 5: Running comparison advisor agent...")
            advisor_data = {
                "diff": diff,
                "diff_text": diff.to_prompt_text(),
                "similarity_text": similarity.to_markdown(),
                "label_a": self.label_a,
                "label_b": self.label_b,
                "path_a": self.path_a,
                "path_b": self.path_b,
                "rank_a": summary_a.rank_count,
                "rank_b": summary_b.rank_count,
                "step_a": summary_a.step_count,
                "step_b": summary_b.step_count,
                "summary_a": summary_a.to_dict(),
                "summary_b": summary_b.to_dict(),
            }

            advisor_result = await self.advisor.analyze(advisor_data)

            if advisor_result.success:
                report.advisor_analysis = advisor_result.raw_response or ""
                report.recommendations = advisor_result.recommendations
            else:
                logger.warning(f"Advisor analysis failed: {advisor_result.error}")

            # -------------------------------------------------------
            # Step 6: 生成对比报告
            # -------------------------------------------------------
            logger.info("Step 6: Generating comparison report...")
            report.final_report = self._generate_comparison_report(
                report, output_format
            )

            report.success = True
            verdict_map = {
                "improved": "性能提升",
                "degraded": "性能劣化",
                "mixed": "喜忧参半",
                "unchanged": "基本不变",
            }
            report.summary = (
                f"对比分析完成: {verdict_map.get(diff.overall_verdict, diff.overall_verdict)}。"
                f" {self.label_a} vs {self.label_b}"
            )

            return report

        except Exception as e:
            logger.error(f"Comparison failed: {e}", exc_info=True)
            report.error = str(e)
            report.summary = f"对比分析失败: {e}"
            return report

    def _generate_not_comparable_report(
        self,
        similarity: SimilarityResult,
        info_a, info_b,
        summary_a: ProfilingSummary,
        summary_b: ProfilingSummary,
    ) -> str:
        """生成不可对比报告"""
        lines = [
            "# Profiling 对比分析报告",
            "",
            "## 结论: 不建议对比",
            "",
            f"> {similarity.summary}",
            "",
            similarity.to_markdown(),
            "",
            "## Profiling A 基本信息",
            f"- 路径: `{info_a.path}`",
            f"- 数据类型: {info_a.data_type}",
            f"- 框架: {info_a.framework}",
            f"- Rank 数: {info_a.rank_count}",
            f"- Step 数: {summary_a.step_count}",
            "",
            "## Profiling B 基本信息",
            f"- 路径: `{info_b.path}`",
            f"- 数据类型: {info_b.data_type}",
            f"- 框架: {info_b.framework}",
            f"- Rank 数: {info_b.rank_count}",
            f"- Step 数: {summary_b.step_count}",
            "",
            "---",
            "",
            "**建议**: 请确认两次 Profiling 是否来自同一训练任务的不同版本。"
            "如确认需要对比，可使用 `--force` 选项跳过相似度检查。",
        ]
        return "\n".join(lines)

    def _generate_comparison_report(
        self,
        report: "ComparisonReport",
        output_format: ReportFormat,
    ) -> str:
        """生成完整的对比报告"""
        if output_format == ReportFormat.HTML:
            return self._generate_html_report(report)
        return self._generate_markdown_report(report)

    def _generate_markdown_report(self, report: "ComparisonReport") -> str:
        """生成 Markdown 格式的对比报告"""
        diff = report.diff
        sim = report.similarity
        summary_a = report.profiling_summary_a
        summary_b = report.profiling_summary_b

        verdict_map = {
            "improved": "性能提升 ✅",
            "degraded": "性能劣化 ⚠️",
            "mixed": "喜忧参半 ⚖️",
            "unchanged": "基本不变 ➡️",
        }
        verdict_text = verdict_map.get(diff.overall_verdict, diff.overall_verdict) if diff else "N/A"

        lines = [
            "# Profiling 对比分析报告",
            "",
            f"**生成时间**: {self._now()}",
            "",
            "---",
            "",
            "## 1. 对比概览",
            "",
            f"| 项目 | {report.label_a} | {report.label_b} |",
            "|------|---------|---------|",
            f"| 路径 | `{report.path_a}` | `{report.path_b}` |",
        ]

        if summary_a and summary_b:
            lines.extend([
                f"| Rank 数 | {summary_a.rank_count} | {summary_b.rank_count} |",
                f"| Step 数 | {summary_a.step_count} | {summary_b.step_count} |",
                f"| 平均 Step 时间 | {summary_a.avg_step_time / 1000:.2f} ms | {summary_b.avg_step_time / 1000:.2f} ms |",
                f"| 框架 | {summary_a.framework} | {summary_b.framework} |",
            ])

        lines.extend([
            "",
            f"**整体判断**: {verdict_text}",
            "",
        ])

        # 主要变化
        if diff and diff.primary_changes:
            lines.append("**主要变化:**")
            for change in diff.primary_changes:
                lines.append(f"- {change}")
            lines.append("")

        # 相似度评估
        if sim:
            lines.append("---")
            lines.append("")
            lines.append("## 2. 相似度评估")
            lines.append("")
            lines.append(sim.to_markdown())
            lines.append("")

        # Summary 级对比
        if diff:
            lines.append("---")
            lines.append("")
            lines.append("## 3. 核心指标对比")
            lines.append("")
            lines.append(f"| 指标 | {report.label_a} | {report.label_b} | 变化 | 评价 |")
            lines.append("|------|---------|---------|------|------|")

            for change in diff.summary_diff.all_changes:
                icon = "✅" if change.is_improvement else ("⚠️" if abs(change.change_pct) > 5 else "➡️")
                sign = "+" if change.change_pct > 0 else ""
                lines.append(
                    f"| {change.label} | "
                    f"{change.value_a:.4g}{change.unit} | "
                    f"{change.value_b:.4g}{change.unit} | "
                    f"{sign}{change.change_pct:.1f}% | "
                    f"{icon} |"
                )
            lines.append("")

        # Operator 级对比
        if diff and diff.operator_diff:
            od = diff.operator_diff
            lines.append("---")
            lines.append("")
            lines.append("## 4. 算子级对比")
            lines.append("")
            lines.append(f"- 共同算子: {od.common_operator_count} 个")
            lines.append(f"- 新增算子: {len(od.new_operators)} 个")
            lines.append(f"- 移除算子: {len(od.removed_operators)} 个")
            lines.append("")

            if od.top_regressions:
                lines.append("### 4.1 Top 劣化算子")
                lines.append("")
                lines.append("| 算子名称 | A 耗时 | B 耗时 | 变化 |")
                lines.append("|---------|--------|--------|------|")
                for op in od.top_regressions[:15]:
                    lines.append(
                        f"| {op.name} | {op.dur_a / 1000:.3f}ms | "
                        f"{op.dur_b / 1000:.3f}ms | "
                        f"+{op.change_pct:.1f}% |"
                    )
                lines.append("")

            if od.top_improvements:
                lines.append("### 4.2 Top 改善算子")
                lines.append("")
                lines.append("| 算子名称 | A 耗时 | B 耗时 | 变化 |")
                lines.append("|---------|--------|--------|------|")
                for op in od.top_improvements[:15]:
                    lines.append(
                        f"| {op.name} | {op.dur_a / 1000:.3f}ms | "
                        f"{op.dur_b / 1000:.3f}ms | "
                        f"{op.change_pct:.1f}% |"
                    )
                lines.append("")

            if od.new_operators:
                lines.append("### 4.3 新增算子")
                lines.append("")
                for op in od.new_operators[:10]:
                    lines.append(f"- **{op.name}**: {op.dur_b / 1000:.3f}ms")
                lines.append("")

        # Timeline 级对比
        if diff and diff.timeline_diff:
            td = diff.timeline_diff
            lines.append("---")
            lines.append("")
            lines.append("## 5. Timeline 级对比")
            lines.append("")
            stability = "更稳定 ✅" if td.stability_improved else "波动增大 ⚠️"
            lines.append(f"| 指标 | {report.label_a} | {report.label_b} | 评价 |")
            lines.append("|------|---------|---------|------|")
            lines.append(
                f"| Step 标准差 | {td.step_time_std_a / 1000:.3f}ms | "
                f"{td.step_time_std_b / 1000:.3f}ms | {stability} |"
            )
            lines.append(
                f"| 变异系数 (CV) | {td.step_time_cv_a:.4f} | "
                f"{td.step_time_cv_b:.4f} | |"
            )
            lines.append("")

        # Communication 级对比
        if diff and diff.comm_diff:
            cd = diff.comm_diff
            if cd.total_comm_time_change or cd.comm_pattern_changes:
                lines.append("---")
                lines.append("")
                lines.append("## 6. 通信级对比")
                lines.append("")
                if cd.total_comm_time_change:
                    c = cd.total_comm_time_change
                    lines.append(
                        f"- 通信总时间: {c.value_a / 1000:.2f}ms → {c.value_b / 1000:.2f}ms "
                        f"({c.change_pct:+.1f}%)"
                    )
                if cd.overlap_ratio_change:
                    c = cd.overlap_ratio_change
                    lines.append(
                        f"- 通信掩盖率: {c.value_a:.1f}% → {c.value_b:.1f}% "
                        f"({c.change_pct:+.1f}%)"
                    )
                for p in cd.comm_pattern_changes:
                    lines.append(f"- {p}")
                lines.append("")

        # LLM 分析
        if report.advisor_analysis:
            lines.append("---")
            lines.append("")
            lines.append("## 7. 深度根因分析")
            lines.append("")
            lines.append(report.advisor_analysis)
            lines.append("")

        # 优化建议
        if report.recommendations:
            lines.append("---")
            lines.append("")
            lines.append("## 8. 优化建议")
            lines.append("")
            for i, rec in enumerate(report.recommendations, 1):
                lines.append(f"{i}. {rec}")
            lines.append("")

        return "\n".join(lines)

    def _generate_html_report(self, report: "ComparisonReport") -> str:
        """生成 HTML 格式的对比报告（基于 Markdown 转换）"""
        # 先生成 Markdown，再包装为 HTML
        md_content = self._generate_markdown_report(report)

        # 简单的 Markdown -> HTML 转换
        html_body = self._md_to_html(md_content)

        return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Profiling 对比分析报告</title>
    <style>
        :root {{
            --primary: #1a73e8;
            --success: #34a853;
            --warning: #fbbc04;
            --danger: #ea4335;
            --bg: #f8f9fa;
            --card-bg: #ffffff;
            --text: #202124;
            --text-secondary: #5f6368;
            --border: #dadce0;
        }}
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            line-height: 1.6;
            color: var(--text);
            background: var(--bg);
            padding: 2rem;
        }}
        .container {{
            max-width: 1100px;
            margin: 0 auto;
            background: var(--card-bg);
            border-radius: 12px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.12);
            padding: 2rem 3rem;
        }}
        h1 {{ color: var(--primary); margin-bottom: 1rem; font-size: 1.8rem; }}
        h2 {{ color: var(--text); margin: 1.5rem 0 0.8rem; font-size: 1.3rem; border-bottom: 2px solid var(--border); padding-bottom: 0.3rem; }}
        h3 {{ color: var(--text-secondary); margin: 1rem 0 0.5rem; font-size: 1.1rem; }}
        table {{ border-collapse: collapse; width: 100%; margin: 0.5rem 0 1rem; }}
        th, td {{ border: 1px solid var(--border); padding: 8px 12px; text-align: left; font-size: 0.9rem; }}
        th {{ background: #f1f3f4; font-weight: 600; }}
        tr:nth-child(even) {{ background: #fafafa; }}
        code {{ background: #f1f3f4; padding: 2px 6px; border-radius: 4px; font-size: 0.85rem; }}
        blockquote {{ border-left: 4px solid var(--primary); padding: 0.5rem 1rem; margin: 1rem 0; background: #e8f0fe; border-radius: 0 4px 4px 0; }}
        ul, ol {{ padding-left: 1.5rem; margin: 0.5rem 0; }}
        li {{ margin: 0.3rem 0; }}
        hr {{ border: none; border-top: 1px solid var(--border); margin: 1.5rem 0; }}
        p {{ margin: 0.5rem 0; }}
        strong {{ color: var(--text); }}
    </style>
</head>
<body>
    <div class="container">
        {html_body}
    </div>
</body>
</html>"""

    def _md_to_html(self, md: str) -> str:
        """简单的 Markdown -> HTML 转换"""
        import re
        lines = md.split("\n")
        html_lines = []
        in_table = False
        in_list = False
        in_blockquote = False

        for line in lines:
            stripped = line.strip()

            # 空行
            if not stripped:
                if in_list:
                    html_lines.append("</ul>")
                    in_list = False
                if in_blockquote:
                    html_lines.append("</blockquote>")
                    in_blockquote = False
                if in_table:
                    html_lines.append("</tbody></table>")
                    in_table = False
                html_lines.append("")
                continue

            # 分割线
            if stripped == "---":
                html_lines.append("<hr>")
                continue

            # 标题
            if stripped.startswith("# "):
                html_lines.append(f"<h1>{self._md_inline(stripped[2:])}</h1>")
                continue
            if stripped.startswith("## "):
                html_lines.append(f"<h2>{self._md_inline(stripped[3:])}</h2>")
                continue
            if stripped.startswith("### "):
                html_lines.append(f"<h3>{self._md_inline(stripped[4:])}</h3>")
                continue

            # 表格
            if stripped.startswith("|"):
                cells = [c.strip() for c in stripped.split("|")[1:-1]]
                if all(set(c) <= set("-: ") for c in cells):
                    continue  # 分隔行
                if not in_table:
                    html_lines.append("<table><thead><tr>")
                    for c in cells:
                        html_lines.append(f"<th>{self._md_inline(c)}</th>")
                    html_lines.append("</tr></thead><tbody>")
                    in_table = True
                else:
                    html_lines.append("<tr>")
                    for c in cells:
                        html_lines.append(f"<td>{self._md_inline(c)}</td>")
                    html_lines.append("</tr>")
                continue

            # 引用
            if stripped.startswith("> "):
                if not in_blockquote:
                    html_lines.append("<blockquote>")
                    in_blockquote = True
                html_lines.append(f"<p>{self._md_inline(stripped[2:])}</p>")
                continue

            # 列表
            if stripped.startswith("- ") or stripped.startswith("* "):
                if not in_list:
                    html_lines.append("<ul>")
                    in_list = True
                html_lines.append(f"<li>{self._md_inline(stripped[2:])}</li>")
                continue

            # 有序列表
            if re.match(r"^\d+\.\s", stripped):
                content = re.sub(r"^\d+\.\s", "", stripped)
                if not in_list:
                    html_lines.append("<ul>")
                    in_list = True
                html_lines.append(f"<li>{self._md_inline(content)}</li>")
                continue

            # 普通段落
            html_lines.append(f"<p>{self._md_inline(stripped)}</p>")

        # 关闭未关闭的标签
        if in_table:
            html_lines.append("</tbody></table>")
        if in_list:
            html_lines.append("</ul>")
        if in_blockquote:
            html_lines.append("</blockquote>")

        return "\n".join(html_lines)

    @staticmethod
    def _md_inline(text: str) -> str:
        """处理内联 Markdown 标记"""
        import re
        # Bold
        text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)
        # Code
        text = re.sub(r"`(.+?)`", r"<code>\1</code>", text)
        # Emoji (preserve as-is)
        return text

    @staticmethod
    def _now() -> str:
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


async def run_comparison(
    path_a: str,
    path_b: str,
    label_a: str = "基准版本 (A)",
    label_b: str = "当前版本 (B)",
    llm_backend: str = "mock",
    force: bool = False,
) -> ComparisonReport:
    """
    便捷的对比分析入口

    Args:
        path_a: 基准 Profiling 路径
        path_b: 当前 Profiling 路径
        label_a: A 的标签
        label_b: B 的标签
        llm_backend: LLM 后端
        force: 是否强制对比

    Returns:
        ComparisonReport
    """
    config = LLMConfig(backend=llm_backend)
    orchestrator = ComparisonOrchestrator(
        path_a=path_a,
        path_b=path_b,
        label_a=label_a,
        label_b=label_b,
        llm_config=config,
        force=force,
    )
    return await orchestrator.run()
