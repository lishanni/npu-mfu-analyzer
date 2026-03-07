"""
CLI 入口 - 命令行接口

Usage:
    npu-analyzer analyze /path/to/profiling
    npu-analyzer analyze /path/to/profiling --output report.md
    npu-analyzer analyze /path/to/profiling --backend mock
"""

import asyncio
import logging
import sys
from pathlib import Path
from typing import Optional, Any
from datetime import datetime

import click

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@click.group()
@click.version_option(version="0.1.0", prog_name="npu-mfu-analyzer")
def cli():
    """NPU MFU Analyzer - 昇腾 NPU 大模型训练性能分析工具"""
    pass


@cli.command()
@click.argument("profiling_path", type=click.Path(exists=True))
@click.option(
    "--output", "-o",
    type=click.Path(),
    help="输出报告路径（默认输出到终端）"
)
@click.option(
    "--backend", "-b",
    type=click.Choice(["openai", "claude", "ollama", "deepseek", "mock"]),
    default="openai",
    help="LLM 后端: openai/claude/ollama/deepseek/mock"
)
@click.option(
    "--model", "-m",
    type=str,
    default=None,
    help="LLM 模型名称"
)
@click.option(
    "--format", "-f",
    type=click.Choice(["markdown", "html", "md"]),
    default="markdown",
    help="报告格式: markdown/html (默认: markdown)"
)
@click.option(
    "--verbose", "-v",
    is_flag=True,
    help="详细输出"
)
@click.option(
    "--comm-matrix/--no-comm-matrix",
    default=True,
    help="启用通信矩阵分析 (默认: 启用)"
)
@click.option(
    "--comm-matrix-output",
    type=str,
    default=None,
    help="通信矩阵可视化 HTML 输出路径 (单独文件)"
)
def analyze(
    profiling_path: str,
    output: Optional[str],
    backend: str,
    model: Optional[str],
    format: str,
    verbose: bool,
    comm_matrix: bool,
    comm_matrix_output: Optional[str],
):
    """
    分析 Profiling 数据，生成性能报告

    PROFILING_PATH: Profiling 数据目录路径

    示例:
        npu-analyzer analyze /path/to/profiling
        npu-analyzer analyze /path/to/profiling -b claude -m GLM-4.7
        npu-analyzer analyze /path/to/profiling -o report.md -f markdown
        npu-analyzer analyze /path/to/profiling -o report.html -f html
        npu-analyzer analyze /path/to/profiling --comm-matrix --comm-matrix-output comm.html
    """
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # 标准化 format 参数
    if format == "md":
        format = "markdown"

    click.echo(f"🚀 开始分析 Profiling 数据: {profiling_path}")
    click.echo(f"   LLM 后端: {backend}")
    click.echo(f"   报告格式: {format}")

    # 运行分析
    try:
        report = asyncio.run(_run_analysis(profiling_path, backend, model, comm_matrix))

        if report.success:
            click.echo(click.style("✅ 分析完成!", fg="green"))

            # 根据格式输出报告
            if format == "html":
                # 使用 orchestrator 生成的 HTML 报告（包含 MFU 指标）
                report_text = report.final_report or _generate_html_report_content(profiling_path, report)

                if output:
                    output_path = Path(output)
                    # 如果是目录，生成文件名
                    if output_path.is_dir():
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        output_path = output_path / f"npu_mfu_report_{timestamp}.html"
                    # 确保有正确的扩展名
                    elif output_path.suffix != '.html':
                        output_path = output_path.with_suffix('.html')
                    output_path.write_text(report_text, encoding="utf-8")
                    click.echo(f"📄 HTML 报告已保存到: {output_path}")
                else:
                    # 默认输出到当前目录
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    output_path = Path(f"npu_mfu_report_{timestamp}.html")
                    output_path.write_text(report_text, encoding="utf-8")
                    click.echo(f"📄 HTML 报告已保存到: {output_path}")
            else:
                # Markdown 格式
                report_text = report.to_markdown()

                if output:
                    output_path = Path(output)
                    # 确保文件扩展名正确
                    if not output_path.suffix or output_path.suffix not in ['.md', '.markdown']:
                        output_path = output_path.with_suffix('.md')
                    output_path.write_text(report_text, encoding="utf-8")
                    click.echo(f"📄 报告已保存到: {output_path}")
                else:
                    click.echo("\n" + "=" * 60)
                    click.echo(report_text)
                    click.echo("=" * 60)

            # 输出摘要
            if report.recommendations:
                click.echo(click.style("\n📌 Top 优化建议:", fg="yellow"))
                for i, rec in enumerate(report.recommendations[:5], 1):
                    click.echo(f"   {i}. {rec}")

            # 处理通信矩阵可视化输出
            if report.comm_matrix_html:
                if comm_matrix_output:
                    output_path = Path(comm_matrix_output)
                    output_path.write_text(report.comm_matrix_html, encoding="utf-8")
                    click.echo(f"🔗 通信矩阵可视化已保存到: {output_path}")
                elif format == "html":
                    # 如果是 HTML 格式且未指定单独输出路径，保存到默认文件
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    default_path = Path(f"communication_matrix_{timestamp}.html")
                    default_path.write_text(report.comm_matrix_html, encoding="utf-8")
                    click.echo(f"🔗 通信矩阵可视化已保存到: {default_path}")
        else:
            click.echo(click.style(f"❌ 分析失败: {report.error}", fg="red"))
            sys.exit(1)

    except Exception as e:
        click.echo(click.style(f"❌ 发生错误: {e}", fg="red"))
        if verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


async def _run_analysis(profiling_path: str, backend: str, model: Optional[str], enable_comm_matrix: bool = True):
    """执行分析"""
    from src.llm.llm_interface import LLMConfig
    from src.agents.orchestrator import Orchestrator

    # 构建 LLM 配置
    config = LLMConfig(backend=backend)
    if model:
        config.model = model

    # 创建 Orchestrator 并运行
    orchestrator = Orchestrator(
        profiling_path,
        llm_config=config,
        enable_comm_matrix=enable_comm_matrix,
    )
    return await orchestrator.run()


def _generate_html_report(profiling_path: str, output: Optional[str], report: Any = None) -> Optional[Path]:
    """
    生成 HTML 格式报告

    Args:
        profiling_path: Profiling 数据路径
        output: 输出路径（可选）
        report: AnalysisReport 对象（可选，包含 Agent 分析结果）

    Returns:
        输出文件路径，失败返回 None
    """
    from src.data_loader.profiling_loader import ProfilingLoader
    from src.agents.operator_agent import FusionAnalyzer
    from datetime import datetime

    try:
        # 1. 加载数据
        loader = ProfilingLoader(profiling_path)
        info = loader.detect()
        top_kernels = loader.get_top_kernels(top_n=100)

        # 2. 融合分析
        analyzer = FusionAnalyzer()
        fusion_opportunities = analyzer.detect_opportunities(
            all_operators=top_kernels,
            timeline_data=None
        )

        # 3. 提取 Agent 分析结果
        agent_results = {}
        if report and hasattr(report, 'agent_results'):
            agent_results = report.agent_results

        # 4. 生成 HTML
        html = _build_html_report(
            profiling_path,
            top_kernels,
            fusion_opportunities,
            info,
            agent_results
        )

        # 4. 确定输出路径
        if output:
            output_path = Path(output)
            # 如果是目录，生成文件名
            if output_path.is_dir():
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_path = output_path / f"npu_mfu_report_{timestamp}.html"
            # 确保有正确的扩展名
            elif output_path.suffix != '.html':
                output_path = output_path.with_suffix('.html')
        else:
            # 默认输出到当前目录
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = Path(f"npu_mfu_report_{timestamp}.html")

        # 5. 保存文件
        output_path.write_text(html, encoding="utf-8")

        return output_path

    except Exception as e:
        logger.error(f"HTML 报告生成失败: {e}", exc_info=True)
        return None


def _generate_html_report_content(profiling_path: str, report: Any = None) -> str:
    """
    生成 HTML 格式报告内容（使用 ReportGenerator）

    这是新版 HTML 生成函数，包含 MFU 指标。

    Args:
        profiling_path: Profiling 数据路径
        report: AnalysisReport 对象

    Returns:
        HTML 内容字符串
    """
    from src.data_loader.profiling_loader import ProfilingLoader
    from src.data_loader.data_summarizer import DataSummarizer
    from src.report.report_generator import ReportGenerator, ReportFormat

    try:
        # 加载数据
        loader = ProfilingLoader(profiling_path)
        summarizer = DataSummarizer(loader)
        profiling_summary = summarizer.summarize()

        # 提取 Agent 分析结果
        agent_results = {}
        advisor_report = None
        mfu_metrics = None
        roofline_analysis = None

        if report and hasattr(report, 'agent_results'):
            agent_results = report.agent_results
        if report and hasattr(report, 'mfu_metrics'):
            mfu_metrics = report.mfu_metrics
        if report and hasattr(report, 'roofline_analysis'):
            roofline_analysis = report.roofline_analysis

        # 使用 ReportGenerator 生成完整的 HTML 报告
        report_generator = ReportGenerator()

        # 生成 HTML 报告
        html = report_generator.generate_from_analysis(
            profiling_path=profiling_path,
            profiling_summary=profiling_summary,
            agent_results=agent_results,
            advisor_report=advisor_report,
            mfu_metrics=mfu_metrics,
            roofline_analysis=roofline_analysis,
            format=ReportFormat.HTML,
        )

        return html

    except Exception as e:
        logger.error(f"HTML 报告内容生成失败: {e}", exc_info=True)
        # 返回简单的错误 HTML
        return f"""
        <!DOCTYPE html>
        <html>
        <head><title>NPU MFU 分析报告</title></head>
        <body>
            <h1>报告生成失败</h1>
            <p>错误: {e}</p>
        </body>
        </html>
        """


def _build_html_report(profiling_path: str, top_kernels: list, fusion_opportunities: list, info, agent_results: dict = None) -> str:
    """构建 HTML 报告内容"""
    from datetime import datetime

    # 计算统计数据
    total_dur = sum(k.get("dur", 0) for k in top_kernels[:15])

    # 算子类型统计
    op_types = {}
    for kernel in top_kernels:
        name = kernel.get("name", "")
        if "matmul" in name.lower():
            op_types["MatMul"] = op_types.get("MatMul", 0) + kernel.get("dur", 0)
        elif "attention" in name.lower() or "attn" in name.lower():
            op_types["Attention"] = op_types.get("Attention", 0) + kernel.get("dur", 0)
        elif "flash" in name.lower():
            op_types["FlashAttention"] = op_types.get("FlashAttention", 0) + kernel.get("dur", 0)
        elif "norm" in name.lower():
            op_types["Norm"] = op_types.get("Norm", 0) + kernel.get("dur", 0)
        elif "add" in name.lower():
            op_types["Add"] = op_types.get("Add", 0) + kernel.get("dur", 0)
        elif "gelu" in name.lower() or "relu" in name.lower() or "silu" in name.lower():
            op_types["Activation"] = op_types.get("Activation", 0) + kernel.get("dur", 0)
        elif "cast" in name.lower():
            op_types["Cast"] = op_types.get("Cast", 0) + kernel.get("dur", 0)
        else:
            op_types["Other"] = op_types.get("Other", 0) + kernel.get("dur", 0)

    total_type_dur = sum(dur for dur in op_types.values())

    # 生成 Top 15 表格行
    table_rows = []
    for i, kernel in enumerate(top_kernels[:15], 1):
        name = kernel.get("name", "unknown")[:60]
        dur_ms = kernel.get("dur", 0) / 1000
        ratio = (kernel.get("dur", 0) / total_dur * 100) if total_dur > 0 else 0
        table_rows.append(f'                <tr><td>{i}</td><td><code>{name}</code></td><td>{dur_ms:.2f}</td><td>{ratio:.1f}%</td></tr>')
    table_rows_html = "\n".join(table_rows)

    # 生成算子类型分布
    type_rows = []
    sorted_types = sorted(op_types.items(), key=lambda x: x[1], reverse=True)
    for op_type, dur in sorted_types:
        ratio = (dur / total_type_dur * 100) if total_type_dur > 0 else 0
        type_rows.append(f'                <li><strong>{op_type}</strong>: {dur/1000:.2f} ms ({ratio:.1f}%)</li>')
    type_rows_html = "\n".join(type_rows)

    # 生成融合机会卡片
    fusion_cards = []
    for i, opp in enumerate(fusion_opportunities, 1):
        priority_class = "high-priority" if opp.end_to_end_speedup >= 1.05 else "medium-priority"
        speedup_percent = (opp.end_to_end_speedup - 1) * 100

        ops_list = "\n".join([
            f'                    <li><code>{op.get("name", "unknown")}</code>: {op.get("dur", 0)/1000:.2f} ms</li>'
            for op in opp.current_ops[:5]
        ])
        if len(opp.current_ops) > 5:
            ops_list += f'\n                    <li><em>... 还有 {len(opp.current_ops) - 5} 个算子</em></li>'

        ascend_op_line = f"                <p><strong>昇腾算子</strong>: <code>{opp.ascend_op}</code></p>\n" if opp.ascend_op else ""

        fusion_cards.append(f'''
            <div class="card {priority_class}">
                <h3>{i}. {opp.name}</h3>
                <p><strong>类型</strong>: {opp.opportunity_type} | <strong>复杂度</strong>: {opp.complexity}</p>
                <p><strong>算子加速</strong>: {opp.estimated_speedup:.1f}x | <strong>端到端加速</strong>: {speedup_percent:.1f}%</p>
                <p><strong>时间占比</strong>: {opp.time_proportion:.1%} | <strong>内存节省</strong>: {opp.memory_saving*100:.0f}%</p>
{ascend_op_line}                <p><strong>实现方式</strong>: {opp.implementation}</p>
                <p><strong>涉及算子</strong> ({len(opp.current_ops)} 个):</p>
                <ul>
{ops_list}
                </ul>
            </div>''')

    fusion_cards_html = "\n".join(fusion_cards)

    html = f'''<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>NPU MFU 性能分析报告</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            line-height: 1.6;
            color: #333;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
        }}

        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 12px;
            box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
            overflow: hidden;
        }}

        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px;
            text-align: center;
        }}

        .header h1 {{
            font-size: 2.5em;
            margin-bottom: 10px;
            text-shadow: 0 2px 4px rgba(0,0,0,0.2);
        }}

        .header .meta {{
            font-size: 0.9em;
            opacity: 0.9;
        }}

        .content {{
            padding: 40px;
        }}

        .section {{
            margin-bottom: 40px;
        }}

        .section h2 {{
            font-size: 1.8em;
            color: #667eea;
            border-bottom: 2px solid #667eea;
            padding-bottom: 10px;
            margin-bottom: 20px;
        }}

        .section h3 {{
            font-size: 1.4em;
            color: #764ba2;
            margin-bottom: 15px;
            margin-top: 25px;
        }}

        .card {{
            background: #f8f9fa;
            border-left: 4px solid #667eea;
            padding: 20px;
            margin-bottom: 20px;
            border-radius: 8px;
        }}

        .card.high-priority {{
            border-left-color: #f59e0b;
            background: #fff8f0;
        }}

        .card.medium-priority {{
            border-left-color: #10b981;
            background: #f0fdf4;
        }}

        .card h3 {{
            margin-top: 0;
            color: #333;
        }}

        .stat-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}

        .stat-box {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
        }}

        .stat-box .value {{
            font-size: 2em;
            font-weight: bold;
            margin-bottom: 5px;
        }}

        .stat-box .label {{
            font-size: 0.9em;
            opacity: 0.9;
        }}

        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}

        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #e5e7eb;
        }}

        th {{
            background: #f9fafb;
            font-weight: 600;
            color: #374151;
        }}

        tr:hover {{
            background: #f9fafb;
        }}

        code {{
            background: #f3f4f6;
            padding: 2px 6px;
            border-radius: 4px;
            font-size: 0.9em;
        }}

        ul {{
            margin: 10px 0;
            padding-left: 20px;
        }}

        li {{
            margin: 5px 0;
        }}

        .footer {{
            background: #f9fafb;
            padding: 20px;
            text-align: center;
            color: #6b7280;
            font-size: 0.9em;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>NPU MFU 性能分析报告</h1>
            <div class="meta">
                <p>生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                <p>数据路径: <code>{profiling_path}</code></p>
            </div>
        </div>

        <div class="content">
            <div class="section">
                <h2>1. 概述</h2>
                <p>本报告基于真实 NPU Profiling 数据，分析算子性能、MFU（Model FLOPS Utilization）以及<strong>算子融合优化机会</strong>。</p>

                <div class="stat-grid">
                    <div class="stat-box">
                        <div class="value">{info.data_type}</div>
                        <div class="label">数据类型</div>
                    </div>
                    <div class="stat-box">
                        <div class="value">{info.framework or 'Unknown'}</div>
                        <div class="label">框架</div>
                    </div>
                    <div class="stat-box">
                        <div class="value">{len(top_kernels)}</div>
                        <div class="label">算子数量</div>
                    </div>
                    <div class="stat-box">
                        <div class="value">{len(fusion_opportunities)}</div>
                        <div class="label">融合机会</div>
                    </div>
                </div>
            </div>

            <div class="section">
                <h2>2. 算子性能分析</h2>
                <h3>Top 15 耗时算子</h3>
                <table>
                    <thead>
                        <tr>
                            <th>排名</th>
                            <th>算子名称</th>
                            <th>耗时 (ms)</th>
                            <th>占比</th>
                        </tr>
                    </thead>
                    <tbody>
{table_rows_html}
                    </tbody>
                </table>

                <h3>算子类型分布</h3>
                <ul>
{type_rows_html}
                </ul>
            </div>

            <div class="section">
                <h2>3. 算子融合机会分析</h2>
                <p>检测到 <strong>{len(fusion_opportunities)}</strong> 个融合机会，按端到端加速效果排序：</p>
{fusion_cards_html}
            </div>
'''

    # 添加 Agent 详细分析章节
    if agent_results:
        html += '''
            <div class="section">
                <h2>4. Agent 详细分析</h2>
'''

        # Timeline Agent 分析
        if 'timeline' in agent_results:
            result = agent_results['timeline']
            analysis_text = result.raw_response if hasattr(result, 'raw_response') and result.raw_response else (result.summary if hasattr(result, 'summary') else '分析失败')
            html += f'''
                <h3>4.1 Timeline 分析</h3>
                <div class="card">
                    <pre style="white-space: pre-wrap; font-family: monospace;">{analysis_text[:2000]}</pre>
                </div>
'''

        # Operator Agent 分析
        if 'operator' in agent_results:
            result = agent_results['operator']
            analysis_text = result.raw_response if hasattr(result, 'raw_response') and result.raw_response else (result.summary if hasattr(result, 'summary') else '分析失败')
            html += f'''
                <h3>4.2 算子分析</h3>
                <div class="card">
                    <pre style="white-space: pre-wrap; font-family: monospace;">{analysis_text[:2000]}</pre>
                </div>
'''

        # Memory Agent 分析
        if 'memory' in agent_results:
            result = agent_results['memory']
            analysis_text = result.raw_response if hasattr(result, 'raw_response') and result.raw_response else (result.summary if hasattr(result, 'summary') else '分析失败')
            html += f'''
                <h3>4.3 内存分析</h3>
                <div class="card">
                    <pre style="white-space: pre-wrap; font-family: monospace;">{analysis_text[:2000]}</pre>
                </div>
'''

        # Communication Agent 分析
        if 'communication' in agent_results:
            result = agent_results['communication']
            analysis_text = result.raw_response if hasattr(result, 'raw_response') and result.raw_response else (result.summary if hasattr(result, 'summary') else '分析失败')
            html += f'''
                <h3>4.4 通信分析</h3>
                <div class="card">
                    <pre style="white-space: pre-wrap; font-family: monospace;">{analysis_text[:2000]}</pre>
                </div>
'''

        # Jitter Agent 分析
        if 'jitter' in agent_results:
            result = agent_results['jitter']
            analysis_text = result.raw_response if hasattr(result, 'raw_response') and result.raw_response else (result.summary if hasattr(result, 'summary') else '分析失败')
            html += f'''
                <h3>4.5 抖动分析</h3>
                <div class="card">
                    <pre style="white-space: pre-wrap; font-family: monospace;">{analysis_text[:2000]}</pre>
                </div>
'''

        html += '''
            </div>
'''

    # 更新后续章节编号
    next_section_num = 5 if agent_results else 4
    html += f'''
            <div class="section">
                <h2>{next_section_num}. 优化建议</h2>
                <h3>{next_section_num}.1 高优先级优化 (端到端加速 > 5%)</h3>
'''

    # 添加高优先级优化建议
    high_priority = [opp for opp in fusion_opportunities if opp.end_to_end_speedup >= 1.05]
    if high_priority:
        for i, opp in enumerate(high_priority, 1):
            speedup_percent = (opp.end_to_end_speedup - 1) * 100
            html += f'''
                <div class="card high-priority">
                    <h3>{i}. {opp.name}</h3>
                    <p><strong>预期收益</strong>: 端到端加速 {speedup_percent:.1f}% (时间占比 {opp.time_proportion:.1%})</p>
                    <p><strong>实现难度</strong>: {opp.complexity}</p>
                    <p><strong>操作方式</strong>: {opp.implementation}</p>
                </div>'''
    else:
        html += '<p>暂无高优先级优化建议。</p>'

    # 添加通用优化建议
    html += f'''
                <h3>{next_section_num}.2 优化方向建议</h3>
                <ul>
                    <li><strong>算子融合</strong>: 使用昇腾原生融合算子（如 FlashAttention、FusedMatMulBiasAct）</li>
                    <li><strong>数据类型优化</strong>: 统一使用 FP16/BF16 数据类型，减少不必要的数据类型转换</li>
                    <li><strong>内存布局优化</strong>: 减少 reshape/transpose 操作，优化张量内存布局</li>
                    <li><strong>数据加载优化</strong>: 增加 DataLoader workers，使用 pin_memory</li>
                </ul>
            </div>

            <div class="section">
                <h2>{next_section_num + 1}. 技术细节</h2>
                <h3>{next_section_num + 1}.1 融合分析方法</h3>
                <ul>
                    <li><strong>全局分析</strong>: 分析所有算子，不局限于 Top 10</li>
                    <li><strong>端到端加速计算</strong>: T_new = T_total - t_op + t_op/speedup</li>
                    <li><strong>融合模式匹配</strong>: 基于算子名称和执行序列</li>
                </ul>

                <h3>{next_section_num + 1}.2 昇腾融合算子映射</h3>
                <table>
                    <thead>
                        <tr>
                            <th>融合模式</th>
                            <th>昇腾算子</th>
                            <th>性能提升</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr><td>FlashAttention</td><td><code>aclnnFlashAttentionScore</code></td><td>5x+</td></tr>
                        <tr><td>MatMul+Bias+Act</td><td><code>aclnnFusedMatMulBiasAct</code></td><td>20-30%</td></tr>
                        <tr><td>Norm+Residual</td><td><code>aclnnAddRmsNorm</code></td><td>20%+</td></tr>
                        <tr><td>QKV Projection</td><td><code>aclnnFusedQKVProjection</code></td><td>2x</td></tr>
                        <tr><td>MoE 融合</td><td><code>aclnnGroupedMatmulV4</code></td><td>MoE 优化</td></tr>
                    </tbody>
                </table>
            </div>
        </div>

        <div class="footer">
            <p>本报告由 npu-mfu-analyzer 自动生成</p>
            <p>Profiling 工具: msprof | 分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
    </div>
</body>
</html>'''

    return html


@cli.command()
@click.argument("profiling_path", type=click.Path(exists=True))
def info(profiling_path: str):
    """
    显示 Profiling 数据信息
    
    PROFILING_PATH: Profiling 数据目录路径
    """
    from src.data_loader.profiling_loader import ProfilingLoader
    
    click.echo(f"📊 检测 Profiling 数据: {profiling_path}")
    
    loader = ProfilingLoader(profiling_path)
    info = loader.detect()
    
    click.echo(f"\n数据类型: {info.data_type}")
    click.echo(f"框架: {info.framework}")
    click.echo(f"Rank 数量: {info.rank_count}")
    click.echo(f"Timeline 数据: {'✓' if info.has_timeline else '✗'}")
    click.echo(f"Memory 数据: {'✓' if info.has_memory else '✗'}")
    click.echo(f"Communication 数据: {'✓' if info.has_communication else '✗'}")
    
    if info.db_paths:
        click.echo(f"\nDB 文件 ({len(info.db_paths)}):")
        for path in info.db_paths[:5]:
            click.echo(f"  - {path}")
        if len(info.db_paths) > 5:
            click.echo(f"  ... 还有 {len(info.db_paths) - 5} 个文件")
    
    if info.json_paths:
        click.echo(f"\nJSON 文件 ({len(info.json_paths)}):")
        for path in info.json_paths[:5]:
            click.echo(f"  - {path}")
        if len(info.json_paths) > 5:
            click.echo(f"  ... 还有 {len(info.json_paths) - 5} 个文件")


@cli.command()
@click.argument("profiling_path", type=click.Path(exists=True))
@click.option("--max-steps", type=int, default=10, help="采样的 Step 数量")
def summary(profiling_path: str, max_steps: int):
    """
    生成 Profiling 数据摘要（不调用 LLM）
    
    PROFILING_PATH: Profiling 数据目录路径
    """
    from src.data_loader.profiling_loader import ProfilingLoader
    from src.data_loader.data_summarizer import DataSummarizer
    
    click.echo(f"📊 生成数据摘要: {profiling_path}")
    
    loader = ProfilingLoader(profiling_path)
    summarizer = DataSummarizer(loader)
    
    summary = summarizer.summarize(max_sample_steps=max_steps)
    
    click.echo("\n" + "=" * 60)
    click.echo(summary.to_prompt_text())
    click.echo("=" * 60)


@cli.command()
def version():
    """显示版本信息"""
    click.echo("NPU MFU Analyzer v0.1.0")
    click.echo("昇腾 NPU 大模型训练性能分析工具")


@cli.command()
@click.argument("profiling_path", type=click.Path(exists=True))
@click.option(
    "--output", "-o",
    type=click.Path(),
    default="./generated_kernels",
    help="融合算子输出目录（默认: ./generated_kernels）"
)
@click.option(
    "--backend", "-b",
    type=click.Choice(["openai", "claude", "ollama", "deepseek", "mock"]),
    default="claude",
    help="LLM 后端（默认: claude）"
)
@click.option(
    "--model", "-m",
    type=str,
    default=None,
    help="LLM 模型名称（Claude 默认: GLM-4.7）"
)
@click.option(
    "--min-speedup",
    type=float,
    default=1.05,
    help="最小加速比阈值（默认: 1.05）"
)
@click.option(
    "--complexity",
    type=click.Choice(["低", "中等", "高"]),
    default="高",
    help="最大实现复杂度（默认: 高）"
)
@click.option(
    "--skip-native/--include-native",
    default=True,
    help="跳过/包含 昇腾已有融合算子（默认: 跳过）"
)
@click.option(
    "--timeout",
    type=int,
    default=300,
    help="单个算子生成超时时间（秒，默认: 300）"
)
@click.option(
    "--max-concurrent",
    type=int,
    default=3,
    help="最大并发生成数（默认: 3）"
)
@click.option(
    "--verbose", "-v",
    is_flag=True,
    help="详细输出"
)
def generate(
    profiling_path: str,
    output: str,
    backend: str,
    model: Optional[str],
    min_speedup: float,
    complexity: str,
    skip_native: bool,
    timeout: int,
    max_concurrent: int,
    verbose: bool
):
    """
    分析 Profiling 数据并生成融合算子代码（AIKG 工作流）

    完整的自动化工作流：
    1. 分析 Profiling 数据
    2. 检测融合机会
    3. 调用 AIKG 生成 Triton-Ascend 代码
    4. 保存生成的代码和脚本

    PROFILING_PATH: Profiling 数据目录路径

    示例:
        npu-analyzer generate /path/to/profiling
        npu-analyzer generate /path/to/profiling -b claude -m GLM-4.7
        npu-analyzer generate /path/to/profiling -o ./kernels --min-speedup 1.1
        npu-analyzer generate /path/to/profiling --complexity 低 --skip-native
    """
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    from src.cli.aikg_workflow import AIKGWorkflow

    click.echo(click.style("🚀 启动 AIKG 融合算子生成工作流", fg="cyan", bold=True))
    click.echo(f"   Profiling 数据: {profiling_path}")
    click.echo(f"   LLM 后端: {backend}")
    click.echo(f"   输出目录: {output}")
    click.echo(f"   最小加速比: {min_speedup}x")
    click.echo(f"   最大复杂度: {complexity}")
    click.echo(f"   跳过昇腾算子: {skip_native}")
    click.echo("")

    # 创建工作流
    workflow = AIKGWorkflow(
        profiling_path=profiling_path,
        output_dir=output,
        backend=backend,
        model=model,
        min_speedup=min_speedup,
        max_complexity=complexity,
        skip_native=skip_native,
        timeout=timeout,
        max_concurrent=max_concurrent,
    )

    # 执行工作流
    result = asyncio.run(workflow.run())

    # 打印结果摘要
    workflow.print_summary(result)

    # 根据结果决定退出码
    if not result.success:
        sys.exit(1)


@cli.command()
@click.argument("profiling_path", type=click.Path(exists=True))
@click.option(
    "--output", "-o",
    type=click.Path(),
    default="./integration_output",
    help="集成输出目录（默认: ./integration_output）"
)
@click.option(
    "--patterns",
    type=str,
    default="add,mul,slice,strided",
    help="融合模式列表（逗号分隔，默认: add,mul,slice,strided）"
)
@click.option(
    "--time-window",
    type=int,
    default=100,
    help="时间窗口（微秒，默认: 100）"
)
@click.option(
    "--limit",
    type=int,
    default=50,
    help="分析的最大调用数（默认: 50）"
)
@click.option(
    "--verbose", "-v",
    is_flag=True,
    help="详细输出"
)
def integrate(
    profiling_path: str,
    output: str,
    patterns: str,
    time_window: int,
    limit: int,
    verbose: bool
):
    """
    分析 Profiling 数据并生成融合算子集成方案

    完整的自动化工作流：
    1. 分析 Trace 数据中的 API 调用栈
    2. 定位需要替换的算子调用源代码位置
    3. 生成自定义融合算子代码
    4. 生成集成到训练脚本的补丁和指南

    PROFILING_PATH: Profiling 数据目录路径

    示例:
        npu-analyzer integrate /path/to/profiling
        npu-analyzer integrate /path/to/profiling -o ./patches
        npu-analyzer integrate /path/to/profiling --patterns add,mul --time-window 50
    """
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    from src.cli.integration_workflow import IntegrationWorkflow

    click.echo(click.style("🔧 启动融合算子集成工作流", fg="cyan", bold=True))
    click.echo(f"   Profiling 数据: {profiling_path}")
    click.echo(f"   输出目录: {output}")
    click.echo(f"   融合模式: {patterns}")
    click.echo(f"   时间窗口: {time_window} μs")
    click.echo("")

    try:
        # 解析融合模式
        fusion_patterns = [p.strip() for p in patterns.split(",")]

        # 创建工作流
        workflow = IntegrationWorkflow(
            profiling_path=profiling_path,
            output_dir=output
        )

        # 执行工作流
        result = workflow.run(
            fusion_patterns=fusion_patterns,
            time_window_ns=time_window * 1000,  # 转换为纳秒
            limit=limit
        )

        # 输出结果
        if result["success"]:
            click.echo("\n" + "=" * 70)
            click.echo(click.style("📊 集成工作流结果摘要", fg="cyan", bold=True))
            click.echo("=" * 70)
            click.echo(click.style(f"✓ 找到融合模式: {result['fusion_patterns_found']} 个", fg="green"))
            click.echo(click.style(f"✓ 生成算子代码: {result['operators_generated']} 个", fg="green"))
            click.echo(click.style(f"✓ 输出目录: {result['output_dir']}", fg="blue"))
            click.echo("")
            click.echo(click.style("📁 生成的文件:", fg="yellow", bold=True))
            click.echo(f"  - 自定义算子代码: *_operator.py")
            click.echo(f"  - 集成补丁: *_patch.txt")
            click.echo(f"  - 集成指南: {result['integration_guide']}")
            click.echo("")
            click.echo(click.style("💡 下一步:", fg="yellow", bold=True))
            click.echo(f"   1. 查看集成指南: cat {result['integration_guide']}")
            click.echo(f"   2. 复制算子代码到项目: cp {result['output_dir']}/*_operator.py /path/to/project/")
            click.echo(f"   3. 按照指南修改训练脚本")
            click.echo("=" * 70)
        else:
            click.echo(click.style(f"❌ 工作流失败: {result.get('error', 'Unknown error')}", fg="red"))
            sys.exit(1)

    except Exception as e:
        click.echo(click.style(f"❌ 发生错误: {e}", fg="red"))
        if verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


@cli.command()
@click.argument("profiling_path", type=click.Path(exists=True))
@click.option(
    "--top-n", "-n",
    type=int,
    default=20,
    help="显示 Top N 算子（默认: 20）"
)
@click.option(
    "--sort-by", "-s",
    type=click.Choice(["duration", "cube_util", "l2_hit", "stall_rate", "name"]),
    default="duration",
    help="排序方式（默认: duration）"
)
@click.option(
    "--show-all", "-a",
    is_flag=True,
    help="显示所有指标"
)
@click.option(
    "--output", "-o",
    type=click.Path(),
    help="输出到文件（CSV 或 Markdown）"
)
@click.option(
    "--severity",
    type=click.Choice(["all", "critical", "high", "medium", "low"]),
    default="all",
    help="按严重度筛选（默认: all）"
)
def analyze_aic(
    profiling_path: str,
    top_n: int,
    sort_by: str,
    show_all: bool,
    output: Optional[str],
    severity: str
):
    """
    分析 AIC Metrics 硬件指标数据

    解析 msprof op --aic-metrics 生成的详细硬件指标，分析算子瓶颈。
    支持 Cube 利用率、L2 缓存命中率、流水线利用率等指标分析。

    PROFILING_PATH: Profiling 数据目录路径

    示例:
        npu-analyzer analyze-aic /path/to/profiling
        npu-analyzer analyze-aic /path/to/profiling --top-n 30 --sort-by cube_util
        npu-analyzer analyze-aic /path/to/profiling --severity critical --output report.md
    """
    from src.data_loader.profiling_loader import ProfilingLoader
    from src.data_loader.aic_metrics import (
        CRITICAL_THRESHOLD,
        HIGH_THRESHOLD,
        BOTTLENECK_COMPUTE,
        BOTTLENECK_MEMORY,
        BOTTLENECK_PIPELINE,
    )

    click.echo(click.style("🔍 分析 AIC Metrics 硬件指标", fg="cyan", bold=True))
    click.echo(f"   数据路径: {profiling_path}")
    click.echo(f"   Top N: {top_n}")
    click.echo(f"   排序: {sort_by}")
    click.echo("")

    # 检查 AIC metrics 是否可用
    import glob
    opprof_dirs = glob.glob(str(Path(profiling_path) / "OPPROF_*"), recursive=True)
    if not opprof_dirs:
        click.echo(click.style("⚠️  未找到 AIC metrics 数据", fg="yellow"))
        click.echo("   请使用 msprof op --aic-metrics 生成数据")
        click.echo("")
        click.echo("   示例命令:")
        click.echo("     msprof op --aic-metrics --output /path/to/profiling")
        sys.exit(1)

    # 加载 AIC metrics
    try:
        loader = ProfilingLoader(profiling_path)
        aic_metrics = loader.get_aic_metrics()

        if not aic_metrics:
            click.echo(click.style("⚠️  未找到有效的 AIC metrics 数据", fg="yellow"))
            sys.exit(1)

        click.echo(click.style(f"✅ 加载了 {len(aic_metrics)} 个算子的 AIC metrics", fg="green"))
        click.echo("")

        # 转换为列表以便排序
        metrics_list = []
        for op_name, metrics in aic_metrics.items():
            cube_util = metrics.arithmetic.cube_utilization if metrics.arithmetic else 100.0
            l2_hit = metrics.memory.l2_cache_hit_rate if metrics.memory else 100.0
            stall_rate = metrics.pipeline.stall_rate if metrics.pipeline else 0.0

            # 确定瓶颈类型和严重度
            if cube_util < CRITICAL_THRESHOLD or l2_hit < CRITICAL_THRESHOLD:
                sev = "critical"
            elif stall_rate > 50:
                sev = "high"
            elif cube_util < HIGH_THRESHOLD or l2_hit < HIGH_THRESHOLD:
                sev = "medium"
            else:
                sev = "low"

            metrics_list.append({
                "name": op_name,
                "metrics": metrics,
                "cube_util": cube_util,
                "l2_hit": l2_hit,
                "stall_rate": stall_rate,
                "duration": metrics.duration_us,
                "severity": sev
            })

        # 按 severity 和选定的指标排序
        severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        metrics_list.sort(key=lambda x: (severity_order[x["severity"]], -x.get(sort_by, 0)))

        # 筛选严重度
        if severity != "all":
            metrics_list = [m for m in metrics_list if m["severity"] == severity]

        # 限制显示数量
        display_list = metrics_list[:top_n]

        # 输出结果
        click.echo("=" * 80)
        click.echo(click.style(f"AIC Metrics 分析结果 (Top {len(display_list)})", fg="cyan", bold=True))
        click.echo("=" * 80)

        results = []
        for i, item in enumerate(display_list, 1):
            m = item["metrics"]
            sev = item["severity"]

            # 严重度颜色
            if sev == "critical":
                sev_emoji = "🔴"
                sev_style = {"fg": "red"}
            elif sev == "high":
                sev_emoji = "🟠"
                sev_style = {"fg": "yellow"}
            elif sev == "medium":
                sev_emoji = "🟡"
                sev_style = {"fg": "yellow"}
            else:
                sev_emoji = "🟢"
                sev_style = {"fg": "green"}

            click.echo(f"\n{i}. {sev_emoji} {click.style(item['name'], bold=True)}")

            # 基本指标
            click.echo(f"   执行时间: {m.duration_us:.2f} μs")
            click.echo(f"   类型: {m.op_type}")

            if m.arithmetic:
                click.echo(f"   算术单元:")
                cube_color = "red" if item['cube_util'] < CRITICAL_THRESHOLD else "green"
                click.echo(click.style(f"     Cube:   {item['cube_util']:.1f}%", fg=cube_color))
                click.echo(f"     Vector: {m.arithmetic.vector_utilization:.1f}%")
                click.echo(f"     Scalar: {m.arithmetic.scalar_utilization:.1f}%")

            if m.memory:
                click.echo(f"   内存:")
                l2_color = "red" if item['l2_hit'] < CRITICAL_THRESHOLD else "green"
                click.echo(click.style(f"     L2 命中率: {item['l2_hit']:.1f}%", fg=l2_color))
                click.echo(f"     UB 使用率: {m.memory.ub_usage:.1f}%")
                click.echo(f"     L0 使用率: {m.memory.l0_usage:.1f}%")

            if m.pipeline:
                click.echo(f"   流水线:")
                click.echo(f"     利用率: {m.pipeline.pipe_utilization:.1f}%")
                stall_color = "red" if item['stall_rate'] > 50 else "green"
                click.echo(click.style(f"     停顿率: {item['stall_rate']:.1f}%", fg=stall_color))

            # 瓶颈诊断
            if item['cube_util'] < CRITICAL_THRESHOLD:
                click.echo(click.style(f"   ⚠️  计算瓶颈: Cube 利用率过低", fg="red"))
            elif item['l2_hit'] < CRITICAL_THRESHOLD:
                click.echo(click.style(f"   ⚠️  内存瓶颈: L2 缓存命中率过低", fg="red"))
            elif item['stall_rate'] > 50:
                click.echo(click.style(f"   ⚠️  流水线瓶颈: 停顿率过高", fg="yellow"))

            results.append(item)

        # 统计摘要
        click.echo("\n" + "=" * 80)
        click.echo(click.style("瓶颈统计", fg="cyan", bold=True))
        click.echo("=" * 80)

        critical_count = sum(1 for m in metrics_list if m["severity"] == "critical")
        high_count = sum(1 for m in metrics_list if m["severity"] == "high")
        medium_count = sum(1 for m in metrics_list if m["severity"] == "medium")
        low_count = sum(1 for m in metrics_list if m["severity"] == "low")

        click.echo(f"🔴 严重 (critical): {critical_count}")
        click.echo(f"🟠 高 (high):       {high_count}")
        click.echo(f"🟡 中 (medium):    {medium_count}")
        click.echo(f"🟢 低 (low):       {low_count}")

        # 输出到文件
        if output:
            output_path = Path(output)
            output_str = ""

            if output_path.suffix == ".csv":
                # CSV 格式
                import csv
                import io

                output = io.StringIO()
                writer = csv.writer(output)
                writer.writerow([
                    "算子名称", "严重度", "执行时间(μs)", "Cube利用率(%)",
                    "Vector利用率(%)", "Scalar利用率(%)", "L2命中率(%)",
                    "UB使用率(%)", "L0使用率(%)", "流水线利用率(%)", "停顿率(%)"
                ])

                for item in results:
                    m = item["metrics"]
                    writer.writerow([
                        item["name"], item["severity"], f"{m.duration_us:.2f}",
                        f"{item['cube_util']:.1f}",
                        f"{m.arithmetic.vector_utilization:.1f}" if m.arithmetic else "",
                        f"{m.arithmetic.scalar_utilization:.1f}" if m.arithmetic else "",
                        f"{item['l2_hit']:.1f}",
                        f"{m.memory.ub_usage:.1f}" if m.memory else "",
                        f"{m.memory.l0_usage:.1f}" if m.memory else "",
                        f"{m.pipeline.pipe_utilization:.1f}" if m.pipeline else "",
                        f"{item['stall_rate']:.1f}"
                    ])

                output_str = output.getvalue()
            else:
                # Markdown 格式
                output_str = f"# AIC Metrics 分析报告\n\n"
                output_str += f"数据路径: `{profiling_path}`\n\n"
                output_str += f"分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"

                output_str += "## 瓶颈统计\n\n"
                output_str += f"- 🔴 严重: {critical_count}\n"
                output_str += f"- 🟠 高: {high_count}\n"
                output_str += f"- 🟡 中: {medium_count}\n"
                output_str += f"- 🟢 低: {low_count}\n\n"

                output_str += f"## Top {len(results)} 算子详情\n\n"

                for i, item in enumerate(results, 1):
                    m = item["metrics"]
                    sev_emoji = {"critical": "🔴", "high": "🟠", "medium": "🟡", "low": "🟢"}[item["severity"]]

                    output_str += f"### {i}. {sev_emoji} {item['name']}\n\n"
                    output_str += f"- **严重度**: {item['severity']}\n"
                    output_str += f"- **执行时间**: {m.duration_us:.2f} μs\n"
                    output_str += f"- **类型**: {m.op_type}\n\n"

                    if m.arithmetic:
                        output_str += "**算术单元**:\n"
                        output_str += f"- Cube: {item['cube_util']:.1f}%\n"
                        output_str += f"- Vector: {m.arithmetic.vector_utilization:.1f}%\n"
                        output_str += f"- Scalar: {m.arithmetic.scalar_utilization:.1f}%\n\n"

                    if m.memory:
                        output_str += "**内存**:\n"
                        output_str += f"- L2 命中率: {item['l2_hit']:.1f}%\n"
                        output_str += f"- UB 使用率: {m.memory.ub_usage:.1f}%\n"
                        output_str += f"- L0 使用率: {m.memory.l0_usage:.1f}%\n\n"

                    if m.pipeline:
                        output_str += "**流水线**:\n"
                        output_str += f"- 利用率: {m.pipeline.pipe_utilization:.1f}%\n"
                        output_str += f"- 停顿率: {item['stall_rate']:.1f}%\n\n"

                output_str += "\n---\n\n*本报告由 npu-mfu-analyzer 自动生成*"

            output_path.write_text(output_str, encoding="utf-8")
            click.echo(f"\n📄 报告已保存到: {output_path}")

    except Exception as e:
        click.echo(click.style(f"❌ 发生错误: {e}", fg="red"))
        import traceback
        traceback.print_exc()
        sys.exit(1)


@cli.command()
@click.argument("path_a", type=click.Path(exists=True))
@click.argument("path_b", type=click.Path(exists=True))
@click.option(
    "--output", "-o",
    type=click.Path(),
    help="输出报告路径（默认输出到终端）"
)
@click.option(
    "--backend", "-b",
    type=click.Choice(["openai", "claude", "ollama", "deepseek", "mock"]),
    default="mock",
    help="LLM 后端（默认: mock，使用规则引擎）"
)
@click.option(
    "--model", "-m",
    type=str,
    default=None,
    help="LLM 模型名称"
)
@click.option(
    "--format", "-f",
    type=click.Choice(["markdown", "html", "md"]),
    default="markdown",
    help="报告格式（默认: markdown）"
)
@click.option(
    "--label-a",
    type=str,
    default=None,
    help="基准版本标签（如: v2.0-升级前）"
)
@click.option(
    "--label-b",
    type=str,
    default=None,
    help="当前版本标签（如: v2.1-升级后）"
)
@click.option(
    "--force",
    is_flag=True,
    help="跳过相似度检查，强制对比"
)
@click.option(
    "--verbose", "-v",
    is_flag=True,
    help="详细输出"
)
def compare(
    path_a: str,
    path_b: str,
    output: Optional[str],
    backend: str,
    model: Optional[str],
    format: str,
    label_a: Optional[str],
    label_b: Optional[str],
    force: bool,
    verbose: bool
):
    """
    对比两个 Profiling 数据

    比较两次 Profiling 的差异，分析性能变化的根本原因。

    \b
    典型场景:
    - 软件版本升级前后的性能对比
    - 不同并行策略的性能对比
    - 参数调优前后的性能对比

    \b
    示例:
        npu-analyzer compare /path/to/profiling_before /path/to/profiling_after
        npu-analyzer compare /p/v1 /p/v2 --label-a "CANN 8.0" --label-b "CANN 8.1" -b openai
        npu-analyzer compare /p/tp4 /p/tp8 --label-a "TP=4" --label-b "TP=8" --force
    """
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # 默认标签
    if not label_a:
        label_a = f"基准版本 (A): {Path(path_a).name}"
    if not label_b:
        label_b = f"当前版本 (B): {Path(path_b).name}"

    click.echo(click.style("🔄 Profiling 对比分析", fg="cyan", bold=True))
    click.echo(f"   A (基准): {path_a}")
    click.echo(f"   B (当前): {path_b}")
    click.echo(f"   LLM 后端: {backend}")
    click.echo(f"   强制对比: {'是' if force else '否'}")
    click.echo("")

    try:
        from src.analyzers.comparison_orchestrator import ComparisonOrchestrator, ComparisonReport
        from src.llm.llm_interface import LLMConfig
        from src.report.report_generator import ReportFormat

        # 配置 LLM
        llm_config = LLMConfig(backend=backend)
        if model:
            llm_config.model = model

        # 确定输出格式
        if format in ("html",):
            output_format = ReportFormat.HTML
        else:
            output_format = ReportFormat.MARKDOWN

        # 创建 Orchestrator
        orchestrator = ComparisonOrchestrator(
            path_a=path_a,
            path_b=path_b,
            label_a=label_a,
            label_b=label_b,
            llm_config=llm_config,
            force=force,
        )

        # 执行对比
        click.echo("⏳ 正在加载和分析 Profiling 数据...")
        report = asyncio.run(orchestrator.run(output_format=output_format))

        if not report.success:
            if report.error == "NOT_COMPARABLE":
                click.echo("")
                click.echo(click.style("❌ 两个 Profiling 数据不适合对比", fg="red", bold=True))
                click.echo("")
                if report.similarity:
                    click.echo(f"   相似度评分: {report.similarity.overall_score * 100:.0f}%")
                    click.echo(f"   {report.similarity.summary}")
                    if report.similarity.warnings:
                        click.echo("")
                        for w in report.similarity.warnings:
                            click.echo(f"   ⚠️ {w}")
                click.echo("")
                click.echo(click.style(
                    "💡 提示: 使用 --force 选项可跳过相似度检查强制对比", fg="yellow"
                ))

                # 即使不可比，仍然输出报告（如果有）
                if report.final_report and output:
                    output_path = Path(output)
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    output_path.write_text(report.final_report, encoding="utf-8")
                    click.echo(f"\n📄 报告已保存到: {output_path}")

                sys.exit(1)
            else:
                click.echo(click.style(f"❌ 对比分析失败: {report.error}", fg="red"))
                sys.exit(1)

        # 输出结果
        click.echo("")
        click.echo(click.style("✅ 对比分析完成", fg="green", bold=True))
        click.echo("")

        # 显示摘要
        if report.diff:
            verdict_map = {
                "improved": ("性能提升 ✅", "green"),
                "degraded": ("性能劣化 ⚠️", "red"),
                "mixed": ("喜忧参半 ⚖️", "yellow"),
                "unchanged": ("基本不变 ➡️", "blue"),
            }
            verdict_text, color = verdict_map.get(
                report.diff.overall_verdict, ("N/A", "white")
            )
            click.echo(f"   整体判断: {click.style(verdict_text, fg=color, bold=True)}")

            if report.diff.primary_changes:
                click.echo("")
                click.echo("   主要变化:")
                for change in report.diff.primary_changes:
                    click.echo(f"   • {change}")

        if report.similarity:
            click.echo(f"\n   相似度评分: {report.similarity.overall_score * 100:.0f}%")

        if report.recommendations:
            click.echo("")
            click.echo(click.style("   优化建议:", fg="yellow", bold=True))
            for i, rec in enumerate(report.recommendations[:5], 1):
                click.echo(f"   {i}. {rec}")

        # 输出报告
        if output:
            output_path = Path(output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(report.final_report, encoding="utf-8")
            click.echo(f"\n📄 报告已保存到: {output_path}")
        elif report.final_report:
            click.echo("")
            click.echo("=" * 70)
            click.echo(report.final_report)

    except Exception as e:
        click.echo(click.style(f"❌ 发生错误: {e}", fg="red"))
        if verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


@cli.command()
@click.option(
    "--host", "-h",
    type=str,
    default="0.0.0.0",
    help="监听地址（默认 0.0.0.0）"
)
@click.option(
    "--port", "-p",
    type=int,
    default=8000,
    help="监听端口（默认 8000）"
)
@click.option(
    "--reload",
    is_flag=True,
    help="开发模式，自动重载"
)
def web(host: str, port: int, reload: bool):
    """
    启动 Web 服务

    启动后访问 http://localhost:8000 使用 Web 界面
    API 文档: http://localhost:8000/docs
    """
    try:
        import uvicorn
    except ImportError:
        click.echo(click.style("❌ 请先安装 web 依赖: pip install fastapi uvicorn", fg="red"))
        sys.exit(1)
    
    click.echo(f"🚀 启动 Web 服务: http://{host}:{port}")
    click.echo(f"📚 API 文档: http://{host}:{port}/docs")
    click.echo("按 Ctrl+C 停止服务\n")
    
    uvicorn.run(
        "src.web.app:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info"
    )


if __name__ == "__main__":
    cli()
